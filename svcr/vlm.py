"""Module 2 - Small Vision-Language Model (SmolVLM).

Two sub-stages, mirroring the architecture diagram:

  A. Visual understanding  -> VisualConcepts (people, objects, actions, gestures,
                              setting, spatial relations, caption)
  B. Language & reasoning  -> DraftReasoning (activity, relationship, intention,
                              rationale, optional answer to a user question)

The engine also exposes two primitives reused by Module 4:
  * ``generate``      - greedy, deterministic text generation
  * ``score_options`` - calibrated multiple-choice probabilities read directly from
                        the next-token distribution over option letters, averaged
                        over several option orderings to cancel position bias.

Fine-tuning is out of scope for Phase I: ``vlm.model_id`` can later point to a
fine-tuned checkpoint without code changes.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Sequence

import numpy as np

from .config import VLMConfig
from .schemas import DraftReasoning, PreprocessedFrame, VisualConcepts
from .text import (PERSON_NOUNS, dedupe, extract_spatial_relations, extract_terms, parse_count,
                   split_list)

log = logging.getLogger(__name__)

LETTERS = string.ascii_uppercase

PROMPTS = {
    "describe": (
        "Describe this image in one paragraph of three to five sentences. Focus on the people: how "
        "many there are, what they are doing, how they are interacting with each other, their "
        "gestures and facial expressions, the objects they hold or use, and the setting. Do not use "
        "lists or headings."
    ),
    "count": "How many people are visible in this image? Answer with a single number.",
    "objects": (
        "List the main objects in this image that the people are holding, using or are close to. "
        "Answer with a comma-separated list of short nouns."
    ),
    "actions": (
        "What actions are the people in this image performing? Answer with a comma-separated list "
        "of short verb phrases, for example: shaking hands, smiling, sitting at a table."
    ),
    "gestures": (
        "Describe the facial expressions, gestures and body language of the people in this image. "
        "Answer with a comma-separated list of a few words each."
    ),
    "setting": "Where does this scene take place? Answer with a short phrase.",
    "people": (
        "Who are the people in this image? Describe each person briefly by role or appearance "
        "(for example: a waiter, a young girl, a police officer). Answer with a comma-separated list."
    ),
    "activity_multi": (
        "What social interaction or activity is happening between the people in this image? "
        "Answer with a short phrase."
    ),
    "activity_single": "What is the person in this image doing? Answer with a short phrase.",
    "relationship": (
        "Based on their appearance and behaviour, what is the most likely relationship between the "
        "people in this image? Answer with a few words."
    ),
    "intention": (
        "The people in this image are {activity}. Why are they doing this? What is their most likely "
        "intention or goal? Answer with one short sentence."
    ),
    "rationale": (
        "The people in this image are {activity}. In one or two sentences, explain what you can see "
        "in the image (people, gestures, objects, setting) that shows this."
    ),
    "question": "{question} Answer briefly, then give the visual reason.",
}


# Sentences that talk about the description itself rather than the image.
_META_SENTENCE = re.compile(r"^(given the description|let'?s|here is|in summary|to summarize|overall,? the image)",
                            re.IGNORECASE)


def complete_sentences(text: str, max_sentences: int = 5) -> str:
    """Plain-prose description: no list formatting, no meta remarks, no unfinished last sentence."""
    text = re.sub(r"\*\*|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", text, flags=re.M)
    text = re.sub(r"\s*\n+\s*", " ", text).strip()
    text = re.sub(r"\s*\d+[.)]\s*(?:[\w' -]{0,40}:)?\s*$", "", text)   # dangling "2. Starfire:"
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s and not _META_SENTENCE.match(s)]
    if sentences and sentences[-1][-1] not in ".!?\"'":                # cut off by the length cap
        if len(sentences) > 1:
            sentences.pop()
        else:
            sentences[-1] += "…"
    return " ".join(sentences[:max_sentences])


def unique_people(items: list[str]) -> list[str]:
    """'children', 'ten children', 'child' -> 'children' (one entry per base noun)."""
    from .text import lemmatize
    seen: dict[str, str] = {}
    for item in items:
        words = item.split()
        if words:
            seen.setdefault(lemmatize(words[-1]), item)
    return list(seen.values())


def resolve_device(name: str = "auto") -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    import torch
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


class SmolVLMEngine:
    """Thin, deterministic wrapper around a SmolVLM checkpoint."""

    def __init__(self, cfg: VLMConfig | None = None):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.cfg = cfg or VLMConfig()
        self.model_id = self.cfg.model_id
        self.device = resolve_device(self.cfg.device)
        self.dtype = resolve_dtype(self.cfg.dtype, self.device)
        torch.manual_seed(self.cfg.seed)

        log.info("Loading %s on %s (%s) ...", self.model_id, self.device, self.dtype)
        common = dict(revision=self.cfg.revision, local_files_only=self.cfg.local_files_only)
        self.processor = AutoProcessor.from_pretrained(self.model_id, **common)
        self._configure_image_size()

        try:
            from transformers import AutoModelForImageTextToText as ModelCls
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as ModelCls
        major = int(transformers.__version__.split(".")[0])
        minor = int(transformers.__version__.split(".")[1])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = ModelCls.from_pretrained(self.model_id, **{dtype_kw: self.dtype}, **common)
        self.adapter = None
        if self.cfg.adapter_path:
            self._merge_adapter(self.cfg.adapter_path)
        self.model.to(self.device).eval()
        self.tokenizer = self.processor.tokenizer
        self._letter_ids = self._build_letter_ids()
        log.info("SmolVLM ready%s.", f" with LoRA adapter '{self.adapter}'" if self.adapter else "")

    def _merge_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter (finetune/train_vcr_lora.py) and merge it into the base weights.

        Merging makes inference exactly as fast as the base model - no extra adapter
        layers at runtime.
        """
        import json
        from pathlib import Path

        from peft import PeftModel

        from .config import PROJECT_ROOT

        path = Path(adapter_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No LoRA adapter at {path} (expected adapter_config.json). Train one with "
                "finetune/train_vcr_lora.py, or unset vlm.adapter_path to use the base model.")
        meta = json.loads((path / "adapter_config.json").read_text())
        base = meta.get("base_model_name_or_path")
        if base and base != self.model_id:
            log.warning("Adapter was trained on %s but the base model is %s - results may be poor.",
                        base, self.model_id)
        training = path / "svcr_training.json"
        if training.exists():
            edge = json.loads(training.read_text()).get("image_longest_edge")
            if edge and self.cfg.image_longest_edge and edge != self.cfg.image_longest_edge:
                log.info("Adapter was trained at image_longest_edge=%s (config: %s).",
                         edge, self.cfg.image_longest_edge)
        self.model = PeftModel.from_pretrained(self.model, str(path)).merge_and_unload()
        self.adapter = path.name
        self.model_id = f"{self.model_id} + LoRA ({path.name})"

    # ------------------------------------------------------------------ setup helpers
    def _configure_image_size(self) -> None:
        """Control how many 384/512-px tiles SmolVLM uses (speed vs. detail).

        Passed as a processor call kwarg, which works for dict (transformers 4.x)
        and SizeDict (5.x) image-processor configs alike.
        """
        self._image_kwargs: dict = {}
        ip = getattr(self.processor, "image_processor", None)
        edge = self.cfg.image_longest_edge
        if ip is None or not edge or getattr(ip, "max_image_size", None) is None:
            return
        base = ip.max_image_size.get("longest_edge", 384) or 384
        edge = max(base, int(round(edge / base)) * base)          # multiple of the tile size
        self._image_kwargs = {"size": {"longest_edge": edge}}

    def _build_letter_ids(self) -> dict[str, list[int]]:
        ids: dict[str, list[int]] = {}
        for letter in LETTERS:
            cands = set()
            for variant in (letter, " " + letter):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    cands.add(toks[0])
            if not cands:
                cands.add(self.tokenizer.encode(letter, add_special_tokens=False)[0])
            ids[letter] = sorted(cands)
        return ids

    def _inputs(self, image, prompt: str):
        import torch
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt", **self._image_kwargs)
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if torch.is_floating_point(v):
                    v = v.to(self.dtype)
            out[k] = v
        return out

    @staticmethod
    def _image(frame_or_image):
        return frame_or_image.to_pil() if isinstance(frame_or_image, PreprocessedFrame) else frame_or_image

    # ------------------------------------------------------------------ primitives
    def generate(self, image, prompt: str, max_new_tokens: int | None = None,
                 repetition_penalty: float | None = None) -> str:
        import torch
        inputs = self._inputs(self._image(image), prompt)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens_short,
                do_sample=False,
                num_beams=1,
                repetition_penalty=repetition_penalty or self.cfg.repetition_penalty,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip().removeprefix("Assistant:").strip()

    def _next_token_logits(self, image, prompt: str) -> np.ndarray:
        import torch
        inputs = self._inputs(image, prompt)
        with torch.inference_mode():
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
        return out.logits[0, -1].float().cpu().numpy()

    def score_options(self, image, question: str, options: Sequence[str], context: str = "",
                      permutations: int | None = None) -> np.ndarray:
        """Return P(option | image, context, question) for each option (sums to 1)."""
        n = len(options)
        if n == 0:
            return np.zeros(0)
        if n == 1:
            return np.ones(1)
        if n > len(LETTERS):
            raise ValueError(f"At most {len(LETTERS)} options are supported")
        image = self._image(image)
        k = max(1, permutations or self.cfg.debias_permutations)
        orders = [list(range(n))]
        if k >= 2:
            orders.append(list(reversed(range(n))))
        for shift in range(1, k - 1):                       # extra cyclic shifts
            orders.append([(i + shift * max(1, n // k)) % n for i in range(n)])

        acc = np.zeros(n)
        for order in orders:
            lines = "\n".join(f"{LETTERS[i]}. {options[j]}" for i, j in enumerate(order))
            prompt = (f"{context.strip()}\n\n" if context.strip() else "") + (
                f"Question: {question}\nOptions:\n{lines}\n"
                "Answer with the letter of the single best option."
            )
            logits = self._next_token_logits(image, prompt)
            letter_logits = [max(logits[t] for t in self._letter_ids[LETTERS[i]]) for i in range(n)]
            probs = _softmax(np.asarray(letter_logits))
            for i, j in enumerate(order):
                acc[j] += probs[i]
        return acc / len(orders)

    # ------------------------------------------------------------------ Module 2A
    def understand(self, frame) -> VisualConcepts:
        image = self._image(frame)
        g = lambda key, n=None: self.generate(image, PROMPTS[key], n or self.cfg.max_new_tokens_short)
        caption = complete_sentences(
            self.generate(image, PROMPTS["describe"], self.cfg.max_new_tokens_description))
        raw = {"caption": caption}
        terms = extract_terms(caption)

        if self.cfg.fast_mode:
            count = parse_count(caption)
            objects, actions, gestures = terms["nouns"], terms["verbs"], terms["adjectives"]
            setting, people = "", terms["people"]
        else:
            for key in ("count", "objects", "actions", "gestures", "setting", "people"):
                raw[key] = g(key, 8 if key == "count" else None)
            count = parse_count(raw["count"])
            objects = split_list(raw["objects"])
            actions = split_list(raw["actions"])
            gestures = split_list(raw["gestures"])
            setting = raw["setting"].strip().rstrip(".")
            people = split_list(raw["people"]) + terms["people"]

        # Guard against a missed count when the caption clearly mentions people.
        caption_count = parse_count(caption) if any(p in caption.lower() for p in PERSON_NOUNS) else None
        if count is None:
            count = caption_count if caption_count is not None else (2 if len(terms["people"]) > 1 else len(terms["people"]))
        count = int(max(0, min(count, 50)))
        if count == 0 and terms["people"]:
            count = 1

        return VisualConcepts(
            caption=caption,
            people_count=count,
            people=unique_people(dedupe(people))[:8],
            objects=[o for o in dedupe(objects) if o not in PERSON_NOUNS][:15],
            actions=dedupe(actions)[:12],
            gestures=dedupe(gestures)[:8],
            setting=setting,
            spatial_relations=extract_spatial_relations(caption),
            raw_answers=raw,
        )

    # ------------------------------------------------------------------ Module 2B
    def draft_reasoning(self, frame, visual: VisualConcepts, question: str | None = None) -> DraftReasoning:
        image = self._image(frame)
        n = self.cfg.max_new_tokens_short
        multi = visual.people_count >= 2
        activity = self.generate(image, PROMPTS["activity_multi" if multi else "activity_single"], n)
        activity_clean = activity.strip().rstrip(".")
        relationship = (self.generate(image, PROMPTS["relationship"], n) if multi
                        else "no relationship (single person)" if visual.people_count == 1 else "no people")
        act_phrase = activity_clean[0].lower() + activity_clean[1:] if activity_clean else "interacting"
        intention = self.generate(image, PROMPTS["intention"].format(activity=act_phrase), n + 16)
        rationale = self.generate(image, PROMPTS["rationale"].format(activity=act_phrase),
                                  self.cfg.max_new_tokens_explanation)
        answer = None
        if question:
            answer = self.generate(image, PROMPTS["question"].format(question=question.strip()),
                                   self.cfg.max_new_tokens_explanation)
        return DraftReasoning(activity=activity_clean, relationship=relationship.strip().rstrip("."),
                              intention=intention.strip(), rationale=rationale.strip(),
                              question=question, answer=answer)

    def run(self, frame, question: str | None = None) -> tuple[VisualConcepts, DraftReasoning]:
        """Module 2 entry point: (visual_concepts, draft) = run_small_vlm(tensor)."""
        visual = self.understand(frame)
        draft = self.draft_reasoning(frame, visual, question)
        return visual, draft
