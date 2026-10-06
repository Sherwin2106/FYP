#!/usr/bin/env python
"""LoRA fine-tuning of SmolVLM on VCR (Hugging Face dataset ``Rowan/vcr``).

Designed to run unattended on one free Kaggle/Colab GPU (T4 / P100, 16 GB), and
also on a Mac (MPS) for small smoke tests. Everything it writes comes from the run
itself: the adapter weights, the trainer state and loss log, and before/after
accuracy on a held-out VCR validation subset.

What it does
------------
1. Streams the first N VCR images (and their 1-3 questions each) straight from the
   dataset's Parquet shards - no need to download the 24 GB training split.
2. Draws the referenced people/objects on each image as labelled boxes
   ("person1", "person2", ...) and rewrites the question/answers/rationales with the
   same tags - the standard way VCR references are grounded for a VLM.
3. Builds three kinds of training examples per question:
     * qa      - multiple choice Q -> A, target = option letter
     * qar     - multiple choice QA -> R (why is the answer right), target = letter
     * explain - free-form "answer + visual reason", target = gold answer + rationale
   The multiple-choice prompts are word-for-word the prompts the Phase I pipeline
   scores at inference (svcr/vlm.py ``score_options``), so training directly improves
   the probabilities Module 4 fuses. Option order is shuffled so no letter is favoured.
4. Trains LoRA adapters (r=16) on the language model's attention + MLP projections;
   the SigLIP vision encoder and the connector stay frozen. Loss is computed only on
   the assistant's answer tokens.
5. Evaluates Q->A and QA->R accuracy on held-out validation questions with the base
   model (adapter disabled) and with the fine-tuned adapter, and saves sample
   explanations from both for qualitative comparison.

Typical Kaggle run (T4, ~2.5-3 h):
    python train_vcr_lora.py --output-dir /kaggle/working/smolvlm-vcr-lora

Smoke test on a Mac (a few minutes, proves the pipeline end-to-end):
    python finetune/train_vcr_lora.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import random
import re
import shutil
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# A Kaggle "T4 x2" machine exposes two GPUs; Trainer would wrap the model in
# DataParallel, which does not play well with PEFT + gradient checkpointing.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

LETTERS = string.ascii_uppercase
TAG_NOTE = "People and objects mentioned in the text are marked in the image with labelled boxes."
MC_INSTRUCTION = "Answer with the letter of the single best option."
EXPLAIN_SUFFIX = "Answer briefly, then give the visual reason."
LORA_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
TAG_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
              (70, 200, 200), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40)]


# =============================================================================== args
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--dataset", default="Rowan/vcr")
    ap.add_argument("--output-dir", default="outputs/smolvlm-vcr-lora")
    ap.add_argument("--train-images", type=int, default=2600,
                    help="VCR training images to stream (each has 1-3 questions)")
    ap.add_argument("--max-train-examples", type=int, default=7500,
                    help="cap on qa/qar/explain examples after expansion")
    ap.add_argument("--tasks", default="qa,qar,explain")
    ap.add_argument("--eval-images", type=int, default=150, help="validation images used for evaluation")
    ap.add_argument("--max-eval-questions", type=int, default=300, help="held-out TEST questions (final report)")
    ap.add_argument("--select-questions", type=int, default=100,
                    help="separate validation questions scored after every epoch to pick the best epoch (0 = off)")
    ap.add_argument("--image-longest-edge", type=int, default=768,
                    help="image size the processor tiles (768 = 2x2 tiles of 384 + global view)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (CUDA only)")
    ap.add_argument("--fp32-base", action="store_true",
                    help="keep base weights in fp32 (AMP fp16 compute) - slower, avoids fp16 overflow")
    ap.add_argument("--save-steps", type=int, default=50, help="checkpoint every N optimizer steps")
    ap.add_argument("--time-limit-hours", type=float, default=9.5,
                    help="stop training gracefully (then save + evaluate) after this long")
    ap.add_argument("--stop-at", metavar="HH:MM",
                    help="pause cleanly at this India time (IST): save checkpoint + finished epochs, then exit; "
                         "continue later with kaggle_launch.py --resume")
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore earlier checkpoints / data cache / baseline and start from scratch")
    ap.add_argument("--num-samples", type=int, default=4, help="qualitative before/after explanations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny end-to-end run (SmolVLM-256M, a few steps) to validate the setup")
    args = ap.parse_args(argv)
    if args.stop_at:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        hh, mm = (int(x) for x in args.stop_at.split(":"))
        stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop <= now:
            stop += timedelta(days=1)
        hours = (stop - now).total_seconds() / 3600
        args.time_limit_hours = min(args.time_limit_hours, hours)
        print(f"Will pause at {stop:%H:%M} IST ({hours:.2f} h from now).", flush=True)
    if args.smoke_test:
        args.model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        args.train_images, args.max_train_examples = 12, 24
        args.eval_images, args.max_eval_questions = 6, 6
        args.max_steps = args.max_steps if (args.max_steps > 0 or args.epochs > 1) else 4
        args.select_questions = min(args.select_questions, 4)
        args.batch_size, args.grad_accum = 1, 2
        args.save_steps, args.num_samples, args.image_longest_edge = 2, 2, 512
        if args.output_dir == "outputs/smolvlm-vcr-lora":
            args.output_dir = "outputs/smolvlm-vcr-lora-smoke"
    return args


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================== VCR -> text
def object_tag(objects: list[str], idx: int) -> str:
    return f"{objects[idx]}{idx + 1}" if 0 <= idx < len(objects) else f"object{idx + 1}"


def render_tokens(tokens: list[dict], objects: list[str], refs: set[int]) -> str:
    """Rebuild VCR text from its token list, naming referenced objects 'person1', ..."""
    words = []
    for tok in tokens:
        idxs = tok.get("object_indices") or []
        if tok.get("kind") == "objects" and idxs:
            refs.update(idxs)
            tags = [object_tag(objects, i) for i in idxs]
            words.append(tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1])
        elif tok.get("text"):
            words.append(tok["text"])
    text = " ".join(words)
    text = re.sub(r" ([?.!,;:'])", r"\1", text)
    text = re.sub(r" (n't|'s|'re|'m|'ll|'ve|'d)\b", r"\1", text)
    return text.strip()


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_tags(img: Image.Image, boxes: list, objects: list[str], refs: set[int],
              meta_size: tuple[int, int]) -> Image.Image:
    """Draw labelled boxes for referenced objects (VCR's grounding convention)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    sx = img.width / max(1, meta_size[0])
    sy = img.height / max(1, meta_size[1])
    big = max(img.width, img.height)
    width = max(3, big // 320)
    font = _font(max(14, int(big * 0.032)))
    for i in sorted(refs):
        if i >= len(boxes) or boxes[i] is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i][:4]]
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        color = TAG_COLORS[i % len(TAG_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = object_tag(objects, i)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 2 * width)
        draw.rectangle([x1, ty, x1 + tw + 3 * width, ty + th + 2 * width], fill=color)
        draw.text((x1 + 1.5 * width, ty + 0.5 * width), label, fill=(255, 255, 255), font=font)
    return img


def resize_longest(img: Image.Image, edge: int) -> Image.Image:
    s = edge / max(img.width, img.height)
    if s >= 1:
        return img
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.LANCZOS)


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@dataclass
class VCRQuestion:
    annot_id: str
    image: bytes                 # JPEG with tags drawn, already resized
    question: str
    answers: list[str]
    answer_label: int
    rationales: list[str]
    rationale_label: int


def stream_vcr(repo: str, split: str, max_images: int, image_edge: int) -> list[VCRQuestion]:
    """Read the first ``max_images`` images of a split row-group by row-group."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}/image_examples/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No image_examples/{split} parquet files found in {repo}")
    cols = ["image", "objects", "boxes", "width", "height", "annotations"]
    out: list[VCRQuestion] = []
    n_images = 0
    for path in files:
        if n_images >= max_images:
            break
        done_in_file = 0                       # rows of this file already consumed
        for attempt in range(4):
            try:
                with fs.open(path, "rb", block_size=16 * 1024 * 1024) as fh:
                    pf = pq.ParquetFile(fh)
                    seen = 0
                    for batch in pf.iter_batches(batch_size=32, columns=cols):
                        for row in batch.to_pylist():
                            seen += 1
                            if seen <= done_in_file:   # skip rows read before a retry
                                continue
                            if n_images >= max_images:
                                break
                            n_images += 1
                            done_in_file += 1
                            out.extend(_row_to_questions(row, image_edge))
                        if n_images >= max_images:
                            break
                        log(f"  {split}: {n_images}/{max_images} images, {len(out)} questions")
                break
            except OSError as exc:                     # transient network errors while streaming
                if attempt == 3:
                    raise
                log(f"  stream error on {path} ({exc}); retrying in {10 * (attempt + 1)}s")
                time.sleep(10 * (attempt + 1))
    return out


def _row_to_questions(row: dict, image_edge: int) -> list[VCRQuestion]:
    img_field = row.get("image") or {}
    raw = img_field.get("bytes") if isinstance(img_field, dict) else None
    if not raw:
        return []
    try:
        base = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return []
    objects = row.get("objects") or []
    boxes = row.get("boxes") or []
    meta_size = (row.get("width") or base.width, row.get("height") or base.height)
    questions = []
    for ann in row.get("annotations") or []:
        if not isinstance(ann, dict) or ann.get("answer_label") is None or ann.get("rationale_label") is None:
            continue
        objs = ann.get("objects") or objects
        refs: set[int] = set()
        q = render_tokens(ann.get("question_tokens") or [], objs, refs)
        answers = [render_tokens(t, objs, refs) for t in ann.get("answer_choice_tokens") or []]
        rationales = [render_tokens(t, objs, refs) for t in ann.get("rationale_choice_tokens") or []]
        if not q or len(answers) < 2 or len(rationales) < 2:
            continue
        tagged = resize_longest(draw_tags(base, boxes, objs, refs, meta_size), image_edge)
        questions.append(VCRQuestion(
            annot_id=str(ann.get("annot_id", "")), image=jpeg_bytes(tagged), question=q,
            answers=answers, answer_label=int(ann["answer_label"]),
            rationales=rationales, rationale_label=int(ann["rationale_label"]),
        ))
    return questions


# =============================================================================== prompts
def mc_prompt(question: str, options: list[str]) -> str:
    """Identical to SmolVLMEngine.score_options in svcr/vlm.py (with the VCR tag note as context)."""
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{TAG_NOTE}\n\nQuestion: {question}\nOptions:\n{lines}\n{MC_INSTRUCTION}"


def qar_question(q: VCRQuestion) -> str:
    return f"{q.question} The answer is: {q.answers[q.answer_label]} Why is this answer correct?"


def build_examples(questions: list[VCRQuestion], tasks: set[str], rng: random.Random) -> list[dict]:
    examples = []
    for q in questions:
        if "qa" in tasks:
            order = list(range(len(q.answers)))
            rng.shuffle(order)
            examples.append({"task": "qa", "image": q.image,
                             "prompt": mc_prompt(q.question, [q.answers[i] for i in order]),
                             "target": LETTERS[order.index(q.answer_label)]})
        if "qar" in tasks:
            order = list(range(len(q.rationales)))
            rng.shuffle(order)
            examples.append({"task": "qar", "image": q.image,
                             "prompt": mc_prompt(qar_question(q), [q.rationales[i] for i in order]),
                             "target": LETTERS[order.index(q.rationale_label)]})
        if "explain" in tasks:
            answer = q.answers[q.answer_label].rstrip()
            if answer and answer[-1] not in ".!?":
                answer += "."
            examples.append({"task": "explain", "image": q.image,
                             "prompt": f"{TAG_NOTE}\n\n{q.question} {EXPLAIN_SUFFIX}",
                             "target": f"{answer} {q.rationales[q.rationale_label]}"})
    rng.shuffle(examples)
    return examples


# =============================================================================== model
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args, device: str):
    import transformers
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    processor = AutoProcessor.from_pretrained(args.model_id)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"

    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    if device == "cuda" and args.fp32_base:
        dtype = torch.float32           # fp32 master weights; Trainer's AMP still computes in fp16
    elif device == "cuda":
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32           # MPS/CPU smoke tests: plain fp32, no AMP
    kwargs = {dtype_kw: dtype}
    if args.qlora:
        if device != "cuda":
            raise SystemExit("--qlora needs a CUDA GPU (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["vision_model", "connector", "lm_head"])
        kwargs["device_map"] = {"": 0}
    model = ModelCls.from_pretrained(args.model_id, **kwargs)
    if not args.qlora:
        model.to(device)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    return processor, model, dtype, use_bf16


def ignore_old_torchao() -> None:
    """Kaggle's image ships torchao 0.10; recent peft refuses to run next to torchao < 0.16.

    LoRA doesn't use torchao at all, so tell peft it isn't there instead of failing.
    """
    try:
        from importlib.metadata import version
        from packaging.version import Version
        found = version("torchao")
    except Exception:
        return
    if Version(found) >= Version("0.16.0"):
        return
    import peft.import_utils as iu
    iu.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as lora_torchao
        lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    log(f"Ignoring incompatible torchao {found} (not used by LoRA).")


def add_lora(model, args):
    ignore_old_torchao()
    from peft import LoraConfig, get_peft_model
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        bias="none", target_modules=LORA_TARGETS, init_lora_weights="gaussian")
    model = get_peft_model(model, config)
    # AMP's grad scaler requires fp32 trainable parameters.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA: {trainable:,} trainable / {total:,} total parameters ({100 * trainable / total:.2f}%)")
    return model, trainable, total


# =============================================================================== training data
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def find_last(seq: list[int], sub: list[int]) -> int | None:
    for start in range(len(seq) - len(sub), -1, -1):
        if seq[start:start + len(sub)] == sub:
            return start
    return None


class Collator:
    """Chat-formats each example and masks everything except the assistant's answer."""

    def __init__(self, processor, image_edge: int, dtype):
        self.processor = processor
        self.image_kwargs = {"size": {"longest_edge": image_edge}}
        self.dtype = dtype
        self.assistant_ids = processor.tokenizer.encode("Assistant:", add_special_tokens=False)

    def __call__(self, batch: list[dict]):
        texts, images = [], []
        for ex in batch:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]},
            ]
            texts.append(self.processor.apply_chat_template(messages, add_generation_prompt=False).strip())
            images.append([Image.open(io.BytesIO(ex["image"])).convert("RGB")])
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True, **self.image_kwargs)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i in range(labels.size(0)):
            start = find_last(enc["input_ids"][i].tolist(), self.assistant_ids)
            if start is None:
                labels[i] = -100
            else:
                labels[i, : start + len(self.assistant_ids)] = -100
        enc["labels"] = labels
        if "pixel_values" in enc and self.dtype != torch.float32:
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        return enc


# =============================================================================== evaluation
def letter_ids(tokenizer) -> dict[str, list[int]]:
    ids = {}
    for letter in LETTERS[:8]:
        cands = {tokenizer.encode(v, add_special_tokens=False)[0] for v in (letter, " " + letter)
                 if len(tokenizer.encode(v, add_special_tokens=False)) == 1}
        ids[letter] = sorted(cands) or [tokenizer.encode(letter, add_special_tokens=False)[0]]
    return ids


def _inputs(processor, image, prompt, image_edge, device, dtype):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=text, images=[image], return_tensors="pt", size={"longest_edge": image_edge})
    enc = {k: v.to(device) for k, v in enc.items()}
    if dtype != torch.float32:
        enc["pixel_values"] = enc["pixel_values"].to(dtype)
    return enc


@torch.inference_mode()
def choose(model, processor, image, prompt, n, ids, image_edge, device, dtype) -> int:
    enc = _inputs(processor, image, prompt, image_edge, device, dtype)
    logits = model(**enc).logits[0, -1].float()
    scores = [max(logits[t].item() for t in ids[LETTERS[i]]) for i in range(n)]
    return int(max(range(n), key=lambda i: scores[i]))


@torch.inference_mode()
def explain(model, processor, q: VCRQuestion, image_edge, device, dtype) -> str:
    enc = _inputs(processor, Image.open(io.BytesIO(q.image)).convert("RGB"),
                  f"{TAG_NOTE}\n\n{q.question} {EXPLAIN_SUFFIX}", image_edge, device, dtype)
    out = model.generate(**enc, max_new_tokens=60, do_sample=False)
    return processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def evaluate(model, processor, questions: list[VCRQuestion], image_edge, device, dtype, label: str) -> dict:
    model.eval()
    ids = letter_ids(processor.tokenizer)
    qa = qar = both = 0
    t0 = time.time()
    for k, q in enumerate(questions, 1):
        image = Image.open(io.BytesIO(q.image)).convert("RGB")
        a = choose(model, processor, image, mc_prompt(q.question, q.answers), len(q.answers),
                   ids, image_edge, device, dtype)
        r = choose(model, processor, image, mc_prompt(qar_question(q), q.rationales), len(q.rationales),
                   ids, image_edge, device, dtype)
        qa += a == q.answer_label
        qar += r == q.rationale_label
        both += (a == q.answer_label) and (r == q.rationale_label)
        if k % 50 == 0 or k == len(questions):
            log(f"  eval[{label}] {k}/{len(questions)}  Q->A {qa / k:.1%}  QA->R {qar / k:.1%}")
    n = max(1, len(questions))
    return {"questions": len(questions), "Q->A": qa / n, "QA->R": qar / n, "Q->AR": both / n,
            "seconds": round(time.time() - t0, 1)}


# =============================================================================== logging callback
def make_callback(log_path: Path, time_limit_s: float, t_start: float):
    from transformers import TrainerCallback

    class CsvLogger(TrainerCallback):
        def __init__(self):
            new_file = not log_path.exists()          # append when resuming a run
            self.fh = open(log_path, "a", newline="")
            self.writer = csv.writer(self.fh)
            if new_file:
                self.writer.writerow(["step", "epoch", "loss", "learning_rate", "grad_norm", "elapsed_s"])

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.writer.writerow([state.global_step, round(state.epoch or 0, 4), logs.get("loss"),
                                      logs.get("learning_rate"), logs.get("grad_norm"),
                                      round(time.time() - t_start, 1)])
                self.fh.flush()
                if not math.isfinite(float(logs["loss"])):
                    log("Loss became NaN/inf (fp16 overflow) - stopping. Re-run with --fp32-base.")
                    control.should_training_stop = True
            return control

        def on_step_end(self, args, state, control, **kw):
            if time.time() - t_start > time_limit_s:
                log("Time limit / stop-at reached - saving a checkpoint and pausing. "
                    "Continue later with: python finetune/kaggle_launch.py --resume")
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_train_end(self, args, state, control, **kw):
            self.fh.close()

    return CsvLogger()


def plot_loss(log_history: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pts = [(h["step"], h["loss"]) for h in log_history if "loss" in h and "step" in h]
    if not pts:
        return
    steps, losses = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
    ax.plot(steps, losses, color="#8B6F5A", linewidth=1.2, alpha=0.45, label="training loss")
    if len(losses) >= 5:
        w = max(3, len(losses) // 15)
        smooth = [sum(losses[max(0, i - w + 1):i + 1]) / len(losses[max(0, i - w + 1):i + 1])
                  for i in range(len(losses))]
        ax.plot(steps, smooth, color="#5C4636", linewidth=2, label=f"moving average ({w})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("SmolVLM LoRA fine-tuning on VCR")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================== per-epoch selection
EPOCH_FIELDS = ["epoch", "step", "train_loss", "Q->A", "QA->R", "Q->AR"]


def make_epoch_callback(model, processor, select_q, out: Path, args, device, dtype):
    """After every epoch: save that epoch's adapter and score it on the selection set."""
    from transformers import TrainerCallback

    class EpochSelector(TrainerCallback):
        def on_epoch_end(self, targs, state, control, **kw):
            k = int(round(state.epoch or 0))
            # Trainer also fires this when a run is paused mid-epoch (time limit / --stop-at):
            # only a truly finished epoch gets saved and scored.
            if k < 1 or abs((state.epoch or 0) - k) > 1e-3:
                return control
            folder = out / "epochs" / f"epoch_{k:02d}"
            model.save_pretrained(str(folder))
            losses = [h["loss"] for h in state.log_history if "loss" in h and k - 1 < h.get("epoch", 0) <= k + 1e-6]
            m = evaluate(model, processor, select_q, args.image_longest_edge, device, dtype, f"epoch {k}")
            model.train()
            row = {"epoch": k, "step": state.global_step,
                   "train_loss": round(sum(losses) / len(losses), 4) if losses else None,
                   "Q->A": round(m["Q->A"], 4), "QA->R": round(m["QA->R"], 4), "Q->AR": round(m["Q->AR"], 4)}
            path = out / "epoch_metrics.csv"
            new = not path.exists()
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=EPOCH_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            log(f"Epoch {k}: train loss {row['train_loss']}  selection Q->A {m['Q->A']:.1%}  "
                f"QA->R {m['QA->R']:.1%}  Q->AR {m['Q->AR']:.1%}  (adapter saved to {folder})")
            return control

    return EpochSelector()


def read_epoch_metrics(out: Path) -> list[dict]:
    path = out / "epoch_metrics.csv"
    if not path.exists():
        return []
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            e = int(r["epoch"])
            if (out / "epochs" / f"epoch_{e:02d}" / "adapter_config.json").exists():
                rows[e] = {"epoch": e, "step": int(r["step"]),
                           "train_loss": float(r["train_loss"]) if r["train_loss"] else None,
                           **{k: float(r[k]) for k in ("Q->A", "QA->R", "Q->AR")}}
    return [rows[e] for e in sorted(rows)]


def load_adapter_weights(model, folder: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(folder / "adapter_model.safetensors")))


def plot_epochs(epochs: list[dict], best: int, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=140)
    for key, color in (("Q->A", "#8B6F5A"), ("QA->R", "#8FA58D"), ("Q->AR", "#5C4636")):
        ax.plot(xs, [100 * r[key] for r in epochs], marker="o", markersize=3.5, color=color, label=key)
    ax.axvline(best, color="#C9A66B", linestyle="--", linewidth=1.2, label=f"best epoch ({best})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("selection-set accuracy (%)")
    ax.set_xticks(xs)
    ax2 = ax.twinx()
    losses = [r["train_loss"] for r in epochs]
    if all(v is not None for v in losses):
        ax2.plot(xs, losses, color="#7A6A5D", linestyle=":", linewidth=1.2, label="train loss")
        ax2.set_ylabel("train loss")
    ax.set_title("Per-epoch validation accuracy (LoRA on VCR)")
    ax.grid(alpha=0.25)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================== checkpoint / resume state
def _latest_checkpoint(dirs: list[Path]) -> Path | None:
    best, best_step = None, -1
    for d in dirs:
        for c in d.glob("checkpoint-*"):
            m = re.fullmatch(r"checkpoint-(\d+)", c.name)
            if m and (c / "trainer_state.json").exists() and int(m.group(1)) > best_step:
                best, best_step = c, int(m.group(1))
    return best


def find_previous_state(out: Path) -> dict:
    """Find resumable state in this run's output dir or in an earlier run attached as input.

    On Kaggle, attaching a previous version's output (it appears under /kaggle/input) lets a
    new run continue from its last checkpoint and reuse its data cache and baseline scores.
    """
    roots = [out]
    kin = Path("/kaggle/input")
    if kin.exists():
        roots += sorted({p.parent.parent for p in kin.rglob("data_cache/meta.json")} |
                        {p.parent.parent for p in kin.rglob("checkpoints/checkpoint-*/trainer_state.json")
                         if p.parent.parent.name == "checkpoints"} |
                        {p.parent for p in kin.rglob("base_eval.json")})
    state: dict = {}
    ckpt = _latest_checkpoint([r / "checkpoints" for r in roots if (r / "checkpoints").exists()])
    if ckpt:
        state["checkpoint"] = str(ckpt)
    if ckpt:
        prev_log = ckpt.parent.parent / "training_log.csv"
        if prev_log.exists() and prev_log.resolve() != (out / "training_log.csv").resolve():
            shutil.copy(prev_log, out / "training_log.csv")   # keep the full loss history
        prev_root = ckpt.parent.parent
        if prev_root.resolve() != out.resolve():
            if (prev_root / "epoch_metrics.csv").exists():
                shutil.copy(prev_root / "epoch_metrics.csv", out / "epoch_metrics.csv")
            if (prev_root / "epochs").exists():
                shutil.copytree(prev_root / "epochs", out / "epochs", dirs_exist_ok=True)
    for r in roots:
        if "data_cache" not in state and (r / "data_cache" / "meta.json").exists():
            state["data_cache"] = r / "data_cache"
        if "base_eval" not in state and (r / "base_eval.json").exists():
            state["base_eval"] = r / "base_eval.json"
    if state:
        log("Found earlier run state: " + ", ".join(f"{k}={v}" for k, v in state.items()))
    return state


def save_data_cache(folder: Path, key: dict, train_q: list, val_q: list) -> None:
    import pickle
    folder.mkdir(parents=True, exist_ok=True)
    for name, qs in (("train", train_q), ("validation", val_q)):
        with open(folder / f"{name}.pkl", "wb") as fh:
            pickle.dump([q.__dict__ for q in qs], fh, protocol=pickle.HIGHEST_PROTOCOL)
    (folder / "meta.json").write_text(json.dumps({"key": key, "train": len(train_q), "validation": len(val_q)}))


def load_data_cache(folder, key: dict):
    if not folder:
        return None
    import pickle
    folder = Path(folder)
    try:
        meta = json.loads((folder / "meta.json").read_text())
        if meta.get("key") != key:
            log("Data cache was built with different settings - ignoring it.")
            return None
        out = []
        for name in ("train", "validation"):
            with open(folder / f"{name}.pkl", "rb") as fh:
                out.append([VCRQuestion(**d) for d in pickle.load(fh)])
        return out[0], out[1]
    except Exception as exc:   # corrupt / partial cache: just stream again
        log(f"Could not read data cache ({exc}) - streaming instead.")
        return None


def load_json_if(path, key: dict):
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
        return data if data.get("key") == key else None
    except Exception:
        return None


def zip_results(out: Path) -> str:
    """Zip the deliverables only - checkpoints and the data cache stay out of the download."""
    import zipfile
    archive = out.parent / f"{out.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            rel = f.relative_to(out)
            if f.is_file() and rel.parts[0] not in ("checkpoints", "data_cache", "epochs"):
                zf.write(f, Path(out.name) / rel)
    return str(archive)


# =============================================================================== main
def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    adapter_dir = out / "adapter"
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = pick_device()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or device
    log(f"Device: {device} ({gpu}); model: {args.model_id}")

    # ---------------------------------------------------------------- model (first: fail fast)
    # Library/GPU problems surface here within ~1 min instead of after ~10 min of data streaming.
    processor, model, dtype, use_bf16 = load_model(args, device)
    model, n_trainable, n_total = add_lora(model, args)

    # ---------------------------------------------------------------- data (cached across runs)
    if args.fresh:
        for sub in ("checkpoints", "data_cache"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
        for f in ("base_eval.json", "training_log.csv"):
            (out / f).unlink(missing_ok=True)
        state = {}
    else:
        state = find_previous_state(out)
    cache_key = {"dataset": args.dataset, "train_images": args.train_images, "eval_images": args.eval_images,
                 "image_longest_edge": args.image_longest_edge}
    cached = load_data_cache(state.get("data_cache"), cache_key)
    if cached:
        train_q, val_q = cached
        log(f"Loaded cached VCR data ({len(train_q)} train / {len(val_q)} validation questions) "
            f"from {state['data_cache']} - no re-streaming needed.")
    else:
        log(f"Streaming {args.train_images} training images from {args.dataset} ...")
        train_q = stream_vcr(args.dataset, "train", args.train_images, args.image_longest_edge)
        log(f"Streaming {args.eval_images} validation images ...")
        val_q = stream_vcr(args.dataset, "validation", args.eval_images, args.image_longest_edge)
        save_data_cache(out / "data_cache", cache_key, train_q, val_q)
    rng.shuffle(val_q)
    select_q = val_q[args.max_eval_questions: args.max_eval_questions + args.select_questions]
    val_q = val_q[: args.max_eval_questions]                      # test set for the final report
    if args.select_questions and len(select_q) < args.select_questions:
        log(f"Only {len(select_q)} selection questions available (wanted {args.select_questions}); "
            "raise --eval-images for more.")
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    examples = build_examples(train_q, tasks, rng)[: args.max_train_examples]
    counts = {t: sum(e["task"] == t for e in examples) for t in sorted(tasks)}
    log(f"Training examples: {len(examples)} {counts} from {len(train_q)} questions; "
        f"test questions: {len(val_q)}; epoch-selection questions: {len(select_q)}")
    data_s = time.time() - t_start

    # ---------------------------------------------------------------- baseline (cached across runs)
    results: dict = {}
    eval_key = {"model_id": args.model_id, "annot_ids": [q.annot_id for q in val_q],
                "image_longest_edge": args.image_longest_edge}
    base = load_json_if(state.get("base_eval"), eval_key)
    if base:
        results["base"], base_samples = base["results"], base["samples"]
        log(f"Reusing the base-model evaluation from {state['base_eval']}.")
    elif not args.skip_baseline_eval:
        log("Evaluating the base model (adapter disabled) ...")
        with model.disable_adapter():
            results["base"] = evaluate(model, processor, val_q, args.image_longest_edge, device, dtype, "base")
            base_samples = [explain(model, processor, q, args.image_longest_edge, device, dtype)
                            for q in val_q[: args.num_samples]]
        (out / "base_eval.json").write_text(json.dumps(
            {"key": eval_key, "results": results["base"], "samples": base_samples}, ensure_ascii=False))
    else:
        base_samples = [None] * min(args.num_samples, len(val_q))

    # ---------------------------------------------------------------- train
    from transformers import Trainer, TrainingArguments
    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=1 if args.smoke_test else 5,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=(device == "cuda" and not use_bf16),
        bf16=use_bf16,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0 if device != "cuda" else 2,
        dataloader_pin_memory=device == "cuda",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ExampleDataset(examples),
                      data_collator=Collator(processor, args.image_longest_edge, dtype),
                      callbacks=[make_callback(out / "training_log.csv",
                                               args.time_limit_hours * 3600, t_start)]
                      + ([make_epoch_callback(model, processor, select_q, out, args, device, dtype)]
                         if select_q else []))
    resume = state.get("checkpoint")
    if resume:
        log(f"Resuming training from {resume}")
    log(f"Training: {len(examples)} examples, batch {args.batch_size} x accum {args.grad_accum}, "
        f"~{total_steps} optimizer steps ...")
    t_train = time.time()
    train_out = trainer.train(resume_from_checkpoint=resume)
    train_s = time.time() - t_train
    log(f"Training finished in {train_s / 60:.1f} min; final train loss {train_out.training_loss:.4f}")

    # ---------------------------------------------------------------- pick the best epoch
    epochs = read_epoch_metrics(out)
    selection = None
    if epochs:
        best = max(epochs, key=lambda r: (r["Q->AR"], r["Q->A"], -r["epoch"]))
        selection = {"best_epoch": best["epoch"], "criterion": "Q->AR on the selection set (ties: Q->A, then earlier)",
                     "epochs": epochs}
        load_adapter_weights(model, out / "epochs" / f"epoch_{best['epoch']:02d}")
        log(f"Best epoch: {best['epoch']} (selection Q->A {best['Q->A']:.1%}, QA->R {best['QA->R']:.1%}, "
            f"Q->AR {best['Q->AR']:.1%}) - using its weights for the final model.")
        plot_epochs(epochs, best["epoch"], out / "epoch_curve.png")

    # ---------------------------------------------------------------- save
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    plot_loss(trainer.state.log_history, out / "loss_curve.png")

    # ---------------------------------------------------------------- eval after
    log("Evaluating the fine-tuned model ...")
    results["finetuned"] = evaluate(model, processor, val_q, args.image_longest_edge, device, dtype, "lora")
    ft_samples = [explain(model, processor, q, args.image_longest_edge, device, dtype)
                  for q in val_q[: args.num_samples]]
    if selection:
        results["selection"] = selection
    results["samples"] = [
        {"annot_id": q.annot_id, "question": q.question,
         "gold": f"{q.answers[q.answer_label]} {q.rationales[q.rationale_label]}",
         "base": b, "finetuned": f}
        for q, b, f in zip(val_q, base_samples, ft_samples)
    ]
    (out / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    import peft
    import transformers
    run = {
        "model_id": args.model_id, "dataset": args.dataset, "device": device, "gpu": gpu,
        "qlora": args.qlora, "dtype": str(dtype).replace("torch.", ""),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": LORA_TARGETS, "trainable_params": n_trainable, "total_params": n_total},
        "data": {"train_images": args.train_images, "train_questions": len(train_q),
                 "train_examples": len(examples), "examples_per_task": counts,
                 "eval_questions": len(val_q), "image_longest_edge": args.image_longest_edge},
        "training": {"epochs": args.epochs, "optimizer_steps": trainer.state.global_step,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
                     "final_train_loss": train_out.training_loss, "train_minutes": round(train_s / 60, 1),
                     "data_prep_minutes": round(data_s / 60, 1)},
        "selection_questions": len(select_q),
        "results": {k: v for k, v in results.items() if k not in ("samples", "selection")},
        "best_epoch": selection["best_epoch"] if selection else None,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "python": platform.python_version()},
        "smoke_test": args.smoke_test,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "run_config.json").write_text(json.dumps(run, indent=2))
    (adapter_dir / "svcr_training.json").write_text(json.dumps(
        {"image_longest_edge": args.image_longest_edge, "base_model": args.model_id}, indent=2))
    write_model_card(out, run, results)

    archive = zip_results(out)
    log(f"Done in {(time.time() - t_start) / 60:.1f} min. Results: {out}  (zip: {archive})")
    for k in ("base", "finetuned"):
        if k in results:
            r = results[k]
            log(f"  {k:9s} Q->A {r['Q->A']:.1%}   QA->R {r['QA->R']:.1%}   Q->AR {r['Q->AR']:.1%}")
    return 0


def epoch_section(selection, n_select) -> str:
    if not selection:
        return ""
    rows = "\n".join(
        f"| {r['epoch']}{' **(best)**' if r['epoch'] == selection['best_epoch'] else ''} | "
        f"{r['train_loss'] if r['train_loss'] is not None else '-'} | {r['Q->A']:.1%} | {r['QA->R']:.1%} | {r['Q->AR']:.1%} |"
        for r in selection["epochs"])
    return f"""
## Epoch selection ({n_select} validation questions, separate from the test set above)
The adapter was saved and scored after every epoch; the final model uses **epoch {selection['best_epoch']}**
(criterion: {selection['criterion']}). Curve: `epoch_curve.png`; raw numbers: `epoch_metrics.csv`.

| epoch | train loss | Q->A | QA->R | Q->AR |
|---|---|---|---|---|
{rows}
"""


def write_model_card(out: Path, run: dict, results: dict) -> None:
    def row(name):
        r = results.get(name)
        return (f"| {name} | {r['Q->A']:.1%} | {r['QA->R']:.1%} | {r['Q->AR']:.1%} |" if r
                else f"| {name} | - | - | - |")
    t = run["training"]
    card = f"""# SmolVLM + LoRA fine-tuned on VCR

Base model: `{run['model_id']}` · dataset: `{run['dataset']}` · hardware: {run['gpu']} ({run['device']})
{'**Smoke test run - not a real training run.**' if run['smoke_test'] else ''}

## Training
* LoRA r={run['lora']['r']}, alpha={run['lora']['alpha']}, dropout={run['lora']['dropout']} on the language model's
  attention + MLP projections ({run['lora']['trainable_params']:,} trainable of {run['lora']['total_params']:,} parameters);
  vision encoder and connector frozen{'; base model loaded in 4-bit (QLoRA)' if run['qlora'] else ''}.
* {run['data']['train_examples']} examples ({run['data']['examples_per_task']}) from {run['data']['train_questions']}
  VCR training questions ({run['data']['train_images']} images), image longest edge {run['data']['image_longest_edge']} px.
* {t['optimizer_steps']} optimizer steps (batch {t['batch_size']} x grad-accum {t['grad_accum']}, lr {t['lr']}, cosine),
  {t['train_minutes']} min; final training loss {t['final_train_loss']:.4f}. Loss per step: `training_log.csv`, `loss_curve.png`.

## Test ({run['data']['eval_questions']} held-out VCR validation questions)
| model | Q->A | QA->R | Q->AR |
|---|---|---|---|
{row('base')}
{row('finetuned')}

Chance level: Q->A 25%, QA->R 25%, Q->AR 6.25%. Sample explanations (base vs fine-tuned): `eval_results.json`.
{epoch_section(results.get("selection"), run.get("selection_questions"))}

## Use in the Phase I pipeline
Copy the `adapter/` folder to `models/smolvlm-vcr-lora/` in the project and run with
`--config configs/finetuned.yaml` (or set `vlm.adapter_path`).
"""
    (out / "README.md").write_text(card)


if __name__ == "__main__":
    sys.exit(main())
