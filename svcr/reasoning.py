"""Module 4 - Social Interaction Reasoning (fusion point of the system).

    context      <- fuse(visual, commonsense, question)
    activity     <- infer_activity(context)
    relationship <- infer_relationship(context | activity)
    intention    <- infer_intention(context | activity, relationship)
    prediction   <- combine(activity, relationship, intention)
    explanation  <- generate_explanation(prediction, context)

Each task is inferred as follows:

1. **Shortlist** candidate labels using a prior built from the ConceptNet support
   (Module 3) and the match with the VLM's free-form draft (Module 2).
2. **VLM scoring** - SmolVLM sees the image plus the fused context (scene facts and
   ConceptNet knowledge) and scores the shortlisted options as a multiple-choice
   question, read off the next-token distribution and debiased over orderings.
3. **Log-linear fusion**

       log P(L) ∝ w_vlm·log p_vlm(L) + w_cn·log p_cn(L) + w_draft·match(L)

   where p_cn = softmax(support / T). The result is a calibrated distribution per
   task, with an ambiguity flag when the top-2 margin is small.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

from .config import ReasoningConfig
from .schemas import (CommonsenseResult, DraftReasoning, LabelScore, PreprocessedFrame,
                      SocialPrediction, TaskPrediction, VisualConcepts)
from .taxonomy import Label, Taxonomy

log = logging.getLogger(__name__)

TASK_QUESTIONS = {
    "activity": "Which option best describes the social interaction or activity taking place between the people in the image?",
    "activity_single": "Which option best describes what is happening in the image?",
    "relationship": "Which option best describes the relationship between the people in the image?",
    "intention": "Which option best describes the main intention or goal of the people in this interaction?",
}


class VLM(Protocol):
    def generate(self, image, prompt: str, max_new_tokens: int | None = None) -> str: ...
    def score_options(self, image, question: str, options: Sequence[str], context: str = "") -> np.ndarray: ...


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def display_name(label: Label) -> str:
    if label.task == "activity":
        return label.id.replace("_", " ").title()
    text = label.description
    for article in ("a ", "an "):
        if text.startswith(article):
            text = text[len(article):]
    return text[0].upper() + text[1:]


class SocialInteractionReasoner:
    def __init__(self, vlm: VLM, taxonomy: Taxonomy, cfg: ReasoningConfig | None = None,
                 max_new_tokens_explanation: int = 120):
        self.vlm = vlm
        self.taxonomy = taxonomy
        self.cfg = cfg or ReasoningConfig()
        self.max_new_tokens_explanation = max_new_tokens_explanation

    # ------------------------------------------------------------------ fusion of inputs
    def fuse(self, visual: VisualConcepts, commonsense: CommonsenseResult,
             question: str | None = None) -> str:
        lines = []
        if self.cfg.include_caption_in_context and visual.caption:
            lines.append(f"Scene description: {visual.caption.strip()}")
        lines.append(f"Number of people: {visual.people_count}")
        if visual.people:
            lines.append(f"People: {', '.join(visual.people[:5])}")
        if visual.actions:
            lines.append(f"Actions: {', '.join(visual.actions[:6])}")
        if visual.gestures:
            lines.append(f"Expressions and gestures: {', '.join(visual.gestures[:5])}")
        if visual.objects:
            lines.append(f"Objects: {', '.join(visual.objects[:8])}")
        if visual.setting:
            lines.append(f"Setting: {visual.setting}")
        facts = commonsense.facts[: self.cfg.max_context_facts]
        if facts:
            lines.append("Commonsense knowledge:")
            lines += [f"- {f.sentence()}." for f in facts]
        if question:
            lines.append(f"User question: {question}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ per-task inference
    def _shortlist(self, task: str, visual: VisualConcepts, commonsense: CommonsenseResult) -> list[Label]:
        labels = list(self.taxonomy[task])
        if task == "relationship":
            labels = [l for l in labels if l.id != "none_single"]

        support = commonsense.support.get(task, {})
        draft = commonsense.draft_match.get(task, {})
        scene_text = " ".join([visual.caption, *visual.actions, *visual.objects, visual.setting])
        prior = {l.id: support.get(l.id, 0.0) + 1.5 * draft.get(l.id, 0.0) + 0.3 * self.taxonomy.lexical_match(scene_text, l)
                 for l in labels}
        informative = [l for l in labels if prior[l.id] > 0.05]
        if len(informative) < 3:
            return labels[:26]                  # no reliable prior -> let the VLM see every option
        ranked = sorted(labels, key=lambda l: -prior[l.id])
        chosen = ranked[: self.cfg.shortlist_size]
        must = commonsense.disambiguated.get(task)
        if must and must not in {l.id for l in chosen} and any(l.id == must for l in labels):
            chosen[-1] = self.taxonomy.get(task, must)
        if (task == "activity" and visual.people_count <= 1 and self.taxonomy.has(task, "no_interaction")
                and all(l.id != "no_interaction" for l in chosen)):
            chosen[-1] = self.taxonomy.get(task, "no_interaction")
        return chosen

    def _infer_task(self, task: str, frame, context: str, visual: VisualConcepts,
                    commonsense: CommonsenseResult, draft_text: str) -> TaskPrediction:
        cfg = self.cfg
        candidates = self._shortlist(task, visual, commonsense)
        qkey = "activity_single" if task == "activity" and visual.people_count <= 1 else task
        p_vlm = np.asarray(self.vlm.score_options(frame, TASK_QUESTIONS[qkey],
                                                  [l.description for l in candidates], context), dtype=np.float64)

        support = np.array([commonsense.support.get(task, {}).get(l.id, 0.0) for l in candidates])
        if support.max(initial=0.0) < 1e-3:
            p_cn = np.full(len(candidates), 1.0 / len(candidates))
        else:
            p_cn = _softmax(support / max(cfg.commonsense_temperature, 1e-3))
        match = np.array([commonsense.draft_match.get(task, {}).get(l.id, 0.0) for l in candidates])

        logits = (cfg.w_vlm * np.log(p_vlm + 1e-9) + cfg.w_commonsense * np.log(p_cn + 1e-9)
                  + cfg.w_draft * match)
        probs = _softmax(logits)
        order = np.argsort(-probs)
        best = candidates[order[0]]
        margin = probs[order[0]] - (probs[order[1]] if len(order) > 1 else 0.0)

        alternatives = [LabelScore(label_id=candidates[i].id, label=display_name(candidates[i]),
                                   probability=float(probs[i]), p_vlm=float(p_vlm[i]),
                                   p_commonsense=float(p_cn[i]), draft_match=float(match[i]))
                        for i in order]
        i0 = order[0]
        evidence = [f"VLM visual probability {p_vlm[i0]:.2f}"]
        if match[i0] > 0 and draft_text:
            evidence.append(f"consistent with the VLM draft: \"{draft_text}\"")
        evidence += commonsense.evidence.get(task, {}).get(best.id, [])
        return TaskPrediction(task=task, label_id=best.id, label=display_name(best),
                              confidence=float(probs[i0]), ambiguous=bool(margin < cfg.ambiguity_margin),
                              alternatives=alternatives, evidence=evidence)

    def infer_activity(self, frame, context, visual, commonsense, draft) -> TaskPrediction:
        return self._infer_task("activity", frame, context, visual, commonsense, draft.activity)

    def infer_relationship(self, frame, context, visual, commonsense, draft, activity) -> TaskPrediction:
        if visual.people_count <= 1:
            label = self.taxonomy.get("relationship", "none_single")
            return TaskPrediction(task="relationship", label_id=label.id, label=display_name(label),
                                  confidence=1.0, ambiguous=False,
                                  alternatives=[LabelScore(label.id, display_name(label), 1.0, 1.0, 1.0, 0.0)],
                                  evidence=[f"only {visual.people_count} person detected"])
        if self.cfg.condition_on_previous:
            context = f"{context}\nThe people are {self.taxonomy.get('activity', activity.label_id).description}."
        return self._infer_task("relationship", frame, context, visual, commonsense, draft.relationship)

    def infer_intention(self, frame, context, visual, commonsense, draft, activity, relationship) -> TaskPrediction:
        if self.cfg.condition_on_previous:
            act = self.taxonomy.get("activity", activity.label_id).description
            context = f"{context}\nThe people are {act}."
            if relationship.label_id != "none_single":
                rel = self.taxonomy.get("relationship", relationship.label_id).description
                context += f" They are most likely {rel}."
        return self._infer_task("intention", frame, context, visual, commonsense, draft.intention)

    # ------------------------------------------------------------------ explanation / QA
    def generate_explanation(self, frame, activity: TaskPrediction, relationship: TaskPrediction,
                             intention: TaskPrediction, visual: VisualConcepts,
                             commonsense: CommonsenseResult) -> str:
        act = self.taxonomy.get("activity", activity.label_id).description
        rel = self.taxonomy.get("relationship", relationship.label_id).description
        intent = self.taxonomy.get("intention", intention.label_id).description
        if visual.people_count >= 2:
            claim = f"The people in this image are {act}. They are most likely {rel}, and their intention is {intent}."
        else:
            claim = f"The person in this image is involved in: {act}. Their likely goal is {intent}."
        facts = "\n".join(f"- {f.sentence()}." for f in commonsense.facts[:4])
        prompt = (f"{claim}\n" + (f"Relevant commonsense knowledge:\n{facts}\n" if facts else "") +
                  "In two or three sentences, explain which visible evidence in the image (people, gestures, "
                  "facial expressions, objects and setting) supports this interpretation.")
        return self.vlm.generate(frame, prompt, self.max_new_tokens_explanation).strip()

    def answer_question(self, frame, question: str, choices: Sequence[str] | None = None,
                        context: str = "") -> tuple[str, list[float] | None]:
        if choices:
            probs = self.vlm.score_options(frame, question, list(choices), context)
            return choices[int(np.argmax(probs))], [float(p) for p in probs]
        prompt = (f"{context}\n\n" if context else "") + f"Question: {question}\nAnswer briefly and give the visual reason."
        return self.vlm.generate(frame, prompt, self.max_new_tokens_explanation).strip(), None

    # ------------------------------------------------------------------ entry point
    def reason(self, frame: PreprocessedFrame, visual: VisualConcepts, draft: DraftReasoning,
               commonsense: CommonsenseResult, question: str | None = None,
               answer_choices: Sequence[str] | None = None) -> SocialPrediction:
        context = self.fuse(visual, commonsense, question)

        if visual.people_count == 0:
            return self._no_people(visual, draft, context, question)

        activity = self.infer_activity(frame, context, visual, commonsense, draft)
        relationship = self.infer_relationship(frame, context, visual, commonsense, draft, activity)
        intention = self.infer_intention(frame, context, visual, commonsense, draft, activity, relationship)

        explanation = ""
        if self.cfg.generate_explanation:
            explanation = self.generate_explanation(frame, activity, relationship, intention, visual, commonsense)

        answer, answer_probs = None, None
        if question:
            if answer_choices:
                answer, answer_probs = self.answer_question(frame, question, answer_choices, context)
            else:
                answer = draft.answer or self.answer_question(frame, question, None, context)[0]

        summary = f"{activity.label} | {relationship.label} | {intention.label}"
        # Commonsense evidence that supports the *final* labels (not just the most generic facts).
        evidence = [e for t in (activity, relationship, intention) for e in t.evidence
                    if e.startswith("ConceptNet:") or "defining concept" in e]
        evidence = list(dict.fromkeys(evidence))[:8] or [f.sentence() for f in commonsense.facts[:5]]
        return SocialPrediction(
            activity=activity, relationship=relationship, intention=intention,
            people_count=visual.people_count, summary=summary, explanation=explanation,
            evidence=evidence, context=context, question=question, answer=answer,
            answer_choices=list(answer_choices) if answer_choices else None,
            answer_probabilities=answer_probs,
        )

    def _no_people(self, visual: VisualConcepts, draft: DraftReasoning, context: str,
                   question: str | None) -> SocialPrediction:
        def fixed(task: str, lid: str, name: str) -> TaskPrediction:
            return TaskPrediction(task=task, label_id=lid, label=name, confidence=0.0, ambiguous=True,
                                  evidence=["no people detected in the frame"])
        return SocialPrediction(
            activity=fixed("activity", "no_interaction", "No Interaction"),
            relationship=fixed("relationship", "none_single", "No relationship (no people)"),
            intention=fixed("intention", "not_applicable", "Not applicable"),
            people_count=0, summary="No people detected - no social interaction",
            explanation=f"No people were detected. Scene: {visual.caption}",
            context=context, question=question, answer=draft.answer if question else None)
