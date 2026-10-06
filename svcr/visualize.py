"""Render a Phase I result as an annotated image (frame + side panel)."""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from .schemas import Phase1Result

PANEL_W = 520
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _conf_color(conf: float, ambiguous: bool) -> tuple[int, int, int]:
    if ambiguous or conf < 0.4:
        return (40, 110, 230)     # orange-ish (BGR)
    if conf < 0.7:
        return (30, 180, 220)     # amber
    return (80, 170, 60)          # green


def render_result(result: Phase1Result, frame_rgb: np.ndarray | None = None) -> np.ndarray:
    """Return a BGR image with the frame on the left and the prediction panel on the right."""
    rgb = frame_rgb if frame_rgb is not None else result.frame.image
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h = max(img.shape[0], 640)
    if img.shape[0] < h:
        img = cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    panel = np.full((h, PANEL_W, 3), 250, np.uint8)
    y = 34

    def put(text: str, scale=0.52, color=(40, 40, 40), bold=False, wrap=60, gap=22):
        nonlocal y
        for line in textwrap.wrap(text, wrap) or [""]:
            if y > h - 12:
                return
            cv2.putText(panel, line, (16, y), FONT, scale, color, 2 if bold else 1, cv2.LINE_AA)
            y += gap

    p = result.prediction
    put("Explainable Social VCR - Phase I", 0.62, (90, 50, 20), True)
    put(f"People detected: {p.people_count}", 0.5, (90, 90, 90))
    y += 6
    for task in (p.activity, p.relationship, p.intention):
        flag = "  (ambiguous)" if task.ambiguous else ""
        put(f"{task.task.upper()}: {task.label}", 0.56, _conf_color(task.confidence, task.ambiguous), True, wrap=44)
        put(f"confidence {task.confidence:.0%}{flag}", 0.45, (110, 110, 110))
        alts = ", ".join(f"{a.label} {a.probability:.0%}" for a in task.alternatives[1:3])
        if alts:
            put(f"alternatives: {alts}", 0.42, (130, 130, 130), wrap=66, gap=19)
        y += 6
    if p.explanation:
        put("EXPLANATION", 0.5, (90, 50, 20), True)
        put(p.explanation, 0.45, wrap=66, gap=19)
        y += 6
    if p.evidence and result.commonsense.available:
        put(f"COMMONSENSE EVIDENCE (ConceptNet, {result.commonsense.backend})", 0.5, (90, 50, 20), True)
        for e in p.evidence[:5]:
            put(f"- {e.removeprefix('ConceptNet: ')}", 0.43, wrap=68, gap=19)
    if p.question:
        y += 6
        put(f"Q: {p.question}", 0.47, (90, 50, 20), True, wrap=60)
        put(f"A: {p.answer}", 0.45, wrap=66, gap=19)
    return np.hstack([img, panel])


def draw_overlay(bgr: np.ndarray, result: Phase1Result | None, status: str = "") -> np.ndarray:
    """Compact overlay for live video."""
    out = bgr.copy()
    lines = [status] if status else []
    if result is not None:
        p = result.prediction
        lines += [f"Activity: {p.activity.label} ({p.activity.confidence:.0%})",
                  f"Relationship: {p.relationship.label} ({p.relationship.confidence:.0%})",
                  f"Intention: {p.intention.label}"[:70]]
    if not lines:
        return out
    box_h = 14 + 26 * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, 28 + 26 * i), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out
