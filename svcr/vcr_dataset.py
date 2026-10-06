"""Loader for the VCR dataset (Zellers et al., 2019) - https://visualcommonsense.com

Expected layout (after downloading and unzipping the official release):

    <vcr_root>/
        train.jsonl  val.jsonl  test.jsonl
        vcr1images/<movie>/<image>.jpg
        vcr1images/<movie>/<image>.json      # boxes / segmentation metadata

Questions, answers and rationales refer to detected objects by index
(e.g. ``[[0], "is", "angry", "at", [1]]``). Following the VCR convention, each
referenced object is drawn on the image with a numbered tag, and the text uses
the same tag (``person1``), so the VLM can ground the reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

TAG_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
              (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]


@dataclass
class VCRItem:
    annot_id: str
    image_path: Path
    objects: list[str]
    boxes: list[list[float]]
    question: str
    answer_choices: list[str]
    answer_label: int | None
    rationale_choices: list[str]
    rationale_label: int | None
    referenced: set[int] = field(default_factory=set)


def object_tag(objects: list[str], idx: int) -> str:
    return f"{objects[idx]}{idx + 1}" if 0 <= idx < len(objects) else f"object{idx + 1}"


def tokens_to_text(tokens: list, objects: list[str], referenced: set[int]) -> str:
    words = []
    for tok in tokens:
        if isinstance(tok, list):
            referenced.update(tok)
            tags = [object_tag(objects, i) for i in tok]
            words.append(tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1])
        else:
            words.append(str(tok))
    text = " ".join(words)
    text = re.sub(r" ([?.!,;:'])", r"\1", text)
    text = re.sub(r" (n't|'s|'re|'m|'ll|'ve|'d)\b", r"\1", text)
    return text.strip()


def load_vcr(root: str | Path, split: str = "val", limit: int | None = None) -> Iterator[VCRItem]:
    root = Path(root)
    ann_path = root / f"{split}.jsonl"
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} not found - download VCR from https://visualcommonsense.com")
    img_root = root / "vcr1images"
    with open(ann_path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                break
            d = json.loads(line)
            objects = d["objects"]
            refs: set[int] = set()
            question = tokens_to_text(d["question"], objects, refs)
            answers = [tokens_to_text(a, objects, refs) for a in d["answer_choices"]]
            rationales = [tokens_to_text(r, objects, refs) for r in d.get("rationale_choices", [])]
            meta_path = img_root / d["metadata_fn"]
            boxes = []
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as mf:
                    boxes = json.load(mf).get("boxes", [])
            yield VCRItem(
                annot_id=d.get("annot_id", str(n)), image_path=img_root / d["img_fn"], objects=objects,
                boxes=boxes, question=question, answer_choices=answers,
                answer_label=d.get("answer_label"), rationale_choices=rationales,
                rationale_label=d.get("rationale_label"), referenced=refs,
            )


def draw_tags(item: VCRItem, only_referenced: bool = True) -> np.ndarray:
    """Return the VCR image (BGR) with numbered boxes for the referenced objects."""
    img = cv2.imread(str(item.image_path))
    if img is None:
        raise FileNotFoundError(item.image_path)
    idxs = sorted(item.referenced) if only_referenced else range(len(item.boxes))
    thick = max(2, img.shape[1] // 400)
    for i in idxs:
        if i >= len(item.boxes):
            continue
        x1, y1, x2, y2 = map(int, item.boxes[i][:4])
        color = TAG_COLORS[i % len(TAG_COLORS)][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag = object_tag(item.objects, i)
        scale = max(0.6, img.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, tag, (x1 + 4, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img
