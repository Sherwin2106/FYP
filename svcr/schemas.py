"""Data structures passed between the Phase I modules."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TASKS = ("activity", "relationship", "intention")


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if not f.metadata.get("exclude")}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- Module 1
@dataclass
class PreprocessedFrame:
    """Output of Module 1."""

    image: np.ndarray = field(metadata={"exclude": True})   # RGB uint8, HxWx3 (processed size)
    tensor: Any = field(metadata={"exclude": True})         # torch.FloatTensor 1x3xHxW, normalised
    original_size: tuple[int, int] = (0, 0)                 # (width, height)
    processed_size: tuple[int, int] = (0, 0)                # (width, height)
    scale: float = 1.0
    quality: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    frame_index: int | None = None
    timestamp_s: float | None = None

    def to_pil(self):
        from PIL import Image
        return Image.fromarray(self.image)

    def summary(self) -> dict[str, Any]:
        return _jsonable(self)


# --------------------------------------------------------------------------- Module 2
@dataclass
class VisualConcepts:
    """Visual understanding produced by the VLM (Module 2A)."""

    caption: str = ""
    people_count: int = 0
    people: list[str] = field(default_factory=list)        # person descriptions / roles
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    gestures: list[str] = field(default_factory=list)      # expressions, gestures, posture
    setting: str = ""
    spatial_relations: list[str] = field(default_factory=list)
    raw_answers: dict[str, str] = field(default_factory=dict)
    # Tidied copies for the UI only (complete sentences, model's direct answers);
    # Modules 3-4 always use the full fields above.
    display_caption: str = ""
    display_people: list[str] = field(default_factory=list)
    display_objects: list[str] = field(default_factory=list)
    display_actions: list[str] = field(default_factory=list)


@dataclass
class DraftReasoning:
    """Draft prediction and explanation produced by the VLM (Module 2B)."""

    activity: str = ""
    relationship: str = ""
    intention: str = ""
    rationale: str = ""
    question: str | None = None
    answer: str | None = None


# --------------------------------------------------------------------------- Module 3
@dataclass
class Fact:
    start: str
    relation: str
    end: str
    weight: float
    surface: str = ""
    source_term: str = ""
    relevance: float = 0.0

    def sentence(self) -> str:
        from .conceptnet.relations import fact_to_sentence
        return fact_to_sentence(self)


@dataclass
class CommonsenseResult:
    """Output of Module 3."""

    available: bool = False
    backend: str = "offline"
    terms: dict[str, float] = field(default_factory=dict)                      # term -> salience
    facts: list[Fact] = field(default_factory=list)
    support: dict[str, dict[str, float]] = field(default_factory=dict)         # task -> label -> [0,1]
    draft_match: dict[str, dict[str, float]] = field(default_factory=dict)     # task -> label -> [0,1]
    evidence: dict[str, dict[str, list[str]]] = field(default_factory=dict)    # task -> label -> reasons
    disambiguated: dict[str, str | None] = field(default_factory=dict)         # task -> label id


# --------------------------------------------------------------------------- Module 4
@dataclass
class LabelScore:
    label_id: str
    label: str
    probability: float
    p_vlm: float
    p_commonsense: float
    draft_match: float


@dataclass
class TaskPrediction:
    task: str
    label_id: str
    label: str
    confidence: float
    ambiguous: bool
    alternatives: list[LabelScore] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class SocialPrediction:
    """Output of Module 4."""

    activity: TaskPrediction
    relationship: TaskPrediction
    intention: TaskPrediction
    people_count: int
    summary: str
    explanation: str
    evidence: list[str] = field(default_factory=list)
    context: str = ""
    question: str | None = None
    answer: str | None = None
    answer_choices: list[str] | None = None
    answer_probabilities: list[float] | None = None


@dataclass
class Phase1Result:
    frame: PreprocessedFrame
    visual: VisualConcepts
    draft: DraftReasoning
    commonsense: CommonsenseResult
    prediction: SocialPrediction
    timings_s: dict[str, float] = field(default_factory=dict)
    model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
