"""Social-interaction label space (activity / relationship / intention)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

from .schemas import TASKS
from .text import token_set, to_concept


@dataclass
class Label:
    id: str
    task: str
    description: str
    anchors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.id.replace("_", " ")

    @cached_property
    def anchor_concepts(self) -> list[str]:
        return [to_concept(a) for a in self.anchors]

    @cached_property
    def lexicon(self) -> set[str]:
        """Lemmatised words used for lexical matching against free-form drafts."""
        words: set[str] = set()
        for text in [self.id.replace("_", " "), *self.keywords, *[a.replace("_", " ") for a in self.anchors]]:
            words |= token_set(text)
        return words


class Taxonomy:
    def __init__(self, labels: dict[str, list[Label]]):
        missing = [t for t in TASKS if not labels.get(t)]
        if missing:
            raise ValueError(f"Taxonomy is missing tasks: {missing}")
        self.labels = labels
        self._by_id = {(l.task, l.id): l for task in labels.values() for l in task}
        # document frequency of each lexicon word within a task: shared words are less informative
        self.df: dict[str, dict[str, int]] = {}
        for task, task_labels in labels.items():
            df: dict[str, int] = {}
            for l in task_labels:
                for w in l.lexicon:
                    df[w] = df.get(w, 0) + 1
            self.df[task] = df

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        labels = {
            task: [Label(id=item["id"], task=task, description=item.get("description", item["id"]),
                         anchors=item.get("anchors", []), keywords=item.get("keywords", []))
                   for item in data.get(task, [])]
            for task in TASKS
        }
        return cls(labels)

    def get(self, task: str, label_id: str) -> Label:
        return self._by_id[(task, label_id)]

    def has(self, task: str, label_id: str) -> bool:
        return (task, label_id) in self._by_id

    def __getitem__(self, task: str) -> list[Label]:
        return self.labels[task]

    def lexical_match(self, text: str, label: Label) -> float:
        return lexical_match(text, label, self.df.get(label.task))

    def all_anchor_concepts(self) -> set[str]:
        return {c for labels in self.labels.values() for l in labels for c in l.anchor_concepts}


def lexical_match(text: str, label: Label, df: dict[str, int] | None = None) -> float:
    """Overlap between a free-form phrase and a label's lexicon, in [0, 1].

    With ``df`` (word -> number of labels using it) each hit is weighted by 1/df,
    so a word shared by several labels ("hand") counts less than a distinctive one ("shake").
    """
    words = token_set(text)
    if not words:
        return 0.0
    hits = words & label.lexicon
    if not hits:
        return 0.0
    score = sum(1.0 / (df.get(w, 1) if df else 1) for w in hits)
    # Recall of the phrase's content words, softened so a single strong hit still counts.
    return min(1.0, score / min(len(words), 3))
