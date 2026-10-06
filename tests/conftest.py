import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svcr.conceptnet.client import ConceptNetBackend, ConceptNetClient, Edge  # noqa: E402
from svcr.config import ConceptNetConfig  # noqa: E402
from svcr.schemas import DraftReasoning, VisualConcepts  # noqa: E402
from svcr.taxonomy import Taxonomy  # noqa: E402

TOY_EDGES = [
    Edge("UsedFor", "handshake", "greeting", 3.0, "[[a handshake]] is used for [[greeting]]"),
    Edge("RelatedTo", "handshake", "greet", 2.0),
    Edge("HasSubevent", "meet", "shake_hands", 2.5),
    Edge("RelatedTo", "shake_hands", "greeting", 2.0),
    Edge("RelatedTo", "suit", "business", 2.0),
    Edge("AtLocation", "suit", "office", 1.5),
    Edge("RelatedTo", "business", "colleague", 1.5),
    Edge("RelatedTo", "handshake", "agreement", 2.0),
    Edge("RelatedTo", "greeting", "hello", 2.0),
    Edge("MotivatedByGoal", "greet", "acknowledge", 2.0),
    Edge("RelatedTo", "smile", "happy", 2.0),
    Edge("RelatedTo", "hug", "embrace", 3.0),
    Edge("RelatedTo", "punch", "fight", 3.0),
    Edge("Antonym", "handshake", "fight", 1.0),
    Edge("RelatedTo", "office", "work", 2.0),
    Edge("RelatedTo", "person", "handshake", 1.0),
]


class ToyBackend(ConceptNetBackend):
    name = "toy"

    def __init__(self, edges=TOY_EDGES):
        self.edges_list = edges

    def edges(self, term, limit):
        return [e for e in self.edges_list if term in (e.start, e.end)][:limit]


@pytest.fixture
def taxonomy():
    return Taxonomy.load(ROOT / "configs" / "taxonomy.yaml")


@pytest.fixture
def toy_client():
    return ConceptNetClient(ToyBackend(), ConceptNetConfig())


@pytest.fixture
def handshake_visual():
    return VisualConcepts(
        caption="Two men in suits are shaking hands in an office. Both are smiling.",
        people_count=2, people=["man in a suit", "businessman"], objects=["suit", "desk"],
        actions=["shaking hands", "smiling"], gestures=["smile", "handshake"], setting="office",
    )


@pytest.fixture
def handshake_draft():
    return DraftReasoning(activity="Shaking hands", relationship="business colleagues",
                          intention="They are greeting each other before a meeting.",
                          rationale="Their hands are clasped and both are smiling.")


class FakeVLM:
    """Deterministic stand-in for SmolVLMEngine used by the tests."""

    model_id = "fake-vlm"

    def __init__(self, visual, draft, preferred=("greeting", "colleagues", "greet")):
        self.visual, self.draft, self.preferred = visual, draft, preferred
        self.calls = []

    def run(self, frame, question=None):
        return self.visual, self.draft

    def generate(self, image, prompt, max_new_tokens=None):
        self.calls.append(("generate", prompt))
        return "The two men are clasping hands and smiling, which shows a friendly greeting."

    def score_options(self, image, question, options, context=""):
        self.calls.append(("score", question, list(options), context))
        scores = np.array([2.0 if any(p in o for p in self.preferred) else 0.0 for o in options])
        e = np.exp(scores - scores.max())
        return e / e.sum()
