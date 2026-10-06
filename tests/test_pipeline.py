import json

import numpy as np

from svcr.config import load_config
from svcr.pipeline import Phase1Pipeline
from svcr.schemas import VisualConcepts
from svcr.visualize import render_result

from conftest import FakeVLM


def _pipeline(toy_client, visual, draft, preferred=("greeting", "colleagues", "greet")):
    cfg = load_config(overrides=["preprocess.target_longest_side=384"])
    return Phase1Pipeline(cfg, vlm=FakeVLM(visual, draft, preferred), conceptnet_client=toy_client)


def test_end_to_end(toy_client, handshake_visual, handshake_draft):
    pipe = _pipeline(toy_client, handshake_visual, handshake_draft)
    img = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipe.run(img, question="Why are they shaking hands?")
    p = result.prediction
    assert p.activity.label_id == "greeting"
    assert p.relationship.label_id == "colleagues"
    assert p.intention.label_id == "greet_acknowledge"
    assert 0.0 < p.activity.confidence <= 1.0
    assert abs(sum(a.probability for a in p.activity.alternatives) - 1.0) < 1e-6
    assert "Commonsense knowledge" in p.context
    assert p.explanation and p.answer
    data = json.loads(result.to_json())
    assert data["prediction"]["activity"]["label_id"] == "greeting"
    assert "image" not in data["frame"] and "tensor" not in data["frame"]
    assert set(data["timings_s"]) >= {"module1_preprocess", "module2_vlm", "module3_conceptnet",
                                      "module4_reasoning", "total"}
    panel = render_result(result)
    assert panel.shape[1] > result.frame.image.shape[1]


def test_commonsense_breaks_vlm_tie(toy_client, handshake_visual, handshake_draft):
    """When the VLM is undecided, ConceptNet + draft evidence must pick the grounded label."""
    pipe = _pipeline(toy_client, handshake_visual, handshake_draft, preferred=())
    result = pipe.run(np.zeros((200, 200, 3), np.uint8) + 128)
    assert result.prediction.activity.label_id == "greeting"


def test_single_person_has_no_relationship(toy_client, handshake_draft):
    visual = VisualConcepts(caption="A woman reading a book.", people_count=1, actions=["reading"])
    pipe = _pipeline(toy_client, visual, handshake_draft)
    result = pipe.run(np.zeros((200, 200, 3), np.uint8) + 100)
    assert result.prediction.relationship.label_id == "none_single"


def test_no_people(toy_client, handshake_draft):
    visual = VisualConcepts(caption="An empty street.", people_count=0)
    pipe = _pipeline(toy_client, visual, handshake_draft)
    result = pipe.run(np.zeros((200, 200, 3), np.uint8) + 100)
    assert result.prediction.people_count == 0
    assert "No people" in result.prediction.summary


def test_multiple_choice_question(toy_client, handshake_visual, handshake_draft):
    pipe = _pipeline(toy_client, handshake_visual, handshake_draft, preferred=("greet",))
    result = pipe.run(np.zeros((200, 200, 3), np.uint8) + 100, question="What are they doing?",
                      answer_choices=["They are fighting.", "They greet each other."])
    assert result.prediction.answer == "They greet each other."
    assert len(result.prediction.answer_probabilities) == 2
