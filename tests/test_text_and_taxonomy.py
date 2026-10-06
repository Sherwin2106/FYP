from svcr.taxonomy import lexical_match
from svcr.text import clean_phrase, parse_count, split_list, to_concept


def test_split_list():
    assert split_list("a cup, two plates and a knife.") == ["cup", "plates", "knife"]
    assert split_list("") == []


def test_parse_count():
    assert parse_count("There are 3 people") == 3
    assert parse_count("Two.") == 2
    assert parse_count("unclear") is None


def test_concept_normalisation():
    assert to_concept("Shake hands!") == "shake_hands"
    assert clean_phrase("- The red car.") == "red car"


def test_taxonomy_loads_all_tasks(taxonomy):
    for task in ("activity", "relationship", "intention"):
        ids = [l.id for l in taxonomy[task]]
        assert len(ids) == len(set(ids)) >= 10
        assert len(ids) <= 26          # must fit A-Z options
    assert taxonomy.has("activity", "no_interaction")
    assert taxonomy.has("relationship", "none_single")


def test_lexical_match(taxonomy):
    greeting = taxonomy.get("activity", "greeting")
    eating = taxonomy.get("activity", "eating_together")
    assert lexical_match("shaking hands", greeting) > 0
    assert lexical_match("having dinner at a restaurant", eating) > 0
    assert lexical_match("shaking hands", eating) == 0
