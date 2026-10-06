"""Natural-language rendering of ConceptNet relations."""

from __future__ import annotations

import re

TEMPLATES = {
    "RelatedTo": "{s} is related to {e}",
    "IsA": "{s} is a kind of {e}",
    "PartOf": "{s} is part of {e}",
    "HasA": "{s} has {e}",
    "UsedFor": "{s} is used for {e}",
    "CapableOf": "{s} can {e}",
    "AtLocation": "{s} is typically found at {e}",
    "Causes": "{s} causes {e}",
    "HasSubevent": "{s} involves {e}",
    "HasFirstSubevent": "{s} starts with {e}",
    "HasLastSubevent": "{s} ends with {e}",
    "HasPrerequisite": "{s} requires {e}",
    "HasProperty": "{s} is {e}",
    "MotivatedByGoal": "you {s} because you want to {e}",
    "Desires": "{s} wants {e}",
    "CausesDesire": "{s} makes you want to {e}",
    "CreatedBy": "{s} is created by {e}",
    "Synonym": "{s} means the same as {e}",
    "SimilarTo": "{s} is similar to {e}",
    "DefinedAs": "{s} is defined as {e}",
    "MannerOf": "{s} is a way of {e}",
    "Entails": "{s} entails {e}",
    "LocatedNear": "{s} is usually near {e}",
    "ReceivesAction": "{s} can be {e}",
    "HasContext": "{s} is used in the context of {e}",
    "DerivedFrom": "{s} is derived from {e}",
    "Antonym": "{s} is the opposite of {e}",
    "DistinctFrom": "{s} is distinct from {e}",
    "NotDesires": "{s} does not want {e}",
    "NotCapableOf": "{s} cannot {e}",
    "NotUsedFor": "{s} is not used for {e}",
}


def clean_surface(surface: str) -> str:
    return re.sub(r"\[\[|\]\]", "", surface or "").strip()


def fact_to_sentence(fact) -> str:
    s, e = fact.start.replace("_", " "), fact.end.replace("_", " ")
    template = TEMPLATES.get(fact.relation)
    if template:
        return template.format(s=s, e=e)
    if fact.surface:
        return clean_surface(fact.surface)
    return f"{s} {fact.relation} {e}"


def fact_to_triple(fact) -> str:
    return f"{fact.start.replace('_', ' ')} --{fact.relation}--> {fact.end.replace('_', ' ')}"
