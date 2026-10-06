"""Text utilities: term extraction, lemmatisation and ConceptNet normalisation.

spaCy (``en_core_web_sm``) is used when installed. A small rule-based fallback
keeps the pipeline working if it is not.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable

log = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "his", "her",
    "their", "its", "our", "my", "your", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "it", "they", "them",
    "he", "she", "we", "you", "i", "who", "which", "what", "there", "here", "as", "into", "while",
    "also", "very", "other", "another", "one", "two", "both", "all", "such", "than", "then", "so",
    "not", "no", "yes", "can", "could", "would", "should", "may", "might", "will", "do", "does",
}

# Words that describe the picture rather than its content.
NON_CONTENT_TERMS = {
    "image", "picture", "photo", "photograph", "scene", "background", "foreground", "view", "frame",
    "shot", "side", "left", "right", "front", "top", "bottom", "middle", "center", "centre", "part",
    "area", "thing", "something", "someone", "somebody", "type", "kind", "lot", "way", "time",
    "moment", "number", "group", "detail", "overall", "focus", "atmosphere", "sense", "appearance",
    "color", "colour", "shade", "light", "lighting", "setting", "position", "posture", "expression",
    "gesture", "body", "look", "figure", "element", "feature", "style", "attention", "camera angle",
}

LIGHT_VERBS = {
    "be", "have", "do", "appear", "seem", "show", "depict", "feature", "suggest", "indicate",
    "include", "contain", "visible", "see", "can", "could", "would", "get", "make", "take", "go",
    "describe", "capture", "surround", "wear", "dress",
}

PERSON_NOUNS = {
    "person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
    "kids", "guy", "lady", "gentleman", "individual", "adult", "teenager", "baby", "toddler",
    "human", "crowd", "someone", "couple",
}

SPATIAL_PREPS = {
    "next to", "beside", "behind", "in front of", "near", "across from", "opposite", "facing",
    "between", "around", "above", "below", "under", "on top of", "towards", "toward", "with",
    "against", "along", "among",
}

NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "single": 1, "a": 1, "an": 1, "two": 2, "couple": 2,
    "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "several": 3, "few": 3, "many": 6, "group": 4,
    "crowd": 10, "dozen": 12,
}

_IRREGULAR_LEMMAS = {
    "men": "man", "women": "woman", "children": "child", "people": "person", "feet": "foot",
    "teeth": "tooth", "mice": "mouse", "geese": "goose", "shaking": "shake", "sitting": "sit",
    "running": "run", "hugging": "hug", "shopping": "shop", "chatting": "chat", "dancing": "dance",
    "smiling": "smile", "holding": "hold", "waving": "wave", "hitting": "hit", "getting": "get",
    "putting": "put", "cutting": "cut", "clapping": "clap", "wrapping": "wrap", "hugged": "hug",
    "sat": "sit", "held": "hold", "ate": "eat", "eaten": "eat", "spoke": "speak", "told": "tell",
    "gave": "give", "given": "give", "fought": "fight", "taught": "teach", "bought": "buy",
    "sold": "sell", "met": "meet", "kids": "kid", "glasses": "glass", "dresses": "dress",
}


@lru_cache(maxsize=1)
def get_nlp():
    """Return a spaCy pipeline, or None when spaCy / the model is unavailable."""
    try:
        import spacy
    except ImportError:
        log.warning("spaCy not installed - using rule-based text processing fallback.")
        return None
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(name, disable=["ner"])
        except OSError:
            continue
    log.warning("No spaCy English model found (run: python -m spacy download en_core_web_sm). "
                "Using rule-based fallback.")
    return None


def _rule_lemma(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            stem = stem[:-1]
        return stem
    if len(w) > 4 and w.endswith("ed") and not w.endswith("eed"):
        return w[:-2] if not w.endswith("ied") else w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def lemmatize(text: str) -> str:
    """Lemmatise a short phrase ("shaking hands" -> "shake hand")."""
    text = clean_phrase(text)
    if not text:
        return ""
    nlp = get_nlp()
    if nlp is not None:
        return " ".join(t.lemma_.lower() for t in nlp(text) if not t.is_punct).strip()
    return " ".join(_rule_lemma(w) for w in text.split())


def clean_phrase(text: str) -> str:
    """Lower-case, strip punctuation, bullets and leading determiners."""
    text = text.lower().strip()
    text = re.sub(r"^[\s\-\*•\d\.\)]+", "", text)          # bullets / numbering
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    words = [w for w in text.replace("-", " ").split() if w]
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return " ".join(words)


def to_concept(term: str) -> str:
    """Normalise a phrase to ConceptNet's term format ("Shake hands" -> "shake_hands")."""
    return "_".join(clean_phrase(term).split())


def concept_to_text(concept: str) -> str:
    return concept.replace("_", " ")


def split_list(text: str) -> list[str]:
    """Split a model answer like "a cup, two plates and a knife." into items."""
    if not text:
        return []
    text = re.sub(r"\s+and\s+", ",", text.replace("\n", ","), flags=re.I)
    items = []
    for part in re.split(r"[,;/]", text):
        item = clean_phrase(part)
        item = re.sub(r"^(?:\d+|one|two|three|four|five|several|some|many)\s+", "", item)
        if item and len(item.split()) <= 5 and item not in NON_CONTENT_TERMS:
            items.append(item)
    return dedupe(items)


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def parse_count(text: str) -> int | None:
    """Parse "There are 3 people" / "two" -> int."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS and word not in {"a", "an"}:
            return NUMBER_WORDS[word]
    return None


def extract_terms(text: str) -> dict[str, list[str]]:
    """Extract salient nouns, noun phrases, verbs and adjectives from free text.

    Returns a dict with keys ``nouns``, ``verbs``, ``adjectives``, ``people``.
    """
    out: dict[str, list[str]] = {"nouns": [], "verbs": [], "adjectives": [], "people": []}
    if not text:
        return out
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            root = chunk.root.lemma_.lower()
            if root in PERSON_NOUNS or chunk.root.text.lower() in PERSON_NOUNS:
                out["people"].append(clean_phrase(chunk.text))
                continue
            if root in STOPWORDS or root in NON_CONTENT_TERMS or not root.isalpha():
                continue
            mods = [t.lemma_.lower() for t in chunk if t.dep_ == "compound" and t.is_alpha]
            phrase = " ".join(mods + [root])
            out["nouns"].append(phrase)
            if phrase != root:
                out["nouns"].append(root)
        for tok in doc:
            lemma = tok.lemma_.lower()
            if tok.pos_ == "VERB" and lemma not in LIGHT_VERBS and tok.is_alpha:
                particle = [c.text.lower() for c in tok.children if c.dep_ == "prt"]
                objs = [c.lemma_.lower() for c in tok.children
                        if c.dep_ in ("dobj", "obj") and c.is_alpha and c.lemma_.lower() not in STOPWORDS]
                objs = [o for o in objs if o not in NON_CONTENT_TERMS and o not in PERSON_NOUNS]
                # "shake hand" is far less ambiguous than bare "shake" (milkshake, fan, ...)
                out["verbs"].append(f"{lemma} {objs[0]}" if objs else " ".join([lemma] + particle))
            elif tok.pos_ == "ADJ" and tok.is_alpha and lemma not in STOPWORDS:
                out["adjectives"].append(lemma)
    else:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        for w in words:
            if w in STOPWORDS or w in NON_CONTENT_TERMS:
                continue
            lemma = _rule_lemma(w)
            if w in PERSON_NOUNS or lemma in PERSON_NOUNS:
                out["people"].append(lemma)
            elif w.endswith("ing"):
                if lemma not in LIGHT_VERBS:
                    out["verbs"].append(lemma)
            elif len(w) > 2:
                out["nouns"].append(lemma)
    return {k: dedupe(v) for k, v in out.items()}


def extract_spatial_relations(text: str, limit: int = 6) -> list[str]:
    """Extract short "X <preposition> Y" phrases from a description."""
    relations: list[str] = []
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text)
        for tok in doc:
            if tok.dep_ != "prep":
                continue
            prep = tok.text.lower()
            # multi-word prepositions: "next to", "in front of"
            if tok.i + 1 < len(doc) and f"{prep} {doc[tok.i + 1].text.lower()}" in SPATIAL_PREPS:
                continue
            if tok.i >= 1 and f"{doc[tok.i - 1].text.lower()} {prep}" in SPATIAL_PREPS:
                prep = f"{doc[tok.i - 1].text.lower()} {prep}"
            if tok.i >= 2 and f"{doc[tok.i - 2].text.lower()} {doc[tok.i - 1].text.lower()} {prep}" in SPATIAL_PREPS:
                prep = f"{doc[tok.i - 2].text.lower()} {doc[tok.i - 1].text.lower()} {prep}"
            if prep not in SPATIAL_PREPS:
                continue
            pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
            head = tok.head
            for _ in range(3):   # climb out of "next (to)", "in front (of)", "on top (of)"
                if head.pos_ in ("ADV", "ADP") or head.lemma_.lower() in {"front", "top", "next"}:
                    head = head.head
            if head.pos_ in ("VERB", "AUX"):
                subj = next((c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                head = subj or head
            if pobj is None or head.pos_ not in ("NOUN", "PROPN", "PRON", "VERB"):
                continue
            rel = f"{head.lemma_.lower()} {prep} {pobj.lemma_.lower()}"
            if pobj.lemma_.lower() not in NON_CONTENT_TERMS:
                relations.append(rel)
    else:
        pattern = r"\b([a-z]+)\s+(" + "|".join(sorted(map(re.escape, SPATIAL_PREPS), key=len, reverse=True)) + r")\s+(?:the |a |an |his |her |their )?([a-z]+)"
        for m in re.finditer(pattern, text.lower()):
            if m.group(3) not in NON_CONTENT_TERMS:
                relations.append(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    return dedupe(relations)[:limit]


def token_set(text: str) -> set[str]:
    """Lemmatised content-word set used for lexical matching."""
    lemmas = lemmatize(text).split()
    return {w for w in lemmas if w not in STOPWORDS and len(w) > 1}
