"""Module 3 - ConceptNet Commonsense Enrichment.

    terms        <- extract_key_terms(visual_concepts, draft)
    for term in terms:
        facts    <- ConceptNet.query(term)
        relevant <- filter_relevant(facts)
    disambiguated <- resolve_ambiguity(visual_concepts, draft, commonsense)

Beyond returning human-readable facts, the module scores how strongly the
commonsense graph links the observed evidence to every candidate label of the
social-interaction taxonomy. For an evidence term *e* and a label *L* (described
by ConceptNet anchor concepts):

  * exact    - *e* is one of L's anchors                               -> 1.0
  * direct   - a ConceptNet edge links *e* to an anchor of L           -> rel_w * (1 - exp(-weight))
  * two-hop  - *e* and L's anchors share ConceptNet neighbours          -> scaled cosine of neighbour profiles
  * keyword  - lemma overlap with L's keywords (works even offline)     -> 0.6
  * negative - Antonym / DistinctFrom / Not* edge to an anchor          -> -0.5

    support(L) = sum_e salience(e) * kind_weight(task, e) * score(e, L) / normaliser

The per-task support distribution and the evidence paths behind it are handed
to Module 4 for fusion and to the explanation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ..config import ConceptNetConfig
from ..schemas import TASKS, CommonsenseResult, DraftReasoning, Fact, VisualConcepts
from ..taxonomy import Label, Taxonomy
from ..text import PERSON_NOUNS, clean_phrase, extract_terms, lemmatize, to_concept, token_set
from .client import ConceptNetClient, Edge
from .relations import fact_to_sentence

log = logging.getLogger(__name__)

# Hub concepts connected to almost everything - they carry no discriminative signal.
GENERIC_HUBS = {
    "person", "people", "human", "thing", "object", "something", "someone", "man", "woman", "it",
    "one", "action", "activity", "place", "time", "way", "act", "event", "being", "entity", "noun",
    "verb", "word", "english", "part", "something_you_do", "good", "bad", "fun", "anything",
}

SALIENCE = {
    "action": 1.0, "gesture": 0.9, "object": 0.7, "setting": 0.6, "role": 0.9, "person": 0.25,
    "caption": 0.5, "draft_activity": 0.8, "draft_relationship": 0.8, "draft_intention": 0.7,
    "question": 0.6,
}

# How much each kind of evidence matters for each task.
TASK_KIND_WEIGHTS = {
    "activity": {"action": 1.0, "gesture": 0.8, "object": 0.8, "setting": 0.6, "role": 0.5,
                 "person": 0.2, "caption": 0.5, "draft_activity": 0.6, "draft_relationship": 0.2,
                 "draft_intention": 0.4, "question": 0.5},
    "relationship": {"action": 0.6, "gesture": 0.6, "object": 0.4, "setting": 0.7, "role": 1.0,
                     "person": 0.6, "caption": 0.5, "draft_activity": 0.4, "draft_relationship": 0.6,
                     "draft_intention": 0.3, "question": 0.5},
    "intention": {"action": 0.8, "gesture": 0.8, "object": 0.5, "setting": 0.5, "role": 0.4,
                  "person": 0.2, "caption": 0.4, "draft_activity": 0.6, "draft_relationship": 0.3,
                  "draft_intention": 0.6, "question": 0.5},
}

# Only these relations say what a term *is*; for them an edge into a phrase that contains an
# anchor word ("handshake IsA form_of_greeting") counts as a link to that anchor.
DEFINITIONAL = {"IsA", "Synonym", "MannerOf", "DefinedAs", "SimilarTo", "Entails"}

SUPPORT_FLOOR = 0.3

DRAFT_KIND = {"activity": "draft_activity", "relationship": "draft_relationship", "intention": "draft_intention"}


@dataclass
class TermInfo:
    phrase: str
    salience: float
    kinds: set[str] = field(default_factory=set)
    concept: str = ""
    edges: list[Edge] = field(default_factory=list)
    profile: dict[str, float] = field(default_factory=dict)       # neighbour -> strength
    best_edge: dict[str, Edge] = field(default_factory=dict)      # neighbour -> strongest edge
    negatives: set[str] = field(default_factory=set)
    phrase_index: dict[str, list[str]] = field(default_factory=dict)  # word -> multi-word neighbours
    backed_off: bool = False        # resolved to a shorter node ("crossed arms" -> "arm")


def _squash(weight: float) -> float:
    return 1.0 - math.exp(-max(weight, 0.0))


def _cosine(a: dict[str, float], b: dict[str, float], na: float, nb: float) -> tuple[float, list[str]]:
    if not a or not b or na == 0 or nb == 0:
        return 0.0, []
    if len(a) > len(b):
        a, b = b, a
    shared = [(k, v * b[k]) for k, v in a.items() if k in b]
    dot = sum(v for _, v in shared)
    shared.sort(key=lambda kv: -kv[1])
    return dot / (na * nb), [k for k, _ in shared[:3]]


class CommonsenseEnricher:
    def __init__(self, client: ConceptNetClient, taxonomy: Taxonomy, cfg: ConceptNetConfig | None = None):
        self.client = client
        self.taxonomy = taxonomy
        self.cfg = cfg or client.cfg
        self._label_profiles: dict[tuple[str, str], tuple[dict[str, float], float]] | None = None
        self._anchor_vocab = taxonomy.all_anchor_concepts()

    # ------------------------------------------------------------------ profiles
    def _profile(self, concept: str, edges: list[Edge]) -> tuple[dict[str, float], dict[str, Edge], set[str]]:
        rw, neg_rels = self.cfg.relation_weights, set(self.cfg.negative_relations)
        profile: dict[str, float] = {concept: 1.0}
        best: dict[str, Edge] = {}
        negatives: set[str] = set()
        for e in edges:
            other = e.other(concept)
            if e.relation in neg_rels:
                negatives.add(other)
                continue
            if e.relation not in rw or e.weight < self.cfg.min_edge_weight or other in GENERIC_HUBS:
                continue
            s = rw[e.relation] * _squash(e.weight)
            if s > profile.get(other, 0.0):
                profile[other] = s
                best[other] = e
        return profile, best, negatives

    def _ensure_label_profiles(self) -> None:
        if self._label_profiles is not None:
            return
        self._label_profiles = {}
        for task in TASKS:
            for label in self.taxonomy[task]:
                merged: dict[str, float] = {}
                for anchor in label.anchor_concepts:
                    edges = self.client._raw(anchor)
                    prof, _, _ = self._profile(anchor, edges)
                    for k, v in prof.items():
                        if v > merged.get(k, 0.0):
                            merged[k] = v
                norm = math.sqrt(sum(v * v for v in merged.values()))
                self._label_profiles[(task, label.id)] = (merged, norm)

    # ------------------------------------------------------------------ step 1: key terms
    def extract_key_terms(self, visual: VisualConcepts, draft: DraftReasoning,
                          question: str | None = None) -> list[TermInfo]:
        terms: dict[str, TermInfo] = {}

        def add(phrase: str, kind: str, factor: float = 1.0) -> None:
            phrase = clean_phrase(phrase)
            if not phrase or len(phrase) < 3 or len(phrase.split()) > 3:
                return
            sal = SALIENCE[kind] * factor
            info = terms.get(phrase)
            if info is None:
                terms[phrase] = TermInfo(phrase, sal, {kind})
            else:
                info.salience = max(info.salience, sal)
                info.kinds.add(kind)

        def add_text(text: str, kind: str) -> None:
            if not text:
                return
            if len(clean_phrase(text).split()) <= 3:
                add(text, kind)
            parts = extract_terms(text)
            for v in parts["verbs"]:
                add(v, kind)
            for n in parts["nouns"]:
                add(n, kind, 0.8)
            for a in parts["adjectives"]:
                add(a, kind, 0.6)

        # Multi-word actions are kept whole: the client backs off to shorter nodes only when
        # the full phrase is unknown. Splitting off bare verbs ("shake") pulls in wrong senses
        # (shake -> milkshake -> drink).
        for a in visual.actions:
            add(a, "action")
        for g in visual.gestures:
            add(g, "gesture")
        for o in visual.objects:
            add(o, "object")
        for p in visual.people:
            words = clean_phrase(p).split()
            if not words:
                continue
            root = lemmatize(words[-1])
            if root in PERSON_NOUNS or words[-1] in PERSON_NOUNS:
                add(words[-1], "person")
                for w in words[:-1]:          # "young girl" -> age cue, "crying man" -> state cue
                    add(w, "person")
            else:
                add(p, "role")
                add(words[-1], "role", 0.9)
        if visual.setting:
            add(visual.setting, "setting")
            for n in extract_terms(visual.setting)["nouns"]:
                add(n, "setting")
        caption_terms = extract_terms(visual.caption)
        for v in caption_terms["verbs"]:
            add(v, "caption")
        for n in caption_terms["nouns"]:
            add(n, "caption", 0.8)
        add_text(draft.activity, "draft_activity")
        add_text(draft.relationship, "draft_relationship")
        add_text(draft.intention, "draft_intention")
        if question:
            add_text(question, "question")

        ranked = sorted(terms.values(), key=lambda t: -t.salience)
        return ranked[: self.cfg.max_terms]

    # ------------------------------------------------------------------ step 2: query + filter
    def query(self, info: TermInfo) -> TermInfo:
        concept, edges = self.client.resolve(info.phrase)
        info.concept, info.edges = concept, edges
        if edges and len(concept.split("_")) < len(info.phrase.split()):
            # The head word alone may carry a different sense: keep it, but trust it less.
            info.backed_off = True
            info.salience *= 0.5
        info.profile, info.best_edge, info.negatives = self._profile(concept, edges)
        info.phrase_index = {}
        for other, e in info.best_edge.items():
            words = other.split("_")
            if e.relation not in DEFINITIONAL:
                continue
            if 1 < len(words) <= 6:
                for w in words:
                    info.phrase_index.setdefault(w, []).append(other)
        return info

    def filter_relevant(self, info: TermInfo, evidence_concepts: set[str]) -> list[Fact]:
        if info.backed_off:
            return []                       # too sense-ambiguous to show as evidence
        rw = self.cfg.relation_weights
        scored: list[Fact] = []
        for other, e in info.best_edge.items():
            if other.count("_") > 2 or not other.replace("_", "").isalpha():
                continue
            relevance = rw[e.relation] * _squash(e.weight)
            if other in self._anchor_vocab:
                relevance *= 2.0            # links into the social-interaction label space
            elif other in evidence_concepts:
                relevance *= 1.5            # links two things seen in the image
            else:
                relevance *= 0.5
            scored.append(Fact(start=e.start, relation=e.relation, end=e.end, weight=e.weight,
                               surface=e.surface, source_term=info.concept,
                               relevance=round(relevance * info.salience, 4)))
        scored.sort(key=lambda f: -f.relevance)
        return scored[: self.cfg.max_facts_per_term]

    # ------------------------------------------------------------------ step 3: label support
    def _term_label_score(self, info: TermInfo, label: Label) -> tuple[float, str]:
        anchors = set(label.anchor_concepts)
        name = label.id
        if info.concept in anchors:
            return 1.0, f"'{info.phrase}' is a defining concept of {name}"

        best, reason = 0.0, ""
        for a in anchors:
            e = info.best_edge.get(a)
            if e is not None:
                s = 0.9 * info.profile[a]
                if s > best:
                    best, reason = s, f"ConceptNet: {fact_to_sentence(e)}"
            # edge into a phrase containing the anchor ("form_of_greeting_among_humans")
            for other in info.phrase_index.get(a, ()):
                s = 0.7 * info.profile[other]
                if s > best:
                    best, reason = s, f"ConceptNet: {fact_to_sentence(info.best_edge[other])}"

        prof, norm = self._label_profiles.get((label.task, label.id), ({}, 0.0))
        if info.profile and norm:
            tnorm = math.sqrt(sum(v * v for v in info.profile.values()))
            cos, shared = _cosine(info.profile, prof, tnorm, norm)
            s = self.cfg.two_hop_weight * min(1.0, 2.5 * cos)
            if s > best and shared:
                best = s
                reason = (f"ConceptNet: '{info.phrase}' and {name} share related concepts "
                          f"({', '.join(x.replace('_', ' ') for x in shared)})")

        hits = token_set(info.phrase) & label.lexicon
        if hits:
            df = self.taxonomy.df.get(label.task, {})
            kw = 0.6 / min(df.get(w, 1) for w in hits)
            if kw > best:
                best, reason = kw, f"'{info.phrase}' matches a keyword of {name}"

        if info.negatives & anchors:
            best -= 0.5
            reason = f"ConceptNet: '{info.phrase}' contrasts with {name}"
        return best, reason

    def label_support(self, terms: list[TermInfo]) -> tuple[dict, dict]:
        self._ensure_label_profiles()
        support: dict[str, dict[str, float]] = {}
        evidence: dict[str, dict[str, list[str]]] = {}
        for task in TASKS:
            kw = TASK_KIND_WEIGHTS[task]
            weights = [t.salience * max(kw.get(k, 0.0) for k in t.kinds) for t in terms]
            total = sum(weights) or 1.0
            support[task], evidence[task] = {}, {}
            for label in self.taxonomy[task]:
                contribs = []
                score = 0.0
                for t, w in zip(terms, weights):
                    if w <= 0:
                        continue
                    s, why = self._term_label_score(t, label)
                    score += w * s
                    if s > 0 and why:
                        contribs.append((w * s, why))
                contribs.sort(key=lambda c: -c[0])
                support[task][label.id] = max(0.0, score / total)
                evidence[task][label.id] = list(dict.fromkeys(w for _, w in contribs))[:3]
            # Scale so the best label has support 1.0 - but only when the evidence is strong.
            # The floor keeps weak, incidental links (e.g. "dinner" for relationships) weak.
            top = max(support[task].values(), default=0.0)
            scale = max(top, SUPPORT_FLOOR)
            support[task] = {k: v / scale for k, v in support[task].items()}
        return support, evidence

    def draft_label_match(self, draft: DraftReasoning, terms: list[TermInfo]) -> dict[str, dict[str, float]]:
        texts = {"activity": draft.activity, "relationship": draft.relationship, "intention": draft.intention}
        out: dict[str, dict[str, float]] = {}
        for task in TASKS:
            kind = DRAFT_KIND[task]
            draft_terms = [t for t in terms if kind in t.kinds]
            out[task] = {}
            for label in self.taxonomy[task]:
                lex = self.taxonomy.lexical_match(texts[task], label)
                cn = max((self._term_label_score(t, label)[0] for t in draft_terms), default=0.0)
                out[task][label.id] = round(max(lex, 0.8 * cn), 4)
        return out

    def resolve_ambiguity(self, support: dict, draft_match: dict) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for task in TASKS:
            combined = {lid: support[task].get(lid, 0.0) + draft_match[task].get(lid, 0.0)
                        for lid in support[task]}
            best = max(combined, key=combined.get, default=None)
            out[task] = best if best and combined[best] > 0 else None
        return out

    # ------------------------------------------------------------------ entry point
    def enrich(self, visual: VisualConcepts, draft: DraftReasoning,
               question: str | None = None) -> CommonsenseResult:
        terms = self.extract_key_terms(visual, draft, question)
        for t in terms:
            self.query(t)
        evidence_concepts = {t.concept for t in terms}

        facts: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()
        for t in terms:
            for f in self.filter_relevant(t, evidence_concepts):
                key = (f.start, f.relation, f.end)
                if key not in seen:
                    seen.add(key)
                    facts.append(f)
        facts.sort(key=lambda f: -f.relevance)
        facts = facts[: self.cfg.max_facts_total]

        support, evidence = self.label_support(terms)
        draft_match = self.draft_label_match(draft, terms)
        disambiguated = self.resolve_ambiguity(support, draft_match)
        found = sum(1 for t in terms if t.edges)
        log.info("ConceptNet (%s): %d/%d terms grounded, %d facts kept.",
                 self.client.backend.name, found, len(terms), len(facts))
        return CommonsenseResult(
            available=self.client.available,
            backend=self.client.backend.name,
            terms={t.concept or to_concept(t.phrase): round(t.salience, 3) for t in terms},
            facts=facts,
            support={k: {l: round(v, 4) for l, v in d.items()} for k, d in support.items()},
            draft_match=draft_match,
            evidence=evidence,
            disambiguated=disambiguated,
        )
