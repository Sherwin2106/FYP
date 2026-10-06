"""ConceptNet access layer.

Backends
--------
* ``LocalConceptNet``  - SQLite database built from the official ConceptNet 5.7
                         assertions dump (``scripts/build_conceptnet_db.py``).
                         Fast, offline and reproducible - the recommended backend.
* ``ApiConceptNet``    - the public REST API (api.conceptnet.io) with a persistent
                         on-disk cache. The public API is frequently unavailable,
                         so it is only used as a fallback.
* ``OfflineConceptNet``- returns no edges; the pipeline then falls back to
                         VLM-only reasoning.

``ConceptNetClient`` adds an in-memory cache and term back-off
("red wine glass" -> "wine_glass" -> "glass").
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import ConceptNetConfig, PROJECT_ROOT
from ..text import clean_phrase, lemmatize, to_concept

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Edge:
    relation: str
    start: str      # normalised concept, e.g. "shake_hands"
    end: str
    weight: float
    surface: str = ""

    def other(self, term: str) -> str:
        return self.end if self.start == term else self.start


def uri_to_term(uri: str) -> str | None:
    """'/c/en/shake_hands/v/wn/...' -> 'shake_hands' (None if not English)."""
    parts = uri.split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return parts[3]


class ConceptNetBackend:
    name = "base"

    def edges(self, term: str, limit: int) -> list[Edge]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OfflineConceptNet(ConceptNetBackend):
    name = "offline"

    def edges(self, term: str, limit: int) -> list[Edge]:
        return []


class LocalConceptNet(ConceptNetBackend):
    name = "local"

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"ConceptNet database not found at {self.path}. Build it with: "
                "python scripts/build_conceptnet_db.py")
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()

    def edges(self, term: str, limit: int) -> list[Edge]:
        sql = ("SELECT rel, start, end, weight, surface FROM edges WHERE start = ? "
               "UNION ALL SELECT rel, start, end, weight, surface FROM edges WHERE end = ? "
               "ORDER BY weight DESC LIMIT ?")
        with self._lock:
            rows = self._conn.execute(sql, (term, term, limit)).fetchall()
        return [Edge(r[0], r[1], r[2], float(r[3]), r[4] or "") for r in rows]

    def num_edges(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class _DiskCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (term TEXT PRIMARY KEY, data TEXT, ts REAL)")
        self._lock = threading.Lock()

    def get(self, term: str) -> list[Edge] | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM cache WHERE term = ?", (term,)).fetchone()
        return [Edge(*e) for e in json.loads(row[0])] if row else None

    def put(self, term: str, edges: list[Edge]) -> None:
        data = json.dumps([[e.relation, e.start, e.end, e.weight, e.surface] for e in edges])
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (term, data, time.time()))
            self._conn.commit()


class ApiConceptNet(ConceptNetBackend):
    name = "api"

    def __init__(self, base_url: str, cache_path: str | Path, timeout: float = 10.0):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache = _DiskCache(Path(cache_path))
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = "svcr-explainable-social-vcr/1.0"
        self._failures = 0

    def ping(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/c/en/greeting", params={"limit": 1}, timeout=self.timeout)
            return r.status_code == 200 and "edges" in r.json()
        except Exception:
            return False

    def edges(self, term: str, limit: int) -> list[Edge]:
        cached = self.cache.get(term)
        if cached is not None:
            return cached[:limit]
        if self._failures >= 3:          # API is down - stop hammering it for this session
            return []
        out: list[Edge] = []
        url = f"{self.base_url}/c/en/{term}"
        params = {"limit": min(1000, limit)}
        try:
            while url and len(out) < limit:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                for e in data.get("edges", []):
                    s, t = uri_to_term(e["start"]["@id"]), uri_to_term(e["end"]["@id"])
                    if s and t:
                        out.append(Edge(e["rel"]["label"], s, t, float(e.get("weight", 1.0)),
                                        e.get("surfaceText") or ""))
                nxt = (data.get("view") or {}).get("nextPage")
                url, params = (f"{self.base_url}{nxt}" if nxt else None), None
            self._failures = 0
        except Exception as exc:  # network / API errors are not fatal
            self._failures += 1
            log.warning("ConceptNet API request failed for %r: %s", term, exc)
            return []
        out.sort(key=lambda e: -e.weight)
        self.cache.put(term, out)
        return out[:limit]


def build_backend(cfg: ConceptNetConfig) -> ConceptNetBackend:
    local = PROJECT_ROOT / cfg.local_db if not Path(cfg.local_db).is_absolute() else Path(cfg.local_db)
    cache = PROJECT_ROOT / cfg.api_cache_db if not Path(cfg.api_cache_db).is_absolute() else Path(cfg.api_cache_db)
    mode = cfg.backend
    if mode == "offline":
        return OfflineConceptNet()
    if mode == "local" or (mode == "auto" and local.exists()):
        return LocalConceptNet(local)
    if mode in ("api", "auto"):
        api = ApiConceptNet(cfg.api_url, cache, cfg.timeout_s)
        if mode == "api" or api.ping():
            return api
        log.warning("ConceptNet: no local database (%s) and the public API is unreachable. "
                    "Commonsense enrichment is DISABLED. Build the local DB with "
                    "`python scripts/build_conceptnet_db.py`.", local)
        return OfflineConceptNet()
    raise ValueError(f"Unknown ConceptNet backend: {mode}")


class ConceptNetClient:
    """Cached ConceptNet lookups with term normalisation and back-off."""

    def __init__(self, backend: ConceptNetBackend, cfg: ConceptNetConfig | None = None):
        self.backend = backend
        self.cfg = cfg or ConceptNetConfig()
        self._mem: dict[str, list[Edge]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: ConceptNetConfig) -> "ConceptNetClient":
        return cls(build_backend(cfg), cfg)

    @property
    def available(self) -> bool:
        return not isinstance(self.backend, OfflineConceptNet)

    def _raw(self, concept: str) -> list[Edge]:
        with self._lock:
            if concept in self._mem:
                return self._mem[concept]
        edges = self.backend.edges(concept, self.cfg.max_edges_per_term) if concept else []
        edges = [e for e in edges if e.start != e.end]
        with self._lock:
            self._mem[concept] = edges
        return edges

    def resolve(self, phrase: str) -> tuple[str, list[Edge]]:
        """Find the best ConceptNet node for a phrase; returns (concept, edges)."""
        phrase = clean_phrase(phrase)
        candidates = [to_concept(phrase)]
        lem = to_concept(lemmatize(phrase))
        candidates.append(lem)
        words, raw_words = lem.split("_"), phrase.split()
        if len(raw_words) > 1:
            # "shaking hands" -> "shake_hands": lemmatise all but the last word
            candidates.append(to_concept(lemmatize(" ".join(raw_words[:-1])) + " " + raw_words[-1]))
        if len(words) > 1:
            candidates += ["_".join(words[-2:]), words[-1]]
        candidates = list(dict.fromkeys(c for c in candidates if c))
        for c in candidates:
            edges = self._raw(c)
            if edges:
                return c, edges
        return candidates[0], []

    def edges(self, phrase: str) -> list[Edge]:
        return self.resolve(phrase)[1]
