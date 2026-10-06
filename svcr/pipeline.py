"""Phase I pipeline: Module 1 -> Module 2 -> Module 3 -> Module 4."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, Sequence

ProgressFn = Callable[[str, str], None]   # (stage, "start" | "done") -> None

from .config import Config, load_config
from .conceptnet import CommonsenseEnricher, ConceptNetClient
from .preprocessing import FrameStream, Preprocessor
from .reasoning import SocialInteractionReasoner
from .schemas import Phase1Result, PreprocessedFrame
from .taxonomy import Taxonomy

log = logging.getLogger(__name__)


class Phase1Pipeline:
    """Core prediction pipeline (report Section 5.12.1).

    ``vlm`` and ``conceptnet_client`` can be injected (e.g. for tests); by default
    SmolVLM and the configured ConceptNet backend are loaded.
    """

    def __init__(self, cfg: Config | None = None, vlm=None, conceptnet_client: ConceptNetClient | None = None):
        self.cfg = cfg or load_config()
        self.preprocessor = Preprocessor(self.cfg.preprocess)                       # Module 1
        if vlm is None:
            from .vlm import SmolVLMEngine
            vlm = SmolVLMEngine(self.cfg.vlm)                                       # Module 2
        self.vlm = vlm
        self.taxonomy = Taxonomy.load(self.cfg.resolve_path(self.cfg.reasoning.taxonomy_path))
        client = conceptnet_client or ConceptNetClient.from_config(self.cfg.conceptnet)
        self.enricher = CommonsenseEnricher(client, self.taxonomy, self.cfg.conceptnet)   # Module 3
        self.reasoner = SocialInteractionReasoner(                                  # Module 4
            self.vlm, self.taxonomy, self.cfg.reasoning,
            max_new_tokens_explanation=self.cfg.vlm.max_new_tokens_explanation)

    @classmethod
    def from_config(cls, path: str | None = None, overrides: list[str] | None = None) -> "Phase1Pipeline":
        return cls(load_config(path, overrides))

    def run_frame(self, frame: PreprocessedFrame, question: str | None = None,
                  answer_choices: Sequence[str] | None = None,
                  on_progress: ProgressFn | None = None) -> Phase1Result:
        def mark(stage: str, status: str) -> None:
            if on_progress is not None:
                on_progress(stage, status)

        timings: dict[str, float] = {}

        mark("vlm", "start")
        t = time.perf_counter()
        visual, draft = self.vlm.run(frame, question)
        timings["module2_vlm"] = time.perf_counter() - t
        mark("vlm", "done")
        log.info("M2 draft: activity=%r relationship=%r (people=%d)", draft.activity, draft.relationship,
                 visual.people_count)

        mark("conceptnet", "start")
        t = time.perf_counter()
        commonsense = self.enricher.enrich(visual, draft, question)
        timings["module3_conceptnet"] = time.perf_counter() - t
        mark("conceptnet", "done")

        mark("reasoning", "start")
        t = time.perf_counter()
        prediction = self.reasoner.reason(frame, visual, draft, commonsense, question, answer_choices)
        timings["module4_reasoning"] = time.perf_counter() - t
        mark("reasoning", "done")
        log.info("M4 prediction: %s", prediction.summary)

        return Phase1Result(frame=frame, visual=visual, draft=draft, commonsense=commonsense,
                            prediction=prediction, timings_s={k: round(v, 3) for k, v in timings.items()},
                            model_id=getattr(self.vlm, "model_id", type(self.vlm).__name__))

    def run(self, source, question: str | None = None,
            answer_choices: Sequence[str] | None = None,
            on_progress: ProgressFn | None = None) -> Phase1Result:
        if on_progress is not None:
            on_progress("preprocess", "start")
        t = time.perf_counter()
        frame = self.preprocessor(source)
        pre = time.perf_counter() - t
        if on_progress is not None:
            on_progress("preprocess", "done")
        result = self.run_frame(frame, question, answer_choices, on_progress)
        result.timings_s = {"module1_preprocess": round(pre, 3), **result.timings_s,
                            "total": round(pre + sum(result.timings_s.values()), 3)}
        return result

    def stream(self, source: int | str, question: str | None = None,
               max_frames: int | None = None) -> Iterator[Phase1Result]:
        """Analyse a video file or camera, yielding one result per sampled frame."""
        with FrameStream(source, self.cfg.preprocess) as frames:
            for n, raw in enumerate(frames.sampled()):
                if max_frames is not None and n >= max_frames:
                    break
                frame = self.preprocessor.process_bgr(raw.bgr, str(source), raw.index, raw.timestamp_s)
                yield self.run_frame(frame, question)
