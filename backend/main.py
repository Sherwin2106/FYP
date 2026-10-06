"""FastAPI backend for the Explainable Social VCR React UI.

Loads the Phase I pipeline (SmolVLM + ConceptNet + reasoning) once at startup
and keeps it resident in memory, so each request only pays for inference, not
model loading. Analysis runs in a background thread; the frontend polls
``GET /api/jobs/{id}`` for live per-module progress (mirrors the four report
modules) and the final result (the same JSON shape as ``Phase1Result.to_dict()``
/ the ``svcr run`` CLI).

    uvicorn backend.main:app --reload --port 8000
    # or:  python backend/main.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svcr.config import load_config  # noqa: E402
from svcr.pipeline import Phase1Pipeline  # noqa: E402
from svcr.vlm import resolve_device  # noqa: E402

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("svcr.api")

app = FastAPI(title="Explainable Social VCR API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STAGE_ORDER = ["preprocess", "vlm", "conceptnet", "reasoning"]
STAGE_LABEL = {
    "preprocess": "Module 1 · Input & preprocessing (OpenCV)",
    "vlm": "Module 2 · SmolVLM visual understanding & reasoning",
    "conceptnet": "Module 3 · ConceptNet commonsense enrichment",
    "reasoning": "Module 4 · Social interaction reasoning",
}
JOB_TTL_S = 3600

_state: dict = {"pipeline": None, "cfg": None, "error": None}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _env_overrides() -> list[str]:
    """SVCR_SET="vlm.model_id=...,conceptnet.backend=offline" - handy for local dev."""
    raw = os.environ.get("SVCR_SET", "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def _load_pipeline() -> None:
    try:
        # SVCR_CONFIG=configs/finetuned.yaml switches to the LoRA fine-tuned SmolVLM.
        cfg = load_config(os.environ.get("SVCR_CONFIG") or None, overrides=_env_overrides())
        log.info("Loading Phase I pipeline (SmolVLM + ConceptNet) - this happens once ...")
        _state["cfg"] = cfg
        _state["pipeline"] = Phase1Pipeline(cfg)
        log.info("Pipeline ready: model=%s device=%s conceptnet=%s", cfg.vlm.model_id,
                 resolve_device(cfg.vlm.device), _state["pipeline"].enricher.client.backend.name)
    except Exception as exc:  # pragma: no cover - startup failure path
        log.exception("Pipeline failed to load")
        _state["error"] = str(exc)


@app.on_event("startup")
def startup() -> None:
    # Load in a background thread so the server answers /api/health immediately
    # instead of hanging the whole process during model download/load.
    threading.Thread(target=_load_pipeline, daemon=True).start()


def _get_pipeline() -> Phase1Pipeline:
    if _state["error"]:
        raise HTTPException(500, f"Model failed to load: {_state['error']}")
    pipe = _state["pipeline"]
    if pipe is None:
        raise HTTPException(503, "Model is still loading - please retry in a few seconds.")
    return pipe


def _cleanup_jobs() -> None:
    cutoff = time.time() - JOB_TTL_S
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v["created"] < cutoff]:
            _jobs.pop(jid, None)


@app.get("/api/health")
def health():
    cfg = _state["cfg"]
    pipe = _state["pipeline"]
    return {
        "status": "error" if _state["error"] else ("ready" if pipe else "loading"),
        "error": _state["error"],
        "model_id": cfg.vlm.model_id if cfg else None,
        "device": resolve_device(cfg.vlm.device) if cfg else None,
        "conceptnet_backend": pipe.enricher.client.backend.name if pipe else None,
        "conceptnet_available": pipe.enricher.client.available if pipe else None,
        "adapter": getattr(pipe.vlm, "adapter", None) if pipe else None,
    }


@app.post("/api/analyze", status_code=202)
async def analyze(image: UploadFile = File(...), question: Optional[str] = Form(None),
                  choices: Optional[str] = Form(None)):
    """Accepts an image upload, starts analysis in the background, returns a job id."""
    pipe = _get_pipeline()
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty file upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image - please upload a JPG, PNG or WEBP file.")

    choice_list = [c.strip() for c in choices.split("|") if c.strip()] if choices else None
    question = question.strip() if question and question.strip() else None

    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "stage": None, "done": [], "created": time.time(),
                         "result": None, "error": None}

    def worker() -> None:
        job = _jobs[job_id]
        try:
            job["status"] = "running"

            def on_progress(stage: str, kind: str) -> None:
                if kind == "start":
                    job["stage"] = stage
                else:
                    job["done"].append(stage)
                    job["stage"] = None

            on_progress("preprocess", "start")
            frame = pipe.preprocessor.process_bgr(bgr, image.filename or "upload")
            on_progress("preprocess", "done")
            result = pipe.run_frame(frame, question, choice_list, on_progress=on_progress)
            job["result"] = result.to_dict()
            job["status"] = "done"
        except Exception as exc:  # pragma: no cover - surfaced to the client instead
            log.exception("Analysis failed for job %s", job_id)
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job id.")
    return {
        "status": job["status"],
        "stage": job["stage"],
        "stage_label": STAGE_LABEL.get(job["stage"]),
        "done_stages": job["done"],
        "stage_order": STAGE_ORDER,
        "error": job["error"],
        "result": job["result"],
    }


# -- serve the built React app (frontend/dist) in production, if present --------------
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Explainable Social VCR API is running. Frontend not built - run `npm run build` "
                           "in frontend/, or use `npm run dev` for local development.",
                "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
