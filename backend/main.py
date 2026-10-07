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

import base64
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
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
from svcr.video import select_key_moments, summarize_moments  # noqa: E402
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
MAX_VIDEO_BYTES = 300 * 1024 * 1024

_state: dict = {"pipeline": None, "cfg": None, "error": None}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()          # one model run at a time (see _run_frame)


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


def _run_frame(pipe: Phase1Pipeline, job: dict, bgr: np.ndarray, source: str, question, choices,
               frame_index=None, timestamp_s=None) -> dict:
    """Analyse one frame for a job, reporting per-module progress into ``job``.

    Runs under _gpu_lock: the model is not thread-safe, and on Apple GPUs (MPS) two
    concurrent runs crash the whole process inside the Metal driver. Later requests wait.
    """
    def on_progress(stage: str, kind: str) -> None:
        if kind == "start":
            job["stage"] = stage
        else:
            job["done"].append(stage)
            job["stage"] = None

    with _gpu_lock:
        job["status"] = "running"
        job["done"], job["stage"] = [], None
        on_progress("preprocess", "start")
        frame = pipe.preprocessor.process_bgr(bgr, source, frame_index, timestamp_s)
        on_progress("preprocess", "done")
        return pipe.run_frame(frame, question, choices, on_progress=on_progress).to_dict()


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
            job["result"] = _run_frame(pipe, job, bgr, image.filename or "upload", question, choice_list)
            job["status"] = "done"
        except Exception as exc:  # pragma: no cover - surfaced to the client instead
            log.exception("Analysis failed for job %s", job_id)
            job["status"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


def _thumbnail(bgr: np.ndarray, width: int = 320) -> str:
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.post("/api/analyze-video", status_code=202)
async def analyze_video(video: UploadFile = File(...), question: Optional[str] = Form(None),
                        max_moments: int = Form(7)):
    """Accepts a video, picks up to ``max_moments`` key moments and analyses each in turn."""
    pipe = _get_pipeline()
    max_moments = max(1, min(int(max_moments), 12))
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    try:
        while chunk := await video.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(413, f"The video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB.")
            tmp.write(chunk)
    except HTTPException:
        tmp.close()
        os.unlink(tmp.name)
        raise
    tmp.close()
    if size == 0:
        os.unlink(tmp.name)
        raise HTTPException(400, "Empty file upload.")
    question = question.strip() if question and question.strip() else None
    filename = video.filename or "video"

    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"kind": "video", "status": "running", "stage": "keyframes", "done": [],
                         "created": time.time(), "result": None, "error": None,
                         "scan_progress": 0.0, "video": None, "moments": [], "current": None}

    def worker() -> None:
        job = _jobs[job_id]
        try:
            moments, info = select_key_moments(
                tmp.name, max_moments=max_moments,
                progress=lambda f: job.__setitem__("scan_progress", round(f, 3)))
            job["scan_progress"], job["stage"] = 1.0, None
            job["video"] = asdict(info)
            job["moments"] = [{"index": m.index, "time_s": m.time_s, "segment": list(m.segment),
                               "sharpness": m.sharpness, "thumbnail": _thumbnail(m.bgr),
                               "status": "pending", "summary": None} for m in moments]
            results: list = []
            for m, entry in zip(moments, job["moments"]):
                job["current"] = m.index
                entry["status"] = "running"
                try:
                    res = _run_frame(pipe, job, m.bgr, f"{filename} @ {m.time_s:.1f}s", question, None,
                                     m.frame_index, m.time_s)
                    p = res["prediction"]
                    entry["summary"] = {t: p[t]["label"] for t in ("activity", "relationship", "intention")}
                    entry["status"] = "done"
                    results.append(res)
                except Exception as exc:                # one bad moment doesn't sink the whole video
                    log.exception("Moment %d of job %s failed", m.index, job_id)
                    entry["status"], entry["error"] = "error", str(exc)
                    results.append(None)
            ok = [r for r in results if r]
            if not ok:
                raise RuntimeError("None of the selected moments could be analysed.")
            job["result"] = {
                "kind": "video", "filename": filename, "video": job["video"],
                "summary": summarize_moments(ok),
                "moments": [{**{k: v for k, v in e.items() if k != "status"}, "result": r}
                            for e, r in zip(job["moments"], results)],
            }
            job["current"], job["stage"], job["status"] = None, None, "done"
        except Exception as exc:  # pragma: no cover - surfaced to the client instead
            log.exception("Video analysis failed for job %s", job_id)
            job["status"], job["error"] = "error", str(exc)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job id.")
    return {
        "kind": job.get("kind", "image"),
        "status": job["status"],
        "stage": job["stage"],
        "stage_label": STAGE_LABEL.get(job["stage"]),
        "done_stages": job["done"],
        "stage_order": STAGE_ORDER,
        "error": job["error"],
        "result": job["result"],
        # video jobs only: scan progress, clip info, per-moment status (without full results)
        "scan_progress": job.get("scan_progress"),
        "video": job.get("video"),
        "moments": job.get("moments"),
        "current": job.get("current"),
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
