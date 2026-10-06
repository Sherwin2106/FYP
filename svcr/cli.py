"""Command-line interface.

  python -m svcr run   --image photo.jpg [--question "..."] [--choices "a" "b" ...]
  python -m svcr run   --video clip.mp4 [--max-frames 5]
  python -m svcr live  [--camera 0]
  python -m svcr conceptnet handshake [--label-support]
  python -m svcr info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

from .config import load_config

log = logging.getLogger("svcr")


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "PIL", "httpx", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_result(result) -> None:
    p = result.prediction
    bar = "=" * 72
    print(f"\n{bar}\n  EXPLAINABLE SOCIAL VCR - PHASE I RESULT\n{bar}")
    print(f"  Source          : {result.frame.source}")
    print(f"  People          : {p.people_count}")
    for t in (p.activity, p.relationship, p.intention):
        amb = "  [ambiguous]" if t.ambiguous else ""
        print(f"  {t.task.capitalize():<16}: {t.label}  ({t.confidence:.0%}){amb}")
        alts = ", ".join(f"{a.label} {a.probability:.0%}" for a in t.alternatives[1:3])
        if alts:
            print(f"  {'':<16}  alternatives: {alts}")
    print(f"\n  Draft (VLM)     : activity='{result.draft.activity}', relationship='{result.draft.relationship}'")
    print(f"  Explanation     : {p.explanation}")
    if p.evidence and result.commonsense.available:
        print(f"\n  Commonsense evidence (ConceptNet, {result.commonsense.backend}):")
        for e in p.evidence[:6]:
            print(f"    - {e.removeprefix('ConceptNet: ')}")
    else:
        print(f"\n  Commonsense     : none (backend: {result.commonsense.backend})")
    if p.question:
        print(f"\n  Q: {p.question}\n  A: {p.answer}")
        if p.answer_probabilities:
            for c, pr in zip(p.answer_choices, p.answer_probabilities):
                print(f"     {pr:6.1%}  {c}")
    print(f"\n  Timings (s)     : {result.timings_s}\n{bar}\n")


def _save(result, out_dir: Path, stem: str, save_image: bool = True) -> None:
    import cv2
    from .visualize import render_result
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(result.to_json(), encoding="utf-8")
    msg = f"Saved {json_path}"
    if save_image:
        img_path = out_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(img_path), render_result(result))
        msg += f" and {img_path}"
    print(msg)


def cmd_run(args, cfg) -> int:
    from .pipeline import Phase1Pipeline
    from .preprocessing import is_video_path

    pipe = Phase1Pipeline(cfg)
    out_dir = cfg.resolve_path(args.out or cfg.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.video or (args.image and is_video_path(args.image)):
        src = args.video or args.image
        for i, result in enumerate(pipe.stream(src, args.question, args.max_frames)):
            _print_result(result)
            _save(result, out_dir, f"{Path(src).stem}_f{result.frame.frame_index}_{stamp}", not args.no_image)
        return 0
    result = pipe.run(args.image, args.question, args.choices)
    _print_result(result)
    stem = Path(args.image).stem if not str(args.image).startswith("http") else "url_image"
    _save(result, out_dir, f"{stem}_{stamp}", not args.no_image)
    return 0


def cmd_live(args, cfg) -> int:
    """Live camera: capture/display on the main thread, analyse sampled frames in a worker."""
    import cv2
    from .pipeline import Phase1Pipeline
    from .preprocessing import assess_quality
    from .visualize import draw_overlay, render_result

    pipe = Phase1Pipeline(cfg)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    state = {"latest": None, "result": None, "busy": False, "stop": False}
    lock = threading.Lock()

    def worker():
        while not state["stop"]:
            with lock:
                job = state["latest"]
                state["latest"] = None
            if job is None:
                time.sleep(0.05)
                continue
            state["busy"] = True
            try:
                frame = pipe.preprocessor.process_bgr(job, f"camera:{args.camera}")
                res = pipe.run_frame(frame, args.question)
                with lock:
                    state["result"] = res
                _print_result(res)
            except Exception:
                log.exception("Analysis failed")
            finally:
                state["busy"] = False

    threading.Thread(target=worker, daemon=True).start()
    last_submit = 0.0
    print("Live mode: press 'q' to quit, 's' to save the latest result, SPACE to analyse now.")
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        force = key == ord(" ")
        if (force or now - last_submit >= cfg.preprocess.sample_interval_s) and not state["busy"]:
            if force or not (cfg.preprocess.skip_blurry_frames and assess_quality(bgr, cfg.preprocess)["is_blurry"]):
                with lock:
                    state["latest"] = bgr.copy()
                last_submit = now
        with lock:
            res = state["result"]
        cv2.imshow("Explainable Social VCR - Phase I", draw_overlay(bgr, res, "analysing..." if state["busy"] else ""))
        if key == ord("q"):
            break
        if key == ord("s") and res is not None:
            _save(res, cfg.resolve_path(cfg.output_dir), time.strftime("live_%Y%m%d_%H%M%S"))
    state["stop"] = True
    cap.release()
    cv2.destroyAllWindows()
    return 0


def cmd_conceptnet(args, cfg) -> int:
    from .conceptnet import CommonsenseEnricher, ConceptNetClient
    from .conceptnet.relations import fact_to_sentence
    from .schemas import DraftReasoning, VisualConcepts
    from .taxonomy import Taxonomy

    client = ConceptNetClient.from_config(cfg.conceptnet)
    print(f"Backend: {client.backend.name}")
    concept, edges = client.resolve(args.term)
    print(f"Resolved '{args.term}' -> '{concept}' ({len(edges)} edges)")
    for e in edges[: args.limit]:
        print(f"  [{e.weight:5.2f}] {fact_to_sentence(e)}")
    if args.label_support:
        tax = Taxonomy.load(cfg.resolve_path(cfg.reasoning.taxonomy_path))
        enricher = CommonsenseEnricher(client, tax, cfg.conceptnet)
        res = enricher.enrich(VisualConcepts(actions=[args.term], people_count=2), DraftReasoning())
        for task, sup in res.support.items():
            top = sorted(sup.items(), key=lambda kv: -kv[1])[:5]
            print(f"\n{task}: " + ", ".join(f"{k}={v:.2f}" for k, v in top))
            for why in res.evidence[task].get(top[0][0], []):
                print(f"    because {why}")
    return 0


def cmd_info(args, cfg) -> int:
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        from .vlm import resolve_device
        info.update(torch=torch.__version__, device=resolve_device(cfg.vlm.device))
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "NOT INSTALLED"
    from .text import get_nlp
    info["spacy_model"] = "ok" if get_nlp() is not None else "missing (fallback in use)"
    from .conceptnet import build_backend
    info["conceptnet_backend"] = build_backend(cfg.conceptnet).name
    info["vlm_model"] = cfg.vlm.model_id
    print(json.dumps(info, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    common.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. vlm.model_id=HuggingFaceTB/SmolVLM-500M-Instruct")
    parser = argparse.ArgumentParser(prog="svcr", description="Explainable Social VCR - Phase I pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, parents=[common])

    p = add("run", help="analyse an image or a video file")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image path or URL")
    src.add_argument("--video", help="video file path")
    p.add_argument("--question", help="optional context/query about the scene")
    p.add_argument("--choices", nargs="+", help="optional answer choices for --question (multiple choice)")
    p.add_argument("--max-frames", type=int, default=None, help="video: stop after N analysed frames")
    p.add_argument("--out", help="output directory (default: outputs/)")
    p.add_argument("--no-image", action="store_true", help="do not save the annotated image")

    p = add("live", help="analyse a live camera feed")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--question")

    p = add("conceptnet", help="inspect ConceptNet knowledge for a term")
    p.add_argument("term")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--label-support", action="store_true", help="show which labels this term supports")

    add("info", help="show environment / backend status")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    _setup_logging(cfg.log_level)
    handlers = {"run": cmd_run, "live": cmd_live, "conceptnet": cmd_conceptnet, "info": cmd_info}
    return handlers[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
