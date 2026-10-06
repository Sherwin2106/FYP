#!/usr/bin/env python
"""Install a finished training run into the project.

Takes the run folder or the ``smolvlm-vcr-lora.zip`` downloaded from Kaggle's Output tab:
  * copies ``adapter/`` to ``models/smolvlm-vcr-lora/`` (what configs/finetuned.yaml loads)
  * keeps the full run (logs, loss curve, eval results, model card) in ``finetune/runs/``
  * prints the before/after validation accuracy

    python finetune/install_adapter.py ~/Downloads/smolvlm-vcr-lora.zip
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "smolvlm-vcr-lora"
RUNS = ROOT / "finetune" / "runs"


def find_run(path: Path) -> Path:
    """Locate the folder that contains adapter/adapter_config.json."""
    for cand in [path, *sorted(path.rglob("adapter_config.json"))]:
        folder = cand if cand.is_dir() else cand.parent.parent
        if (folder / "adapter" / "adapter_config.json").exists():
            return folder
    raise SystemExit(f"No adapter/adapter_config.json found under {path}")


def install(source: Path) -> Path:
    source = source.expanduser().resolve()
    tmp = None
    if source.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp)
        source = tmp
    run = find_run(source)

    run_config = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    if run_config.get("smoke_test"):
        print("WARNING: this is a smoke-test run (a few steps), not a real fine-tuning run.")

    dest_run = RUNS / f"{run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, dest_run, ignore=shutil.ignore_patterns("checkpoints"))
    if MODELS.exists():
        shutil.rmtree(MODELS)
    MODELS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run / "adapter", MODELS)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"Adapter installed -> {MODELS.relative_to(ROOT)}")
    print(f"Run artifacts     -> {dest_run.relative_to(ROOT)}")
    res = run_config.get("results", {})
    for name in ("base", "finetuned"):
        if name in res:
            r = res[name]
            print(f"  {name:9s} Q->A {r['Q->A']:.1%}   QA->R {r['QA->R']:.1%}   Q->AR {r['Q->AR']:.1%}")
    print("\nUse it:\n  python -m svcr run --image photo.jpg --config configs/finetuned.yaml\n"
          "  SVCR_CONFIG=configs/finetuned.yaml uvicorn backend.main:app --port 8000")
    return dest_run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    install(Path(sys.argv[1]))
