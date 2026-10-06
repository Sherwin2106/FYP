#!/usr/bin/env python
"""Run the fine-tuning notebook on Kaggle from this machine, wait, and install the result.

Needs a Kaggle API token (Kaggle -> Settings -> API -> "Create New Token"):
  * ~/.kaggle/access_token (new-style token), or ~/.kaggle/kaggle.json, or
  * KAGGLE_USERNAME + KAGGLE_KEY environment variables.
Your Kaggle account must be phone-verified (needed for GPU + internet in notebooks).

    python finetune/kaggle_launch.py              # push, wait (~3-4 h), download, install
    python finetune/kaggle_launch.py --no-wait    # just push; check later with --status / --download
    python finetune/kaggle_launch.py --status     # one status check of the latest run
    python finetune/kaggle_launch.py --download   # fetch + install the latest finished run
    python finetune/kaggle_launch.py --resume     # continue a failed/stopped run from its last checkpoint

The training itself runs on Kaggle's servers: once pushed, it does not need this machine
or its network. Waiting here is only for convenience; network drops while waiting are
tolerated and polling simply continues.

--resume starts a continuation notebook ("<slug>-r1", "-r2", ...) with the previous run's
output attached as input; train_vcr_lora.py finds the last checkpoint, the cached data and
the baseline scores there and carries on from that step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE_DIR = HERE / "kaggle"
LAST_RUN = KAGGLE_DIR / "last_run.json"
SLUG = "smolvlm-vcr-lora"
TITLE = "SmolVLM VCR LoRA"
RESULT_FILES = r"(smolvlm-vcr-lora\.zip|\.log)$"     # skip checkpoints / data cache on download


def username(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        return json.loads(cred.read_text())["username"]
    # Access-token auth (~/.kaggle/access_token): ask the CLI who we are.
    out = subprocess.run([sys.executable, str(HERE / "kaggle_cli.py"), "config", "view"],
                         capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if m and m.group(1) != "None":
        return m.group(1)
    raise SystemExit("No Kaggle credentials: put access_token or kaggle.json in ~/.kaggle/.")


def kaggle(*args: str, capture: bool = False, retries: int = 3) -> str:
    """Run the Kaggle CLI via kaggle_cli.py (routes through www.kaggle.com), retrying network errors."""
    try:
        import kaggle as _  # noqa: F401
    except ImportError:
        raise SystemExit("The kaggle package is not installed: pip install kaggle")
    err = ""
    for attempt in range(retries):
        res = subprocess.run([sys.executable, str(HERE / "kaggle_cli.py"), *args],
                             capture_output=capture, text=True)
        if res.returncode == 0:
            return res.stdout if capture else ""
        err = res.stderr if capture else ""
        if attempt < retries - 1:
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{err}")


def last_ref(user: str) -> str:
    if LAST_RUN.exists():
        return json.loads(LAST_RUN.read_text())["ref"]
    return f"{user}/{SLUG}"


def push(user: str, slug: str, title: str, sources: list[str], machine_shape: str | None,
         train_args: list[str]) -> str:
    sys.path.insert(0, str(HERE))
    from build_kaggle_notebook import build
    build(train_args)
    meta = {
        "id": f"{user}/{slug}", "title": title, "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": sources, "model_sources": [],
    }
    if machine_shape:
        meta["machine_shape"] = machine_shape
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    ref = f"{user}/{slug}"
    print(f"Pushing {ref} (GPU + internet{', resuming from ' + sources[0] if sources else ''}) ...")
    kaggle("kernels", "push", "-p", str(KAGGLE_DIR))
    LAST_RUN.write_text(json.dumps({"ref": ref, "sources": sources, "train_args": train_args,
                                    "pushed": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"Running on Kaggle: https://www.kaggle.com/code/{ref}")
    return ref


def status(ref: str) -> str:
    try:
        out = kaggle("kernels", "status", ref, capture=True, retries=2).lower()
    except RuntimeError:
        return "unknown (network?)"
    m = re.search(r'status "?(?:kernelworkerstatus\.)?(\w+)', out)
    return m.group(1) if m else out.strip()


def download(ref: str) -> None:
    dest = HERE / "runs" / "kaggle-output"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    kaggle("kernels", "output", ref, "-p", str(dest), "--file-pattern", RESULT_FILES, "-o")
    zips = list(dest.rglob("smolvlm-vcr-lora.zip"))
    if not zips:
        logs = list(dest.rglob("*.log"))
        raise SystemExit(f"No results zip in the output of {ref} (did the run finish?). "
                         f"Log: {logs[0] if logs else 'none'}")
    sys.path.insert(0, str(HERE))
    from install_adapter import install
    install(zips[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--status", action="store_true", help="print the latest run's status and exit")
    ap.add_argument("--download", action="store_true", help="only download + install the latest finished run")
    ap.add_argument("--resume", action="store_true",
                    help="continue the latest run from its last checkpoint (new notebook, previous output attached)")
    ap.add_argument("--machine-shape", help="optional Kaggle machine shape, e.g. NvidiaTeslaT4")
    ap.add_argument("--poll-minutes", type=float, default=5)
    ap.add_argument("--username", help="Kaggle username (normally detected from the token)")
    ap.add_argument("--slug", default=SLUG, help="Kaggle notebook name for a new run")
    ap.add_argument("--train-args", default=None,
                    help='extra train_vcr_lora.py arguments, e.g. "--epochs 20 --max-train-examples 1200"; '
                         "--resume reuses the previous run's arguments unless given")
    ap.add_argument("--stop-at", metavar="HH:MM", help="pause the run at this IST time (resume later)")
    args = ap.parse_args()

    user = username(args.username)
    if args.status:
        ref = last_ref(user)
        print(f"{ref}: {status(ref)}")
        return 0
    if args.download:
        download(last_ref(user))
        return 0

    prev_args = json.loads(LAST_RUN.read_text()).get("train_args", []) if LAST_RUN.exists() else []
    train_args = args.train_args.split() if args.train_args is not None else (prev_args if args.resume else [])
    train_args = [a for i, a in enumerate(train_args)            # drop an old --stop-at HH:MM
                  if a != "--stop-at" and (i == 0 or train_args[i - 1] != "--stop-at")]
    if args.stop_at:
        train_args += ["--stop-at", args.stop_at]
    if args.resume:
        prev = last_ref(user)
        stem = re.sub(r"-r\d+$", "", prev.split("/", 1)[1])
        m = re.search(r"-r(\d+)$", prev)
        n = int(m.group(1)) + 1 if m else 1
        ref = push(user, f"{stem}-r{n}", f"{stem.replace('-', ' ')} r{n}", [prev], args.machine_shape, train_args)
    else:
        ref = push(user, args.slug, args.slug.replace("-", " "), [], args.machine_shape, train_args)
    if args.no_wait:
        print("Check later with: python finetune/kaggle_launch.py --status   (then --download)")
        return 0

    t0 = time.time()
    while True:
        time.sleep(args.poll_minutes * 60)
        st = status(ref)
        print(f"[{(time.time() - t0) / 60:5.0f} min] status: {st}", flush=True)
        if st in ("complete", "completed"):
            download(ref)
            return 0
        if st in ("error", "cancelacknowledged", "cancelrequested", "cancelled"):
            print(f"Run ended with status {st}. Continue it from its last checkpoint with:\n"
                  f"  python finetune/kaggle_launch.py --resume")
            return 1


if __name__ == "__main__":
    sys.exit(main())
