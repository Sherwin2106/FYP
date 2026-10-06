#!/usr/bin/env python
"""Generate finetune/kaggle/smolvlm-vcr-lora.ipynb from finetune/train_vcr_lora.py.

The notebook embeds the training script verbatim (via %%writefile), so it is
self-contained: upload this single .ipynb to Kaggle, switch on a GPU + internet and
"Save & Run All". Re-run this builder after editing the training script so the two
never drift apart.

    python finetune/build_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "train_vcr_lora.py"
OUT_DIR = HERE / "kaggle"
NOTEBOOK = OUT_DIR / "smolvlm-vcr-lora.ipynb"
OUTPUT = "/kaggle/working/smolvlm-vcr-lora"

INTRO = """# SmolVLM × VCR — LoRA fine-tuning (Phase II, Module 2)

Fine-tunes **HuggingFaceTB/SmolVLM-Instruct** with **LoRA** on the **VCR** dataset
(`Rowan/vcr` on Hugging Face) for the *Explainable Visual Commonsense Reasoning for Social
Interactions* project.

**Before running — notebook settings (right-hand panel):**
1. **Accelerator → GPU T4 x2** (or P100). Only one GPU is used.
2. **Internet → On** (needs a phone-verified Kaggle account) — the dataset is streamed from Hugging Face.
3. Click **Save Version → Save & Run All (Commit)** so it runs in the background (~3–4 h);
   you can close the browser.

**What it produces** in `/kaggle/working/smolvlm-vcr-lora/` (also zipped as `smolvlm-vcr-lora.zip`):
* `adapter/` — `adapter_config.json`, `adapter_model.safetensors` (the trained LoRA weights)
* `trainer_state.json`, `training_log.csv`, `loss_curve.png` — the real training history
* `eval_results.json` — Q→A / QA→R accuracy of the base vs fine-tuned model on held-out VCR
  validation questions, plus sample explanations from both
* `README.md` — model card with all of the above filled in from this run

Then download the zip from the **Output** tab and follow `finetune/README.md` in the project.
"""

RESULTS_CELL = f"""import json
from pathlib import Path
from IPython.display import Image, Markdown, display

out = Path("{OUTPUT}")
display(Markdown((out / "README.md").read_text()))
if (out / "loss_curve.png").exists():
    display(Image(str(out / "loss_curve.png")))
res = json.loads((out / "eval_results.json").read_text())
for s in res.get("samples", []):
    print("Q:        ", s["question"])
    print("gold:     ", s["gold"])
    print("base:     ", s["base"])
    print("finetuned:", s["finetuned"])
    print("-" * 100)
"""


PREFLIGHT = """# Fail fast if Kaggle didn't attach a GPU or internet. Kaggle silently withholds both on
# accounts that are not phone-verified, which would otherwise surface 20 minutes later.
import shutil, socket, subprocess
problems = []
if not shutil.which("nvidia-smi"):
    problems.append("no GPU attached (Settings -> Accelerator -> GPU T4 x2; the account must be phone-verified)")
else:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
for host in ("huggingface.co", "pypi.org"):
    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        problems.append(f"no internet - cannot resolve {host} (Settings -> Internet -> On; needs phone verification)")
        break
if problems:
    raise RuntimeError("Preflight failed: " + "; ".join(problems))
print("Preflight OK: GPU and internet available.")
"""


TRAIN_CELL = """# Streams the log live (visible in Kaggle's Logs tab) and stops the notebook if training fails.
import subprocess, sys
proc = subprocess.Popen([sys.executable, "-u", "train_vcr_lora.py", "--output-dir", "{OUTPUT}", *{ARGS}],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    raise RuntimeError(f"Training failed (exit code {proc.returncode}) - see the log above.")
"""


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build(train_args: list[str] | None = None) -> Path:
    script = SCRIPT.read_text()
    train_cell = TRAIN_CELL.replace("{OUTPUT}", OUTPUT).replace("{ARGS}", repr(list(train_args or [])))
    cells = [
        markdown(INTRO),
        code(PREFLIGHT),
        code("# Kaggle's image ships torchao 0.10, which recent peft refuses to run alongside (LoRA doesn't use it).\n"
             "!pip uninstall -y -q torchao\n"
             '!pip install -q -U "transformers>=4.50" "peft>=0.13" "accelerate>=1.0" '
             '"huggingface_hub>=0.26" pyarrow bitsandbytes'),
        code("%%writefile train_vcr_lora.py\n" + script),
        markdown("## Train + evaluate\nDefaults: 2,600 VCR training images → 7,500 examples "
                 "(Q→A, QA→R, answer+rationale), 1 epoch, LoRA r=16, 300 held-out validation questions. "
                 "Checkpoints every 50 steps; if this run is interrupted, `python finetune/kaggle_launch.py --resume` "
                 "continues from the last checkpoint. Stops gracefully after 9.5 h (Kaggle's limit is 12 h) "
                 "and still saves + evaluates."),
        code(train_cell),
        markdown("## Results"),
        code(RESULTS_CELL),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"          # nbformat 4.5 requires cell ids
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return NOTEBOOK


if __name__ == "__main__":
    print(f"Wrote {build()}")
