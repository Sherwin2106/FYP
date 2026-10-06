# Fine-tuning SmolVLM on VCR with LoRA

The Phase I pipeline uses SmolVLM zero-shot. This folder fine-tunes it on the VCR dataset
(`Rowan/vcr` on Hugging Face) with **LoRA**, on a free Kaggle GPU, and plugs the result back into
the pipeline with one config switch.

## Method

| | |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-Instruct` (2.2B) |
| Technique | LoRA, r=16, alpha=32, dropout=0.05, on the language model's `q/k/v/o_proj` and `gate/up/down_proj`. The SigLIP vision encoder and the connector stay frozen: perception is already good, the reasoning is what needs to change. |
| Data | VCR training images streamed from the dataset's Parquet shards (default 2,600 images, about 7,000 questions). Referenced people and objects are drawn as labelled boxes (`person1`, `person2`, …) and the text uses the same tags. |
| Tasks | `qa`: multiple-choice Q→A (target is the option letter). `qar`: multiple-choice QA→R (why is the answer right). `explain`: free-form "answer + visual reason", with the gold answer and rationale as the target. The multiple-choice prompts are word-for-word the prompts Module 4 scores at inference, so training directly sharpens the probabilities the pipeline fuses. Option order is shuffled so no letter is favoured. |
| Loss | Only on the assistant's answer tokens. The prompt and image tokens are masked. |
| Training | 1 epoch, batch 2 × gradient accumulation 8, lr 1e-4 with cosine decay, fp16 on T4/P100 (bf16 on newer GPUs), gradient checkpointing, a checkpoint every 100 steps. |
| Evaluation | Q→A / QA→R / Q→AR accuracy on 300 held-out VCR validation questions, for the base model (adapter disabled) and the fine-tuned model, plus sample explanations from both. |

## Files

| file | purpose |
|---|---|
| `train_vcr_lora.py` | the whole run: data → LoRA training → evaluation → artifacts |
| `build_kaggle_notebook.py` | builds `kaggle/smolvlm-vcr-lora.ipynb` with the script embedded |
| `kaggle/smolvlm-vcr-lora.ipynb` | the notebook to run on Kaggle |
| `kaggle_launch.py` | optional: push the notebook with the Kaggle API, wait, download, install |
| `install_adapter.py` | install a finished run into `models/` and `runs/` |

## Run it on Kaggle (free GPU)

**Option A: upload the notebook (no setup)**
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → choose
   `finetune/kaggle/smolvlm-vcr-lora.ipynb`.
2. Right-hand panel: **Accelerator → GPU T4 x2** (or P100) and **Internet → On**. Internet needs a
   phone-verified account.
3. **Save Version → Save & Run All (Commit)**. It runs in the background, about 3–4 h on a T4,
   and you can close the tab.
4. When it finishes, go to the version's **Output** tab and download `smolvlm-vcr-lora.zip`.
5. `python finetune/install_adapter.py ~/Downloads/smolvlm-vcr-lora.zip`

**Option B: from this machine with the Kaggle API**
```bash
pip install kaggle
# Kaggle → Settings → API → Create New Token → save kaggle.json to ~/.kaggle/
python finetune/kaggle_launch.py          # push, wait, download, install — all automatic
```

## Use the fine-tuned model

```bash
python -m svcr run --image photo.jpg --config configs/finetuned.yaml
SVCR_CONFIG=configs/finetuned.yaml uvicorn backend.main:app --port 8000   # web UI
```

The adapter is merged into the base weights when it loads, so inference is as fast as the base
model. The web UI header shows `+ LoRA (VCR)` when it's active.

## Compare base vs fine-tuned on more data

The run's `eval_results.json` / `README.md` already contain base vs fine-tuned numbers. For a
larger evaluation with the full pipeline's scoring, which averages two option orderings, run:
```bash
python scripts/eval_vcr.py --vcr-root /path/to/vcr1 --limit 1000
python scripts/eval_vcr.py --vcr-root /path/to/vcr1 --limit 1000 --config configs/finetuned.yaml
```

## Multi-epoch training with best-epoch selection

```bash
python finetune/kaggle_launch.py --slug smolvlm-vcr-lora-20ep \
    --train-args "--epochs 20 --train-images 600 --max-train-examples 1200 --time-limit-hours 10.5"
```
After every epoch the adapter is saved (`epochs/epoch_NN/`) and scored on 100 validation
questions kept apart from the 300 test questions (`epoch_metrics.csv`, `epoch_curve.png`).
At the end the epoch with the best selection-set Q→AR is loaded and becomes `adapter/`. The
final before/after numbers are measured on the separate test set, so they aren't biased by
the selection.

## Pause and resume

* `--stop-at 12:15` (IST): the run saves a checkpoint and its finished epochs at that time,
  writes a best-so-far result, and stops.
* `python finetune/kaggle_launch.py --resume` continues later. It starts a continuation notebook
  with the paused run's output attached, reuses the cached data, baseline scores, finished
  epochs and training settings, and carries on from the exact step it stopped at.
  Add `--stop-at HH:MM` again to pause the continuation too.
* Checkpoints are also written every 50 steps, so a crash loses at most a few minutes of training.

The run lives on Kaggle's servers: your laptop or Wi-Fi going off doesn't affect it.

## Tuning knobs

* **Less time:** `--train-images 1300 --max-train-examples 3500` (about 1.5 h).
* **More data:** `--train-images 5200 --max-train-examples 15000` (about 6–7 h; stays under Kaggle's 12 h limit).
* **Loss goes NaN on T4/P100** (fp16 overflow): add `--fp32-base`.
* **Out of GPU memory:** `--batch-size 1 --grad-accum 16`, or `--qlora` (4-bit base weights).
* **Quick check that everything works** (a few minutes, on a laptop): `python finetune/train_vcr_lora.py --smoke-test`.
  This proves the pipeline only; it is not a trained model.

VCR is licensed for non-commercial research use. Check the dataset card's terms before use.
