# Explainable Visual Commonsense Reasoning for Social Interactions — Phase I

Implementation of **Phase I (Modules 1–4)** of the SSN final-year project
*"Explainable Visual Commonsense Reasoning for Social Interactions"*:

```
Image / video / live camera
        │
        ▼
Module 1  Input & Preprocessing (OpenCV)        svcr/preprocessing.py
        │  RGB frame + normalised 1×3×H×W tensor + quality report
        ▼
Module 2  Small VLM — SmolVLM                   svcr/vlm.py
        │  A. visual understanding → people, objects, actions, gestures, setting, caption
        │  B. language & reasoning → draft activity / relationship / intention / rationale
        ▼
Module 3  ConceptNet Enrichment                 svcr/conceptnet/
        │  key terms → ConceptNet facts → filtered facts → label support + disambiguation
        ▼
Module 4  Social Interaction Reasoning          svcr/reasoning.py
           fuse(visual, commonsense, question) → activity, relationship, intention
           (calibrated probabilities, alternatives, evidence) + textual explanation
```

By default SmolVLM runs zero-shot. [`finetune/`](finetune/README.md) fine-tunes it on VCR with LoRA
on a free Kaggle GPU; `--config configs/finetuned.yaml` then loads the trained adapter (section 4).

---

## 1. Setup

```bash
cd Visual_commonsense_reasoning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 1.1 ConceptNet (one-time, ~15–30 min)

The public ConceptNet API (`api.conceptnet.io`) is frequently **down**; it returned HTTP 502
while this project was being built. The pipeline therefore uses a **local SQLite copy of the
official ConceptNet 5.7 dump** (English edges only, ~3.4 M edges):

```bash
python scripts/build_conceptnet_db.py        # downloads ~475 MB, builds data/conceptnet/conceptnet_en.sqlite
```

Backend selection (`conceptnet.backend`):

| value     | behaviour |
|-----------|-----------|
| `auto`    | local DB if present → else public API (cached on disk) → else disabled with a warning |
| `local`   | local DB only |
| `api`     | public REST API with an on-disk cache (`data/conceptnet/api_cache.sqlite`) |
| `offline` | no ConceptNet. Module 3 still does keyword matching against the taxonomy |

### 1.2 SmolVLM weights

Weights download automatically on first run. To fetch them in advance:

```bash
python scripts/download_model.py                                      # default: SmolVLM-Instruct (2.2B)
python scripts/download_model.py HuggingFaceTB/SmolVLM-500M-Instruct  # lighter alternative
```

If `huggingface.co` is blocked on your network (e.g. campus Wi-Fi), use a mirror
(`export HF_ENDPOINT=https://hf-mirror.com`) or download on another network.

| model | params | RAM (bf16) | notes |
|-------|--------|-----------|-------|
| `HuggingFaceTB/SmolVLM-Instruct` (default) | 2.2 B | ~5 GB | best quality |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | 0.5 B | ~1.2 GB | ~3× faster, good for live camera |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | 0.26 B | ~0.6 GB | edge devices |

Check the environment with `python -m svcr info`.

---

## 2. Usage

```bash
# single image (path or URL); writes JSON + annotated image into outputs/
python -m svcr run --image path/to/image.jpg

# with an optional user question (free-form, or multiple choice)
python -m svcr run --image path/to/party.jpg --question "Why are they raising their glasses?"
python -m svcr run --image path/to/party.jpg --question "What is happening?" \
       --choices "They are celebrating." "They are arguing." "They are working."

# video file: analyses sampled frames (scene-change + blur aware)
python -m svcr run --video clip.mp4 --max-frames 5

# live camera (q = quit, SPACE = analyse now, s = save)
python -m svcr live --camera 0 --set vlm.model_id=HuggingFaceTB/SmolVLM-500M-Instruct

# inspect ConceptNet knowledge and which labels a concept supports
python -m svcr conceptnet handshake --label-support
```

Any config value can be overridden with `--set section.key=value` (see `configs/default.yaml`).

### Python API

```python
from svcr.pipeline import Phase1Pipeline

pipe = Phase1Pipeline.from_config()
result = pipe.run("path/to/image.jpg", question="Why are they shaking hands?")
print(result.prediction.summary)            # "Greeting | Colleagues or business associates | To greet ..."
print(result.prediction.activity.confidence)
print(result.to_json())                     # everything, including evidence and timings
```

### Output (abridged)

```json
{
  "prediction": {
    "activity":     {"label": "Greeting", "confidence": 0.81, "ambiguous": false,
                     "alternatives": [{"label": "Greeting", "probability": 0.81, "p_vlm": 0.64,
                                       "p_commonsense": 0.52, "draft_match": 1.0}, ...],
                     "evidence": ["VLM visual probability 0.64",
                                  "consistent with the VLM draft: \"Shaking hands\"",
                                  "ConceptNet: handshake is used for greeting"]},
    "relationship": {"label": "Colleagues or business associates", ...},
    "intention":    {"label": "To greet or acknowledge each other", ...},
    "explanation":  "The two men are clasping hands and smiling ...",
    "summary":      "Greeting | Colleagues or business associates | To greet or acknowledge each other"
  },
  "visual":      {"caption": "...", "people_count": 2, "actions": [...], "objects": [...], ...},
  "draft":       {"activity": "Shaking hands", "relationship": "...", "rationale": "..."},
  "commonsense": {"backend": "local", "facts": [...], "support": {...}, "disambiguated": {...}},
  "timings_s":   {"module1_preprocess": 0.02, "module2_vlm": 21.4, "module3_conceptnet": 0.6, ...}
}
```

---

## 3. Web UI

A React dashboard (`frontend/`) and a FastAPI backend (`backend/`) that wraps the same
`Phase1Pipeline` used by the CLI. The model loads once when the backend starts and stays
resident, so each analysis only pays for inference. Analysis runs in a background thread; the
page polls for progress and shows which of the four modules is currently running.

```bash
# Terminal 1 — backend (loads SmolVLM + ConceptNet once, then serves the API)
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000

# Terminal 2 — frontend (dev server with hot reload, proxies /api to :8000)
cd frontend
npm install
npm run dev          # → http://localhost:5173
```

For a single deployable server, build the frontend and let FastAPI serve it:

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000      # now also serves the built UI at http://localhost:8000
```

The UI covers: image upload or live camera capture, an optional free-form or multiple-choice
question, a live per-module progress stepper, the activity/relationship/intention predictions
with confidence bars and alternatives, the generated explanation, the ConceptNet evidence behind
each prediction, SmolVLM's raw visual concepts, per-module timings, a raw-JSON viewer, and a
session history strip. `backend/main.py` exposes:

| endpoint | purpose |
|---|---|
| `GET /api/health` | model/device/ConceptNet backend status |
| `POST /api/analyze` | upload an image (+ optional `question`, `choices`) → `{job_id}` |
| `GET /api/jobs/{job_id}` | poll status, current module, and the final result JSON |

## 4. Fine-tuning SmolVLM (LoRA on VCR)

Zero-shot SmolVLM sometimes invents unsupported reasons, e.g. "she wears glasses, so she may be hard
of hearing". `finetune/train_vcr_lora.py` trains LoRA adapters (r=16, on the language model's
attention and MLP projections, with the vision encoder frozen) on the VCR dataset (`Rowan/vcr`).
It trains on three tasks: multiple-choice Q→A, multiple-choice QA→R, and free-form
"answer + visual reason". Accuracy is measured on held-out VCR validation questions both before
and after training.

```bash
# on Kaggle: import finetune/kaggle/smolvlm-vcr-lora.ipynb, GPU + Internet on, Save & Run All
python finetune/install_adapter.py ~/Downloads/smolvlm-vcr-lora.zip     # after the run
python -m svcr run --image photo.jpg --config configs/finetuned.yaml     # use it
```

The method, every file it produces, and the tuning options are described in
[finetune/README.md](finetune/README.md).

## 5. How each module works

### Module 1 — Input & Preprocessing (`svcr/preprocessing.py`)
* Sources: image path / URL / numpy / PIL, video files, live camera (`FrameStream`).
* Quality report: brightness, contrast, blur (variance of Laplacian).
* Low-light or low-contrast frames get CLAHE on the LAB L-channel plus a gamma lift. Optional denoising.
* Aspect-preserving resize (longest side 1152 = 3×384 SmolVLM tiles) or centre crop.
* Output: RGB uint8 frame (consumed by the SmolVLM processor) plus a normalised
  `1×3×H×W` tensor (SigLIP mean/std 0.5), which Phase II needs for Grad-CAM and Insertion–Deletion.
* Video/live: frames are sampled every `sample_interval_s`. Blurry frames are skipped, and
  unchanged scenes are skipped (histogram distance) but still refreshed every 3 intervals.

### Module 2 — SmolVLM (`svcr/vlm.py`)
* **A. Visual understanding:** a detailed people-centric caption plus targeted short questions
  (people count, objects, actions, gestures/expressions, setting, people/roles), parsed with
  spaCy into clean concept lists. Spatial relations are extracted from the caption's dependency parse.
* **B. Language & reasoning:** free-form draft answers for activity, relationship and intention,
  plus a visual rationale (and an answer to the optional user question).
* Decoding is greedy with a repetition penalty, so the same input always gives the same output.
* `score_options()`: multiple-choice probabilities read directly from the next-token logits of
  the option letters (one forward pass, no sampling). They are averaged over several option
  orderings to cancel letter/position bias.

### Module 3 — ConceptNet enrichment (`svcr/conceptnet/`)
1. `extract_key_terms`: salience-weighted terms (actions > gestures > roles > objects > setting > caption).
2. `ConceptNet.query`: node lookup with back-off (`"shaking hands"` → `shake_hands` → `hand`).
3. `filter_relevant`: keep semantically useful English relations (UsedFor, HasSubevent,
   MotivatedByGoal, IsA, …), minimum edge weight, drop hub concepts, rank facts that link into
   the social-interaction label space.
4. **Label support:** for each label of `configs/taxonomy.yaml` (defined by ConceptNet anchor
   concepts), the evidence terms are scored by exact match, direct edge, 2-hop neighbourhood
   overlap (cosine of neighbour profiles), and keyword match with IDF weighting. Negative
   relations (Antonym, DistinctFrom, Not*) subtract. The result is a per-task support
   distribution plus readable evidence paths.
5. `resolve_ambiguity`: combines support with the match to the VLM draft → disambiguated labels.

### Module 4 — Social interaction reasoning (`svcr/reasoning.py`)
1. `fuse`: builds a context from the caption, visual concepts, top ConceptNet facts and the user question.
2. For **activity → relationship → intention**, in sequence (each later task sees the earlier prediction):
   * shortlist candidates from the commonsense prior and the draft match;
   * SmolVLM scores the shortlist as a multiple-choice question over *image + fused context*;
   * log-linear fusion: `log P = w_vlm·log p_vlm + w_cn·log p_cn + w_draft·match`.
3. Output per task: label, calibrated confidence, ranked alternatives (each with its VLM,
   commonsense and draft scores), an `ambiguous` flag (top-2 margin < 0.15), and evidence.
4. `generate_explanation`: SmolVLM writes a grounded textual explanation of the final prediction.
   This is the input to Phase II's textual explanation path (Module 5A).
5. Edge cases: 0 people (no interaction), 1 person (no relationship; solo activity).

The label space (26 activities, 15 relationships, 22 intentions) lives in
`configs/taxonomy.yaml`. You can add or edit labels without touching code.

---

## 6. Evaluation on VCR (optional)

Download VCR from <https://visualcommonsense.com> (registration required), then:

```bash
python scripts/eval_vcr.py --vcr-root /path/to/vcr1 --split val --limit 500
python scripts/eval_vcr.py --vcr-root /path/to/vcr1 --split val --limit 500 --commonsense
```

This reports zero-shot **Q→A**, **QA→R** and **Q→AR** accuracy. Referenced people/objects are drawn
on the image as labelled boxes (`person1`, `person2`, …), following the VCR convention. These
numbers are the baseline that Phase II fine-tuning should beat.

## 7. Tests

```bash
pytest -q
```

The tests cover all four modules without downloading the VLM: a deterministic fake VLM and a toy
ConceptNet graph are used, plus a synthetic video for stream sampling.

## 8. Project layout

```
configs/default.yaml        all settings
configs/taxonomy.yaml       activity / relationship / intention label space
svcr/preprocessing.py       Module 1
svcr/vlm.py                 Module 2
svcr/conceptnet/            Module 3 (client.py = backends, enrichment.py = logic)
svcr/reasoning.py           Module 4
svcr/pipeline.py            Phase I pipeline (M1→M4), video streaming
svcr/visualize.py           annotated result image / live overlay (CLI)
svcr/vcr_dataset.py         VCR loader (for evaluation and future fine-tuning)
svcr/cli.py                 command line
backend/main.py             FastAPI wrapper around Phase1Pipeline for the web UI
frontend/                   React dashboard (Vite) — see section 3
finetune/                   LoRA fine-tuning on VCR (Kaggle notebook, launcher, installer)
configs/finetuned.yaml      loads the trained LoRA adapter from models/smolvlm-vcr-lora
scripts/                    ConceptNet DB builder, model downloader, VCR evaluation
tests/                      unit + end-to-end tests
```

## 9. Hand-off to Phase II

Phase II modules can use these Phase I outputs directly:
* `result.frame.tensor`: normalised pixel tensor for Grad-CAM (5B) and Insertion–Deletion (7);
* `result.prediction.explanation`: textual explanation for concept extraction (5A);
* `result.prediction.activity.label_id`: target class for Grad-CAM;
* `result.commonsense.facts` and the `evidence` fields: commonsense links for the dashboard (8).
