#!/usr/bin/env python
"""Pre-download SmolVLM weights so the pipeline can run offline.

    python scripts/download_model.py                                   # default model from config
    python scripts/download_model.py HuggingFaceTB/SmolVLM-500M-Instruct

If huggingface.co is blocked on your network, set a mirror first:
    export HF_ENDPOINT=https://hf-mirror.com
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402

from svcr.config import load_config  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else load_config().vlm.model_id
path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
print(f"{model_id} downloaded to {path}")
