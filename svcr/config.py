"""Configuration for the Phase I pipeline.

All settings live in dataclasses with sensible defaults. They can be overridden
from a YAML file (``configs/default.yaml``) and from the command line using
dotted ``section.key=value`` overrides.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class PreprocessConfig:
    # Longest side of the processed frame (pixels). SmolVLM tiles images into
    # 384px patches, so multiples of 384 avoid wasted resampling.
    target_longest_side: int = 1152
    # "fit": keep aspect ratio; "center_crop": square crop then resize; "none": keep size
    resize_mode: str = "fit"
    upscale_small: bool = False
    # SmolVLM (SigLIP) normalisation constants
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Quality handling
    enhance_low_light: bool = True
    low_light_threshold: float = 60.0      # mean luma below this => CLAHE enhancement
    denoise: bool = False
    blur_threshold: float = 60.0           # variance of Laplacian below this => blurry
    # Video / live camera sampling
    sample_interval_s: float = 2.0
    scene_change_threshold: float = 0.25   # histogram distance needed to re-analyse
    skip_blurry_frames: bool = True


@dataclass
class VLMConfig:
    # Other options: HuggingFaceTB/SmolVLM-500M-Instruct, HuggingFaceTB/SmolVLM-256M-Instruct,
    # HuggingFaceTB/SmolVLM2-2.2B-Instruct, or a local path to a fine-tuned checkpoint.
    model_id: str = "HuggingFaceTB/SmolVLM-Instruct"
    revision: str | None = None
    # Folder with a LoRA adapter from finetune/train_vcr_lora.py; merged into the base at load.
    adapter_path: str | None = None
    device: str = "auto"                   # auto | cuda | mps | cpu
    dtype: str = "auto"                    # auto | bfloat16 | float16 | float32
    local_files_only: bool = False
    # Longest edge the SmolVLM processor resizes to before tiling (None = model default).
    image_longest_edge: int | None = 1152
    max_new_tokens_description: int = 220
    max_new_tokens_short: int = 32
    max_new_tokens_explanation: int = 120
    repetition_penalty: float = 1.1
    # Number of option orderings averaged when scoring multiple choice (removes letter bias).
    debias_permutations: int = 2
    # Fast mode derives objects/actions from the caption instead of asking separate questions.
    fast_mode: bool = False
    seed: int = 0


@dataclass
class ConceptNetConfig:
    backend: str = "auto"                  # auto | local | api | offline
    local_db: str = "data/conceptnet/conceptnet_en.sqlite"
    api_url: str = "https://api.conceptnet.io"
    api_cache_db: str = "data/conceptnet/api_cache.sqlite"
    timeout_s: float = 10.0
    max_edges_per_term: int = 400
    min_edge_weight: float = 1.0
    max_terms: int = 20
    max_facts_per_term: int = 6
    max_facts_total: int = 40
    two_hop_weight: float = 0.6
    relation_weights: dict[str, float] = field(default_factory=lambda: {
        "UsedFor": 1.0, "HasSubevent": 1.0, "HasFirstSubevent": 0.9, "HasLastSubevent": 0.8,
        "MotivatedByGoal": 1.0, "CausesDesire": 0.9, "Causes": 0.9, "HasPrerequisite": 0.8,
        "CapableOf": 0.9, "IsA": 0.9, "MannerOf": 0.9, "Entails": 0.9, "Synonym": 1.0,
        "SimilarTo": 0.9, "Desires": 0.8, "AtLocation": 0.7, "LocatedNear": 0.6,
        "PartOf": 0.6, "HasA": 0.5, "HasProperty": 0.6, "ReceivesAction": 0.6,
        "CreatedBy": 0.5, "DefinedAs": 0.9, "RelatedTo": 0.6, "HasContext": 0.4,
        "DerivedFrom": 0.4,
    })
    # Relations that count as evidence *against* a link
    negative_relations: list[str] = field(default_factory=lambda: [
        "Antonym", "DistinctFrom", "NotDesires", "NotCapableOf", "NotUsedFor", "NotHasProperty",
    ])


@dataclass
class ReasoningConfig:
    taxonomy_path: str = "configs/taxonomy.yaml"
    shortlist_size: int = 8
    # Fusion weights (log-linear): final = w_vlm*log p_vlm + w_cn*log p_cn + w_draft*match
    w_vlm: float = 1.0
    w_commonsense: float = 0.5
    w_draft: float = 1.0
    commonsense_temperature: float = 0.2
    ambiguity_margin: float = 0.15
    max_context_facts: int = 10
    include_caption_in_context: bool = True
    condition_on_previous: bool = True     # relationship/intention prompts see the predicted activity
    generate_explanation: bool = True


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    conceptnet: ConceptNetConfig = field(default_factory=ConceptNetConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    output_dir: str = "outputs"
    log_level: str = "INFO"

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _update_dataclass(obj: Any, values: dict[str, Any], prefix: str = "") -> None:
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in (values or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value, prefix=f"{prefix}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)


def _parse_override(item: str) -> dict[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got: {item!r}")
    dotted, raw = item.split("=", 1)
    value = yaml.safe_load(raw) if raw.strip() else None
    out: dict[str, Any] = {}
    node = out
    parts = dotted.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load defaults, then the YAML file (if any), then dotted CLI overrides."""
    cfg = Config()
    yaml_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as fh:
            _update_dataclass(cfg, yaml.safe_load(fh) or {})
    elif path:
        raise FileNotFoundError(yaml_path)
    for item in overrides or []:
        _update_dataclass(cfg, _parse_override(item))
    return cfg
