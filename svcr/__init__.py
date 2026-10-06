"""Explainable Visual Commonsense Reasoning for Social Interactions - Phase I.

Module 1  Input & Preprocessing       svcr.preprocessing
Module 2  Small Vision-Language Model svcr.vlm
Module 3  ConceptNet Enrichment       svcr.conceptnet
Module 4  Social Interaction Reasoning svcr.reasoning
"""

__version__ = "1.0.0"

from .config import Config, load_config  # noqa: E402

__all__ = ["Config", "load_config", "__version__"]
