"""Utilities for reproducible benchmark reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def write_json(path: str, payload: Dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
