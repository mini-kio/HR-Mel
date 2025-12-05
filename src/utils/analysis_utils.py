#!/usr/bin/env python3
"""Shared analysis utilities for HR-Mel experiments."""

import io
from pathlib import Path
from typing import Iterable

import numpy as np


def rel_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(target, "fro") + 1e-12
    return float(np.linalg.norm(target - approx, "fro") / denom)


def compressed_size_bytes(**arrays: np.ndarray) -> int:
    """Return size in bytes of a compressed npz containing the provided arrays."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return len(buf.getvalue())


def mean_std(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
