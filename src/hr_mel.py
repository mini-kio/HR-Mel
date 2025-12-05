#!/usr/bin/env python3
"""HR-Mel utilities: basis construction, encoding/decoding, and extraction."""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import librosa

# Default configuration (44.1 kHz spec).
DEFAULT_SR = 44_100
N_FFT = 2048
HOP_LENGTH = 441
WIN_LENGTH = 2048
DEFAULT_FMAX = 20_000.0

# Default 40/32/24 with sqrt-log on the top band.
DEFAULT_BANDS: List[Dict] = [
    {"fmin": 0.0, "fmax": 1_500.0, "bins": 40, "compression": "log1p"},
    {"fmin": 1_500.0, "fmax": 6_000.0, "bins": 32, "compression": "log1p"},
    {"fmin": 6_000.0, "fmax": DEFAULT_FMAX, "bins": 24, "compression": "sqrt_log1p"},
]

VALID_COMPRESSIONS = {"log1p", "sqrt_log1p", "pow075"}


def build_hr_mel_basis(
    sr: int, n_fft: int, fmax: float, bands: Sequence[Dict] = DEFAULT_BANDS
) -> Tuple[np.ndarray, List[slice]]:
    """Create custom mel basis for HR-Mel."""
    bases: List[np.ndarray] = []
    slices: List[slice] = []
    start = 0
    for band in bands:
        bins = int(band["bins"])
        fmin = float(band["fmin"])
        fmax_band = min(float(band["fmax"]), fmax)
        basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=bins, fmin=fmin, fmax=fmax_band, norm="slaney"
        )
        bases.append(basis)
        end = start + bins
        slices.append(slice(start, end))
        start = end
    return np.vstack(bases), slices


def encode_band(values: np.ndarray, compression: str) -> np.ndarray:
    if compression == "log1p":
        return np.log1p(values)
    if compression == "sqrt_log1p":
        return np.sqrt(np.log1p(values))
    if compression == "pow075":
        return values ** 0.75
    raise ValueError(f"Unsupported compression: {compression}")


def decode_band(values: np.ndarray, compression: str) -> np.ndarray:
    if compression == "log1p":
        return np.expm1(values)
    if compression == "sqrt_log1p":
        return np.expm1(values**2)
    if compression == "pow075":
        return values ** (1 / 0.75)
    raise ValueError(f"Unsupported compression: {compression}")


def encode_hr(power: np.ndarray, slices: List[slice], bands: Sequence[Dict]) -> np.ndarray:
    encoded = power.copy()
    for band, sl in zip(bands, slices):
        encoded[sl] = encode_band(encoded[sl], band["compression"])
    return encoded


def decode_hr(encoded: np.ndarray, slices: List[slice], bands: Sequence[Dict]) -> np.ndarray:
    decoded = encoded.copy()
    for band, sl in zip(bands, slices):
        decoded[sl] = decode_band(decoded[sl], band["compression"])
    return decoded


def hr_mel(
    y: np.ndarray,
    sr: int,
    fmax: float = DEFAULT_FMAX,
    bands: Sequence[Dict] = DEFAULT_BANDS,
) -> Tuple[np.ndarray, Dict]:
    power_spec = np.abs(
        librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    ) ** 2
    basis, band_slices = build_hr_mel_basis(sr, N_FFT, fmax, bands=bands)
    mel = basis @ power_spec
    encoded = encode_hr(mel, band_slices, bands)
    meta = {
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "n_mels": int(encoded.shape[0]),
        "bands": [
            {"range_hz": [float(b["fmin"]), float(min(b["fmax"], fmax))], "bins": int(b["bins"]), "compression": b["compression"]}
            for b in bands
        ],
        "sr": sr,
        "fmax": float(fmax),
    }
    return encoded, meta


def save_hr_mel(encoded: np.ndarray, meta: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, encoded=encoded, meta=json.dumps(meta))

