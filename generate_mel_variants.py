#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure numba cache has a writable home before importing librosa.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

import librosa  # noqa: E402  # isort:skip

INPUT_PATH = Path("playlist.mp3")
OUT_DIR = Path("output")

SR_TARGET = 44_100
N_FFT = 2048
HOP_LENGTH = 441
WIN_LENGTH = 2048
FMAX = 20_000.0


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_hr_mel_basis(sr: int, n_fft: int) -> Tuple[np.ndarray, List[slice]]:
    """Create custom mel basis for HR-Mel."""
    bands = [
        (0.0, 1_500.0, 40),
        (1_500.0, 6_000.0, 32),
        (6_000.0, 20_000.0, 24),
    ]
    bases = []
    slices = []
    start = 0
    for fmin, fmax, bins in bands:
        basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=bins, fmin=fmin, fmax=fmax, norm="slaney"
        )
        bases.append(basis)
        end = start + bins
        slices.append(slice(start, end))
        start = end
    return np.vstack(bases), slices


def hr_mel(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
    power_spec = np.abs(
        librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    ) ** 2
    basis, bands = build_hr_mel_basis(sr, N_FFT)
    mel = basis @ power_spec

    encoded = mel.copy()
    # Low and mid bands: log(1 + mel)
    encoded[bands[0]] = np.log1p(encoded[bands[0]])
    encoded[bands[1]] = np.log1p(encoded[bands[1]])
    # High band: sqrt(log(1 + mel))
    encoded[bands[2]] = np.sqrt(np.log1p(encoded[bands[2]]))

    meta = {
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "n_mels": int(encoded.shape[0]),
        "bands": [
            {"range_hz": [0, 1500], "bins": 40, "compression": "log1p"},
            {"range_hz": [1500, 6000], "bins": 32, "compression": "log1p"},
            {"range_hz": [6000, 20000], "bins": 24, "compression": "sqrt(log1p)"},
        ],
        "sr": sr,
        "fmax": FMAX,
    }
    return encoded, meta


def main() -> None:
    ensure_out_dir()
    y, sr = librosa.load(INPUT_PATH, sr=SR_TARGET, mono=True)

    # High-Resolution Log-Mel
    hr_encoded, hr_meta = hr_mel(y, sr)
    np.savez_compressed(
        OUT_DIR / "hr_mel.npz", encoded=hr_encoded, meta=json.dumps(hr_meta)
    )
    summary = {
        "input_sr": sr,
        "duration_sec": round(len(y) / sr, 2),
        "hr_mel_encoded": list(hr_encoded.shape),
        "bands": hr_meta["bands"],
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "fmax": FMAX,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
