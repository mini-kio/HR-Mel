#!/usr/bin/env python3
"""Extract HR-Mel (40/32/24 bins with sqrt-log in the top band) from an input file.

Defaults follow the 44.1 kHz spec in README, but you can override input/output/sr/fmax.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure numba cache has a writable home before importing librosa.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

import librosa  # noqa: E402  # isort:skip

# Default configuration (can be overridden via CLI).
DEFAULT_SR = 44_100
N_FFT = 2048
HOP_LENGTH = 441
WIN_LENGTH = 2048
DEFAULT_FMAX = 20_000.0


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def build_hr_mel_basis(sr: int, n_fft: int, fmax: float) -> Tuple[np.ndarray, List[slice]]:
    """Create custom mel basis for HR-Mel."""
    high_band_end = fmax
    bands = [
        (0.0, 1_500.0, 40),
        (1_500.0, 6_000.0, 32),
        (6_000.0, high_band_end, 24),
    ]
    bases = []
    slices = []
    start = 0
    for fmin, fmax_band, bins in bands:
        basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=bins, fmin=fmin, fmax=fmax_band, norm="slaney"
        )
        bases.append(basis)
        end = start + bins
        slices.append(slice(start, end))
        start = end
    return np.vstack(bases), slices


def hr_mel(y: np.ndarray, sr: int, fmax: float = DEFAULT_FMAX) -> Tuple[np.ndarray, Dict]:
    power_spec = np.abs(
        librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    ) ** 2
    basis, bands = build_hr_mel_basis(sr, N_FFT, fmax)
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
            {"range_hz": [6000, float(fmax)], "bins": 24, "compression": "sqrt(log1p)"},
        ],
        "sr": sr,
        "fmax": float(fmax),
    }
    return encoded, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HR-Mel features.")
    parser.add_argument("--input", type=Path, default=Path("playlist.mp3"), help="Input audio file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for outputs")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Hz)")
    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX,
        help="Upper frequency for Mel filters (Hz, clipped to Nyquist)",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    input_path: Path = args.input
    out_dir: Path = args.output_dir
    target_sr = args.sr

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fmax = min(args.fmax, target_sr / 2.0)
    ensure_out_dir(out_dir)
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    hr_encoded, hr_meta = hr_mel(y, sr, fmax=fmax)
    np.savez_compressed(
        out_dir / "hr_mel.npz", encoded=hr_encoded, meta=json.dumps(hr_meta)
    )
    summary = {
        "input_sr": sr,
        "duration_sec": round(len(y) / sr, 2),
        "hr_mel_encoded": list(hr_encoded.shape),
        "bands": hr_meta["bands"],
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "fmax": fmax,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
