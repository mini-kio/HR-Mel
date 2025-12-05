#!/usr/bin/env python3
"""Extract HR-Mel features (configurable bands/compression) from an input file."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure numba cache has a writable home before importing librosa.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

import librosa  # noqa: E402  # isort:skip

from hr_mel import (
    DEFAULT_BANDS,
    DEFAULT_FMAX,
    DEFAULT_SR,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    hr_mel,
    save_hr_mel,
)


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
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    hr_encoded, hr_meta = hr_mel(y, sr, fmax=fmax, bands=DEFAULT_BANDS)
    save_hr_mel(hr_encoded, hr_meta, out_path=out_dir / "hr_mel.npz")
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
