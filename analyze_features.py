#!/usr/bin/env python3
"""Compare STFT, Mel, Log-Mel, and HR-Mel representations on an input file."""

import argparse
import io
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

import librosa  # noqa: E402  # isort:skip

from generate_mel_variants import (
    DEFAULT_FMAX,
    DEFAULT_SR,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    build_hr_mel_basis,
)


def rel_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(target, "fro") + 1e-12
    return float(np.linalg.norm(target - approx, "fro") / denom)


def decode_hr(encoded: np.ndarray, slices: List[slice]) -> np.ndarray:
    decoded = encoded.copy()
    split_mid = slices[1].stop
    decoded[:split_mid] = np.expm1(decoded[:split_mid])
    decoded[split_mid:] = np.expm1(decoded[split_mid:] ** 2)
    return decoded


def compressed_size_bytes(**arrays: np.ndarray) -> int:
    """Return size in bytes of a compressed npz containing the provided arrays."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return len(buf.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze STFT/Mel/HR-Mel representations.")
    parser.add_argument("--input", type=Path, default=Path("playlist.mp3"), help="Input audio file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for analysis output")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Hz)")
    parser.add_argument(
        "--fmax", type=float, default=DEFAULT_FMAX, help="Upper frequency for Mel filters (Hz, clipped to Nyquist)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    input_path: Path = args.input
    out_dir: Path = args.output_dir
    target_sr = args.sr
    fmax = min(args.fmax, target_sr / 2.0)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    stft_power = np.abs(
        librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    ) ** 2

    frames = stft_power.shape[1]
    results: Dict[str, Dict] = {}

    # STFT baseline
    results["stft"] = {
        "bins": int(stft_power.shape[0]),
        "frames": int(frames),
        "relative_recon_error": 0.0,
        "bytes_compressed": compressed_size_bytes(S=stft_power),
    }

    # Standard Mel (power)
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=N_FFT, n_mels=80, fmax=fmax, norm="slaney"
    )
    mel_power = mel_basis @ stft_power
    mel_pinv = np.linalg.pinv(mel_basis)
    mel_recon = np.maximum(mel_pinv @ mel_power, 0)
    results["mel"] = {
        "bins": int(mel_power.shape[0]),
        "frames": int(mel_power.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel_recon),
        "bytes_compressed": compressed_size_bytes(M=mel_power),
        "note": "80-bin mel power",
    }

    # Log-Mel
    mel_log = np.log1p(mel_power)
    mel_from_log = np.expm1(mel_log)
    mel_log_recon = np.maximum(mel_pinv @ mel_from_log, 0)
    results["log_mel"] = {
        "bins": int(mel_log.shape[0]),
        "frames": int(mel_log.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel_log_recon),
        "bytes_compressed": compressed_size_bytes(M=mel_log),
        "note": "log1p on 80-bin mel power",
    }

    # Uniform Mel with same bin count as HR (96 bins)
    mel96_basis = librosa.filters.mel(
        sr=sr, n_fft=N_FFT, n_mels=96, fmax=fmax, norm="slaney"
    )
    mel96_power = mel96_basis @ stft_power
    mel96_pinv = np.linalg.pinv(mel96_basis)
    mel96_recon = np.maximum(mel96_pinv @ mel96_power, 0)
    mel96_log = np.log1p(mel96_power)
    mel96_log_recon = np.maximum(mel96_pinv @ np.expm1(mel96_log), 0)
    results["mel_96"] = {
        "bins": int(mel96_power.shape[0]),
        "frames": int(mel96_power.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel96_recon),
        "bytes_compressed": compressed_size_bytes(M=mel96_power),
        "note": "96-bin mel power (baseline to match HR bins)",
    }
    results["log_mel_96"] = {
        "bins": int(mel96_log.shape[0]),
        "frames": int(mel96_log.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel96_log_recon),
        "bytes_compressed": compressed_size_bytes(M=mel96_log),
        "note": "log1p on 96-bin mel power",
    }

    # HR-Mel
    hr_basis, hr_slices = build_hr_mel_basis(sr, N_FFT, fmax)
    hr_mel_power = hr_basis @ stft_power
    hr_encoded = hr_mel_power.copy()
    hr_encoded[hr_slices[0]] = np.log1p(hr_encoded[hr_slices[0]])
    hr_encoded[hr_slices[1]] = np.log1p(hr_encoded[hr_slices[1]])
    hr_encoded[hr_slices[2]] = np.sqrt(np.log1p(hr_encoded[hr_slices[2]]))
    hr_decoded = decode_hr(hr_encoded, hr_slices)
    # Default rcond works well for these bases; tune rcond if experimenting with other configs.
    hr_pinv = np.linalg.pinv(hr_basis)
    hr_recon = np.maximum(hr_pinv @ hr_decoded, 0)
    results["hr_mel"] = {
        "bins": int(hr_encoded.shape[0]),
        "frames": int(hr_encoded.shape[1]),
        "relative_recon_error": rel_error(stft_power, hr_recon),
        "bytes_compressed": compressed_size_bytes(M=hr_encoded),
        "note": "40/32/24 bins with sqrt-log in the top band",
    }

    summary = {
        "input_sr": sr,
        "duration_sec": round(len(y) / sr, 2),
        "frames": int(frames),
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "fmax": fmax,
        "representations": results,
    }

    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
