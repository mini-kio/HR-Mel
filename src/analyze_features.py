#!/usr/bin/env python3
"""Compare STFT, Mel, Log-Mel, and HR-Mel across one or more input files."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

import librosa  # noqa: E402  # isort:skip

from utils.analysis_utils import compressed_size_bytes, mean_std, rel_error
from hr_mel import (
    DEFAULT_BANDS,
    DEFAULT_FMAX,
    DEFAULT_SR,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    build_hr_mel_basis,
    decode_hr,
    encode_hr,
)
from utils.mel_utils import log_compress, log_decompress, mel_power

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze STFT/Mel/HR-Mel representations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Input audio file or directory containing audio files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for analysis output")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Hz)")
    parser.add_argument(
        "--fmax", type=float, default=DEFAULT_FMAX, help="Upper frequency for Mel filters (Hz, clipped to Nyquist)"
    )
    return parser.parse_args()


def collect_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No audio files with extensions {sorted(AUDIO_EXTENSIONS)} found in {input_path}")
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def analyze_audio_file(input_path: Path, target_sr: int, fmax: float) -> Dict:
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
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=80, fmax=fmax, norm="slaney")
    mel_power_80 = mel_basis @ stft_power
    mel_pinv = np.linalg.pinv(mel_basis)
    mel_recon = np.maximum(mel_pinv @ mel_power_80, 0)
    results["mel"] = {
        "bins": int(mel_power_80.shape[0]),
        "frames": int(mel_power_80.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel_recon),
        "bytes_compressed": compressed_size_bytes(M=mel_power_80),
        "note": "80-bin mel power",
    }

    # Log-Mel
    mel_log = log_compress(mel_power_80)
    mel_from_log = log_decompress(mel_log)
    mel_log_recon = np.maximum(mel_pinv @ mel_from_log, 0)
    results["log_mel"] = {
        "bins": int(mel_log.shape[0]),
        "frames": int(mel_log.shape[1]),
        "relative_recon_error": rel_error(stft_power, mel_log_recon),
        "bytes_compressed": compressed_size_bytes(M=mel_log),
        "note": "log1p on 80-bin mel power",
    }

    # Uniform Mel with same bin count as HR (96 bins)
    mel96_power, mel96_basis, mel96_pinv = mel_power(
        stft_power, sr=sr, n_fft=N_FFT, n_mels=96, fmax=fmax
    )
    mel96_recon = np.maximum(mel96_pinv @ mel96_power, 0)
    mel96_log = log_compress(mel96_power)
    mel96_log_recon = np.maximum(mel96_pinv @ log_decompress(mel96_log), 0)
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
    hr_basis, hr_slices = build_hr_mel_basis(sr, N_FFT, fmax, bands=DEFAULT_BANDS)
    hr_mel_power = hr_basis @ stft_power
    hr_encoded = encode_hr(hr_mel_power, hr_slices, bands=DEFAULT_BANDS)
    hr_decoded = decode_hr(hr_encoded, hr_slices, bands=DEFAULT_BANDS)
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
        "file": str(input_path),
    }

    return summary


def aggregate_results(per_file: List[Dict]) -> Dict:
    aggregate = {
        "file_count": len(per_file),
        "duration_sec": mean_std(f["duration_sec"] for f in per_file),
        "frames": mean_std(f["frames"] for f in per_file),
        "representations": {},
    }

    rep_names = per_file[0]["representations"].keys()
    for rep in rep_names:
        sample = per_file[0]["representations"][rep]
        rep_stats: Dict[str, Dict] = {}
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                values = [f["representations"][rep][key] for f in per_file]
                rep_stats[key] = mean_std(values)
            else:
                rep_stats[key] = value
        aggregate["representations"][rep] = rep_stats

    return aggregate


def main(args: argparse.Namespace) -> None:
    input_path: Path = args.input
    out_dir: Path = args.output_dir
    target_sr = args.sr
    fmax = min(args.fmax, target_sr / 2.0)

    audio_files = collect_audio_files(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_file = [analyze_audio_file(audio_file, target_sr, fmax) for audio_file in audio_files]
    aggregate = aggregate_results(per_file)

    report = {
        "settings": {
            "target_sr": target_sr,
            "fmax": fmax,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
        },
        "audio_files": [str(p) for p in audio_files],
        "per_file": per_file,
        "aggregate": aggregate,
    }

    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
