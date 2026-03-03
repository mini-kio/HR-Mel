#!/usr/bin/env python3
"""Run step-by-step paradigm studies for HR-Mel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa

from src.hr_mel import (
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
from src.utils.analysis_utils import compressed_size_bytes, rel_error
from src.utils.mel_utils import log_compress, log_decompress, mel_power

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass(frozen=True)
class HRConfig:
    f1: float
    f2: float
    bins: Tuple[int, int, int]
    high_compression: str

    def to_bands(self, fmax: float) -> List[Dict]:
        b1, b2, b3 = self.bins
        return [
            {"fmin": 0.0, "fmax": float(self.f1), "bins": int(b1), "compression": "log1p"},
            {"fmin": float(self.f1), "fmax": float(self.f2), "bins": int(b2), "compression": "log1p"},
            {"fmin": float(self.f2), "fmax": float(fmax), "bins": int(b3), "compression": self.high_compression},
        ]

    @property
    def name(self) -> str:
        b1, b2, b3 = self.bins
        return (
            f"f{int(round(self.f1))}-{int(round(self.f2))}_"
            f"b{b1}-{b2}-{b3}_{self.high_compression}"
        )


@dataclass
class FileBundle:
    path: Path
    y: np.ndarray
    sr: int
    duration_sec: float
    stft_power: np.ndarray
    mel96_power: np.ndarray
    mel96_basis: np.ndarray
    mel96_pinv: np.ndarray
    mel96_log: np.ndarray
    centroid_hz: float
    flatness: float
    onset_strength: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paradigm-level HR-Mel studies.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT,
        help="Input audio file or directory with audio files (default: repo root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory where study outputs are written.",
    )
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate.")
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX, help="Max mel frequency.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of audio files (>0 means first N files).",
    )
    return parser.parse_args()


def collect_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No audio files with extensions {sorted(AUDIO_EXTENSIONS)} in {input_path}"
        )
    return files


def load_bundle(path: Path, target_sr: int, fmax: float) -> FileBundle:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    stft_complex = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    stft_power = np.abs(stft_complex) ** 2

    mel96_power, mel96_basis, mel96_pinv = mel_power(
        stft_power=stft_power, sr=sr, n_fft=N_FFT, n_mels=96, fmax=fmax
    )
    mel96_log = log_compress(mel96_power)

    centroid = float(
        np.mean(librosa.feature.spectral_centroid(S=np.abs(stft_complex), sr=sr))
    )
    flatness = float(
        np.mean(librosa.feature.spectral_flatness(S=np.abs(stft_complex) + 1e-10))
    )
    onset_strength = float(
        np.mean(librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH))
    )

    return FileBundle(
        path=path,
        y=y,
        sr=sr,
        duration_sec=float(len(y) / sr),
        stft_power=stft_power,
        mel96_power=mel96_power,
        mel96_basis=mel96_basis,
        mel96_pinv=mel96_pinv,
        mel96_log=mel96_log,
        centroid_hz=centroid,
        flatness=flatness,
        onset_strength=onset_strength,
    )


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def evaluate_hr_config(bundle: FileBundle, fmax: float, config: HRConfig) -> Dict[str, float]:
    bands = config.to_bands(fmax=fmax)
    basis, slices = build_hr_mel_basis(bundle.sr, N_FFT, fmax, bands=bands)
    mel = basis @ bundle.stft_power
    encoded = encode_hr(mel, slices, bands=bands)
    decoded = decode_hr(encoded, slices, bands=bands)
    pinv = np.linalg.pinv(basis)
    recon = np.maximum(pinv @ decoded, 0.0)
    error = rel_error(bundle.stft_power, recon)
    size_bytes = compressed_size_bytes(M=encoded)
    return {"error": error, "size_bytes": float(size_bytes)}


def generate_design_space_configs() -> List[HRConfig]:
    f1_candidates = [1_200.0, 1_500.0, 1_800.0]
    f2_candidates = [5_000.0, 6_000.0, 7_500.0]
    bin_candidates = [
        (48, 32, 16),
        (44, 32, 20),
        (40, 32, 24),
        (36, 36, 24),
        (32, 32, 32),
        (24, 32, 40),
    ]
    high_compressions = ["log1p", "sqrt_log1p", "pow075"]

    configs: List[HRConfig] = []
    for f1 in f1_candidates:
        for f2 in f2_candidates:
            if f2 <= f1:
                continue
            for bins in bin_candidates:
                if sum(bins) != 96:
                    continue
                for comp in high_compressions:
                    configs.append(HRConfig(f1=f1, f2=f2, bins=bins, high_compression=comp))
    return configs


def run_study_01_design_space(
    bundles: Sequence[FileBundle], out_dir: Path, fmax: float
) -> Dict:
    configs = generate_design_space_configs()
    baseline_size = mean(compressed_size_bytes(M=b.mel96_log) for b in bundles)

    per_file_rows: List[Dict] = []
    aggregate_rows: List[Dict] = []

    by_config: Dict[str, List[Dict]] = {}
    for config in configs:
        config_rows: List[Dict] = []
        for bundle in bundles:
            res = evaluate_hr_config(bundle=bundle, fmax=fmax, config=config)
            row = {
                "file": bundle.path.name,
                "config": config.name,
                "f1_hz": config.f1,
                "f2_hz": config.f2,
                "bins_low": config.bins[0],
                "bins_mid": config.bins[1],
                "bins_high": config.bins[2],
                "high_compression": config.high_compression,
                "relative_error": res["error"],
                "size_bytes": int(res["size_bytes"]),
            }
            config_rows.append(row)
            per_file_rows.append(row)
        by_config[config.name] = config_rows

    for config in configs:
        rows = by_config[config.name]
        mean_error = mean(r["relative_error"] for r in rows)
        mean_size = mean(r["size_bytes"] for r in rows)
        rd_score = mean_error + 0.10 * (mean_size / (baseline_size + 1e-9))
        aggregate_rows.append(
            {
                "config": config.name,
                "f1_hz": config.f1,
                "f2_hz": config.f2,
                "bins_low": config.bins[0],
                "bins_mid": config.bins[1],
                "bins_high": config.bins[2],
                "high_compression": config.high_compression,
                "mean_relative_error": mean_error,
                "mean_size_bytes": int(round(mean_size)),
                "rd_score": rd_score,
            }
        )

    aggregate_rows.sort(key=lambda x: (x["rd_score"], x["mean_relative_error"]))
    best = aggregate_rows[0]

    write_csv(
        out_dir / "study_01_design_space_per_file.csv",
        per_file_rows,
        [
            "file",
            "config",
            "f1_hz",
            "f2_hz",
            "bins_low",
            "bins_mid",
            "bins_high",
            "high_compression",
            "relative_error",
            "size_bytes",
        ],
    )
    write_csv(
        out_dir / "study_01_design_space_aggregate.csv",
        aggregate_rows,
        [
            "config",
            "f1_hz",
            "f2_hz",
            "bins_low",
            "bins_mid",
            "bins_high",
            "high_compression",
            "mean_relative_error",
            "mean_size_bytes",
            "rd_score",
        ],
    )

    top10 = aggregate_rows[:10]
    summary = {"baseline_mel96_log_size_bytes": baseline_size, "top10": top10, "best": best}
    (out_dir / "study_01_best_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return best


def uniform_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    if levels <= 0:
        return values.copy()
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return values.copy()
    q = np.round((values - vmin) / (vmax - vmin) * levels)
    q = np.clip(q, 0, levels)
    return (q / levels) * (vmax - vmin) + vmin


def quantize_bandwise(values: np.ndarray, bits: int, slices: Sequence[slice]) -> np.ndarray:
    out = values.copy()
    for sl in slices:
        out[sl] = uniform_quantize(out[sl], bits)
    return out


def run_study_02_rate_distortion(
    bundles: Sequence[FileBundle], out_dir: Path, fmax: float, best_cfg_row: Dict
) -> None:
    best_cfg = HRConfig(
        f1=float(best_cfg_row["f1_hz"]),
        f2=float(best_cfg_row["f2_hz"]),
        bins=(
            int(best_cfg_row["bins_low"]),
            int(best_cfg_row["bins_mid"]),
            int(best_cfg_row["bins_high"]),
        ),
        high_compression=str(best_cfg_row["high_compression"]),
    )
    best_bands = best_cfg.to_bands(fmax=fmax)

    bits_list = [4, 6, 8, 10, 12, 16]
    rows: List[Dict] = []

    for bundle in bundles:
        frame_rate = bundle.sr / HOP_LENGTH

        hr_basis, hr_slices = build_hr_mel_basis(bundle.sr, N_FFT, fmax, bands=best_bands)
        hr_mel = hr_basis @ bundle.stft_power
        hr_encoded = encode_hr(hr_mel, hr_slices, best_bands)
        hr_pinv = np.linalg.pinv(hr_basis)

        mel_bins = bundle.mel96_log.shape[0]
        hr_bins = hr_encoded.shape[0]

        for bits in bits_list:
            mel_q = uniform_quantize(bundle.mel96_log, bits)
            mel_recon = np.maximum(bundle.mel96_pinv @ log_decompress(mel_q), 0.0)
            mel_error = rel_error(bundle.stft_power, mel_recon)
            rows.append(
                {
                    "file": bundle.path.name,
                    "representation": "mel96_log_q",
                    "bits": bits,
                    "bitrate_kbps": (bits * mel_bins * frame_rate) / 1000.0,
                    "relative_error": mel_error,
                }
            )

            hr_q = quantize_bandwise(hr_encoded, bits, hr_slices)
            hr_decoded = decode_hr(hr_q, hr_slices, best_bands)
            hr_recon = np.maximum(hr_pinv @ hr_decoded, 0.0)
            hr_error = rel_error(bundle.stft_power, hr_recon)
            rows.append(
                {
                    "file": bundle.path.name,
                    "representation": "hr_best_q",
                    "bits": bits,
                    "bitrate_kbps": (bits * hr_bins * frame_rate) / 1000.0,
                    "relative_error": hr_error,
                }
            )

    agg_rows: List[Dict] = []
    for rep in sorted({r["representation"] for r in rows}):
        for bits in sorted({int(r["bits"]) for r in rows}):
            subset = [r for r in rows if r["representation"] == rep and int(r["bits"]) == bits]
            agg_rows.append(
                {
                    "representation": rep,
                    "bits": bits,
                    "mean_bitrate_kbps": mean(r["bitrate_kbps"] for r in subset),
                    "mean_relative_error": mean(r["relative_error"] for r in subset),
                }
            )
    agg_rows.sort(key=lambda x: (x["representation"], x["bits"]))

    write_csv(
        out_dir / "study_02_rate_distortion_per_file.csv",
        rows,
        ["file", "representation", "bits", "bitrate_kbps", "relative_error"],
    )
    write_csv(
        out_dir / "study_02_rate_distortion_aggregate.csv",
        agg_rows,
        ["representation", "bits", "mean_bitrate_kbps", "mean_relative_error"],
    )


def high_band_temporal_contrast(
    y: np.ndarray, sr: int, fmax: float, hop: int, n_fft: int = 1024, bins: int = 24
) -> float:
    stft_power = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, win_length=n_fft)) ** 2
    basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=bins, fmin=6_000.0, fmax=min(fmax, sr / 2.0), norm="slaney"
    )
    mel = np.log1p(basis @ stft_power)
    diff = np.diff(mel, axis=1)
    if not diff.size:
        return 0.0
    # Normalize by frame duration so hops are comparable as a change-rate metric.
    return float(np.mean(np.abs(diff)) / (hop / float(sr)))


def low_band_peak_jitter(
    y: np.ndarray, sr: int, n_fft: int, hop: int = HOP_LENGTH, f_hi: float = 1_500.0
) -> float:
    mag = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, win_length=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = freqs <= f_hi
    if not np.any(mask):
        return 0.0
    band_mag = mag[mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_mag, axis=0)
    peaks = band_freqs[peak_idx]
    d = np.diff(peaks)
    return float(np.std(d)) if d.size else 0.0


def run_study_03_multirate_evidence(
    bundles: Sequence[FileBundle], out_dir: Path, fmax: float
) -> None:
    rows: List[Dict] = []
    for bundle in bundles:
        high_long = high_band_temporal_contrast(bundle.y, bundle.sr, fmax=fmax, hop=HOP_LENGTH)
        high_short = high_band_temporal_contrast(bundle.y, bundle.sr, fmax=fmax, hop=220)
        low_short_fft = low_band_peak_jitter(bundle.y, bundle.sr, n_fft=1024)
        low_long_fft = low_band_peak_jitter(bundle.y, bundle.sr, n_fft=2048)

        rows.append(
            {
                "file": bundle.path.name,
                "high_temporal_contrast_hop441": high_long,
                "high_temporal_contrast_hop220": high_short,
                "high_contrast_gain_ratio": (high_short + 1e-12) / (high_long + 1e-12),
                "low_peak_jitter_fft1024": low_short_fft,
                "low_peak_jitter_fft2048": low_long_fft,
                "low_jitter_reduction_ratio": (low_long_fft + 1e-12) / (low_short_fft + 1e-12),
            }
        )

    aggregate = {
        "mean_high_contrast_gain_ratio": mean(r["high_contrast_gain_ratio"] for r in rows),
        "mean_low_jitter_reduction_ratio": mean(r["low_jitter_reduction_ratio"] for r in rows),
        "suggested_multirate": {
            "low": {"n_fft": 2048, "hop": 441},
            "mid": {"n_fft": 1536, "hop": 320},
            "high": {"n_fft": 1024, "hop": 220},
        },
    }

    write_csv(
        out_dir / "study_03_multirate_evidence.csv",
        rows,
        [
            "file",
            "high_temporal_contrast_hop441",
            "high_temporal_contrast_hop220",
            "high_contrast_gain_ratio",
            "low_peak_jitter_fft1024",
            "low_peak_jitter_fft2048",
            "low_jitter_reduction_ratio",
        ],
    )
    (out_dir / "study_03_multirate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )


def a_weighting_db(freq_hz: np.ndarray) -> np.ndarray:
    f = np.maximum(freq_hz.astype(float), 1e-6)
    f2 = f * f
    ra_num = (12200.0**2) * (f2**2)
    ra_den = (f2 + 20.6**2) * (f2 + 12200.0**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
    ra = ra_num / (ra_den + 1e-20)
    return 2.0 + 20.0 * np.log10(np.maximum(ra, 1e-20))


def allocate_bins_from_weights(
    weights: Sequence[float], total_bins: int = 96, min_each: int = 12
) -> Tuple[int, int, int]:
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    if w.size != 3:
        raise ValueError("Expected 3-band weights.")
    base = np.full(3, min_each, dtype=int)
    remaining = int(total_bins - int(np.sum(base)))
    if remaining < 0:
        raise ValueError("min_each too large for total_bins")
    prop = w / np.sum(w)
    raw = prop * remaining
    add = np.floor(raw).astype(int)
    bins = base + add
    leftover = int(total_bins - int(np.sum(bins)))
    if leftover > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        for idx in order[:leftover]:
            bins[idx] += 1
    return int(bins[0]), int(bins[1]), int(bins[2])


def run_study_04_psycho_allocator(
    bundles: Sequence[FileBundle], out_dir: Path, fmax: float
) -> None:
    rows: List[Dict] = []
    baseline_cfg = HRConfig(f1=1500.0, f2=6000.0, bins=(40, 32, 24), high_compression="sqrt_log1p")

    for bundle in bundles:
        freqs = librosa.fft_frequencies(sr=bundle.sr, n_fft=N_FFT)
        avg_spec = np.mean(bundle.stft_power, axis=1)
        adb = a_weighting_db(freqs)
        a_linear = 10.0 ** (adb / 10.0)

        masks = [
            (freqs >= 0.0) & (freqs < 1500.0),
            (freqs >= 1500.0) & (freqs < 6000.0),
            (freqs >= 6000.0) & (freqs <= fmax),
        ]
        weighted_energy = [float(np.sum(avg_spec[m] * a_linear[m])) for m in masks]
        bins = allocate_bins_from_weights(weighted_energy, total_bins=96, min_each=12)
        psy_cfg = HRConfig(f1=1500.0, f2=6000.0, bins=bins, high_compression="sqrt_log1p")

        base_res = evaluate_hr_config(bundle=bundle, fmax=fmax, config=baseline_cfg)
        psy_res = evaluate_hr_config(bundle=bundle, fmax=fmax, config=psy_cfg)

        rows.append(
            {
                "file": bundle.path.name,
                "psy_bins_low": bins[0],
                "psy_bins_mid": bins[1],
                "psy_bins_high": bins[2],
                "baseline_error": base_res["error"],
                "psy_error": psy_res["error"],
                "error_improvement": base_res["error"] - psy_res["error"],
                "baseline_size_bytes": int(base_res["size_bytes"]),
                "psy_size_bytes": int(psy_res["size_bytes"]),
            }
        )

    aggregate = {
        "mean_error_improvement": mean(r["error_improvement"] for r in rows),
        "mean_baseline_error": mean(r["baseline_error"] for r in rows),
        "mean_psy_error": mean(r["psy_error"] for r in rows),
    }
    write_csv(
        out_dir / "study_04_psycho_allocator.csv",
        rows,
        [
            "file",
            "psy_bins_low",
            "psy_bins_mid",
            "psy_bins_high",
            "baseline_error",
            "psy_error",
            "error_improvement",
            "baseline_size_bytes",
            "psy_size_bytes",
        ],
    )
    (out_dir / "study_04_psycho_allocator_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )


def run_study_05_content_adaptive(
    bundles: Sequence[FileBundle], out_dir: Path, fmax: float
) -> None:
    profiles = {
        "music_lowharm": HRConfig(f1=1800.0, f2=7000.0, bins=(48, 32, 16), high_compression="log1p"),
        "music_balanced": HRConfig(f1=1500.0, f2=6000.0, bins=(40, 32, 24), high_compression="sqrt_log1p"),
        "music_transient": HRConfig(f1=1200.0, f2=5000.0, bins=(24, 32, 40), high_compression="sqrt_log1p"),
    }

    centroid_threshold = float(np.median([b.centroid_hz for b in bundles]))
    onset_threshold = float(np.median([b.onset_strength for b in bundles]))
    flatness_threshold = float(np.median([b.flatness for b in bundles]))

    def pick_profile(bundle: FileBundle) -> str:
        if bundle.flatness < flatness_threshold:
            return "music_lowharm"
        if bundle.onset_strength > onset_threshold and bundle.centroid_hz > centroid_threshold:
            return "music_transient"
        return "music_balanced"

    rows: List[Dict] = []
    for bundle in bundles:
        chosen = pick_profile(bundle)
        chosen_cfg = profiles[chosen]
        chosen_res = evaluate_hr_config(bundle=bundle, fmax=fmax, config=chosen_cfg)
        baseline_res = evaluate_hr_config(bundle=bundle, fmax=fmax, config=profiles["music_balanced"])

        oracle_name = ""
        oracle_error = float("inf")
        for name, cfg in profiles.items():
            err = evaluate_hr_config(bundle=bundle, fmax=fmax, config=cfg)["error"]
            if err < oracle_error:
                oracle_error = err
                oracle_name = name

        rows.append(
            {
                "file": bundle.path.name,
                "centroid_hz": bundle.centroid_hz,
                "flatness": bundle.flatness,
                "onset_strength": bundle.onset_strength,
                "chosen_profile": chosen,
                "chosen_error": chosen_res["error"],
                "baseline_profile": "music_balanced",
                "baseline_error": baseline_res["error"],
                "oracle_profile": oracle_name,
                "oracle_error": oracle_error,
                "adaptive_gain_vs_baseline": baseline_res["error"] - chosen_res["error"],
                "gap_to_oracle": chosen_res["error"] - oracle_error,
            }
        )

    aggregate = {
        "centroid_threshold_hz": centroid_threshold,
        "onset_threshold": onset_threshold,
        "flatness_threshold": flatness_threshold,
        "mean_adaptive_gain_vs_baseline": mean(r["adaptive_gain_vs_baseline"] for r in rows),
        "mean_gap_to_oracle": mean(r["gap_to_oracle"] for r in rows),
    }
    write_csv(
        out_dir / "study_05_content_adaptive.csv",
        rows,
        [
            "file",
            "centroid_hz",
            "flatness",
            "onset_strength",
            "chosen_profile",
            "chosen_error",
            "baseline_profile",
            "baseline_error",
            "oracle_profile",
            "oracle_error",
            "adaptive_gain_vs_baseline",
            "gap_to_oracle",
        ],
    )
    (out_dir / "study_05_content_adaptive_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )


def run_study_06_standard_api(out_dir: Path, best_cfg_row: Dict) -> None:
    best_profile = {
        "name": "music-v1",
        "f1_hz": float(best_cfg_row["f1_hz"]),
        "f2_hz": float(best_cfg_row["f2_hz"]),
        "bins": [
            int(best_cfg_row["bins_low"]),
            int(best_cfg_row["bins_mid"]),
            int(best_cfg_row["bins_high"]),
        ],
        "high_compression": str(best_cfg_row["high_compression"]),
    }

    api_spec = {
        "class": "HRMelSpec",
        "constructor": {
            "signature": "HRMelSpec(sr=44100, n_fft=2048, hop=441, fmax=20000, profile='music-v1')",
            "profiles": [
                "music-v1",
                "music-lowharm",
                "music-transient",
                "speech-v1",
                "foley-v1",
            ],
        },
        "methods": [
            "encode(y) -> encoded, meta",
            "decode(encoded, meta) -> pseudo_mel_power",
            "quantize(encoded, mode='bandwise', bits=8) -> encoded_q",
            "select_profile(y) -> profile_name",
        ],
        "benchmark_schema": {
            "rd_metrics": ["bitrate_kbps", "relative_recon_error"],
            "perceptual_metrics": ["loudness_weighted_error", "band_weighted_error"],
            "downstream_metrics": ["codec_mos_proxy", "lm_token_efficiency"],
        },
        "recommended_music_profile": best_profile,
    }

    (out_dir / "study_06_standard_api_spec.json").write_text(
        json.dumps(api_spec, indent=2), encoding="utf-8"
    )


def render_summary(out_dir: Path, best_cfg_row: Dict) -> None:
    rd_agg_path = out_dir / "study_02_rate_distortion_aggregate.csv"
    rd_lines = rd_agg_path.read_text(encoding="utf-8").strip().splitlines()
    rd_preview = "\n".join(rd_lines[:7]) if rd_lines else "(empty)"

    summary = [
        "# HR-Mel Paradigm Study (v1)",
        "",
        "## Completed Steps",
        "1. Design-space search over boundaries/bins/nonlinearity.",
        "2. Rate-distortion under scalar quantization.",
        "3. Multi-rate evidence with high-band transient and low-band stability metrics.",
        "4. Psychoacoustic bin allocator (A-weighted energy).",
        "5. Content-adaptive profile routing.",
        "6. Standard API and benchmark schema draft.",
        "",
        "## Best Config from Study 01",
        json.dumps(best_cfg_row, indent=2),
        "",
        "## RD Preview (first lines)",
        "```csv",
        rd_preview,
        "```",
        "",
        "All detailed outputs are in this folder.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.resolve()
    fmax = min(float(args.fmax), float(args.sr) / 2.0)
    audio_files = collect_audio_files(input_path)
    if args.max_files and args.max_files > 0:
        audio_files = audio_files[: args.max_files]
    if not audio_files:
        raise RuntimeError("No audio files selected for analysis.")

    bundles = [load_bundle(path=p, target_sr=args.sr, fmax=fmax) for p in audio_files]

    best_cfg_row = run_study_01_design_space(bundles=bundles, out_dir=out_dir, fmax=fmax)
    run_study_02_rate_distortion(
        bundles=bundles, out_dir=out_dir, fmax=fmax, best_cfg_row=best_cfg_row
    )
    run_study_03_multirate_evidence(bundles=bundles, out_dir=out_dir, fmax=fmax)
    run_study_04_psycho_allocator(bundles=bundles, out_dir=out_dir, fmax=fmax)
    run_study_05_content_adaptive(bundles=bundles, out_dir=out_dir, fmax=fmax)
    run_study_06_standard_api(out_dir=out_dir, best_cfg_row=best_cfg_row)
    render_summary(out_dir=out_dir, best_cfg_row=best_cfg_row)

    run_meta = {
        "input_path": str(input_path),
        "files": [p.name for p in audio_files],
        "target_sr": args.sr,
        "fmax": fmax,
        "output_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
