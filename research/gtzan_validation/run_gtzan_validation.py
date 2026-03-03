#!/usr/bin/env python3
"""GTZAN external validation for HR-Mel.

Runs on GTZAN (1000 tracks, 10 genres, native 22.05 kHz → upsampled to 44.1 kHz).
Applies Sturm fault filter, then executes:
  1) Basic representation comparison (STFT / Mel-80 / Mel-96 / HR-Mel original / HR-Mel best)
  2) Rate–Distortion curve reproduction (scalar quantization)
  3) Per-genre breakdown
  4) Genre classification downstream test (SVM on aggregated features)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa

from src.hr_mel import (
    DEFAULT_BANDS,
    DEFAULT_FMAX,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    build_hr_mel_basis,
    decode_hr,
    encode_hr,
)
from src.utils.analysis_utils import compressed_size_bytes, rel_error
from src.utils.mel_utils import log_compress, log_decompress, mel_power

# ── Constants ───────────────────────────────────────────────────────────
TARGET_SR = 44_100
FMAX = 20_000.0
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Sturm fault filter: known corrupt / problematic files in GTZAN.
# Ref: Sturm 2013, "The GTZAN dataset: Its contents, its faults, their effects"
FAULT_FILES = {
    "jazz.00054.wav",      # mostly silence / corrupt
}

# HR-Mel original config (paper v1: 40/32/24)
ORIGINAL_BANDS = DEFAULT_BANDS  # 0-1.5k/40, 1.5-6k/32, 6-20k/24

# HR-Mel best config from Study 01 (48/32/16, f1=1200, f2=5000)
BEST_BANDS = [
    {"fmin": 0.0,      "fmax": 1_200.0,  "bins": 48, "compression": "log1p"},
    {"fmin": 1_200.0,  "fmax": 5_000.0,  "bins": 32, "compression": "log1p"},
    {"fmin": 5_000.0,  "fmax": FMAX,     "bins": 16, "compression": "sqrt_log1p"},
]


@dataclass
class FileResult:
    path: str
    genre: str
    duration_sec: float
    # Representation errors
    mel80_error: float = 0.0
    log_mel80_error: float = 0.0
    mel96_error: float = 0.0
    log_mel96_error: float = 0.0
    hr_orig_error: float = 0.0
    hr_best_error: float = 0.0
    # Band-wise errors for HR-best
    hr_best_low_error: float = 0.0
    hr_best_mid_error: float = 0.0
    hr_best_high_error: float = 0.0
    mel96_low_error: float = 0.0
    mel96_mid_error: float = 0.0
    mel96_high_error: float = 0.0
    # RD data: {bits: {rep: error}}
    rd_data: Dict = field(default_factory=dict)
    # Features for classification
    hr_best_features: Optional[np.ndarray] = None
    mel96_features: Optional[np.ndarray] = None


# ── File collection ─────────────────────────────────────────────────────

def collect_gtzan_files(data_dir: Path) -> List[Tuple[Path, str]]:
    """Recursively collect (path, genre) from genres_original/."""
    genres_dir = data_dir / "genres_original"
    if not genres_dir.is_dir():
        raise FileNotFoundError(f"GTZAN directory not found: {genres_dir}")

    files = []
    for genre_dir in sorted(genres_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        for f in sorted(genre_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTENSIONS and f.name not in FAULT_FILES:
                files.append((f, genre))
    return files


# ── Analysis per file ───────────────────────────────────────────────────

def band_error(stft_power: np.ndarray, recon: np.ndarray, sr: int,
               fmin: float, fmax_band: float) -> float:
    """Compute rel_error over a specific frequency band of the STFT."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    mask = (freqs >= fmin) & (freqs <= fmax_band)
    if not np.any(mask):
        return 0.0
    return rel_error(stft_power[mask], recon[mask])


def uniform_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    if levels <= 0:
        return values.copy()
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-15:
        return values.copy()
    q = np.round((values - vmin) / (vmax - vmin) * levels)
    q = np.clip(q, 0, levels)
    return (q / levels) * (vmax - vmin) + vmin


def quantize_bandwise(values: np.ndarray, bits: int, slices: Sequence[slice]) -> np.ndarray:
    out = values.copy()
    for sl in slices:
        out[sl] = uniform_quantize(out[sl], bits)
    return out


def extract_agg_features(encoded: np.ndarray) -> np.ndarray:
    """Mean + std per mel bin over time → flat feature vector."""
    m = np.mean(encoded, axis=1)
    s = np.std(encoded, axis=1)
    return np.concatenate([m, s])


def analyze_one_file(
    path: Path,
    genre: str,
    mel80_basis: np.ndarray,
    mel80_pinv: np.ndarray,
    mel96_basis: np.ndarray,
    mel96_pinv: np.ndarray,
    hr_orig_basis: np.ndarray,
    hr_orig_slices: List[slice],
    hr_orig_pinv: np.ndarray,
    hr_best_basis: np.ndarray,
    hr_best_slices: List[slice],
    hr_best_pinv: np.ndarray,
    bits_list: Sequence[int],
) -> FileResult:
    """Analyze a single file: upsample, compare representations, compute RD."""
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    duration = len(y) / sr

    stft_power = np.abs(
        librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    ) ** 2

    frame_rate = sr / HOP_LENGTH
    result = FileResult(path=str(path), genre=genre, duration_sec=duration)

    # ── Mel-80 ──
    mel80 = mel80_basis @ stft_power
    mel80_recon = np.maximum(mel80_pinv @ mel80, 0)
    result.mel80_error = rel_error(stft_power, mel80_recon)

    log_mel80 = log_compress(mel80)
    log_mel80_recon = np.maximum(mel80_pinv @ log_decompress(log_mel80), 0)
    result.log_mel80_error = rel_error(stft_power, log_mel80_recon)

    # ── Mel-96 ──
    mel96 = mel96_basis @ stft_power
    mel96_recon = np.maximum(mel96_pinv @ mel96, 0)
    result.mel96_error = rel_error(stft_power, mel96_recon)

    mel96_log = log_compress(mel96)
    mel96_log_recon = np.maximum(mel96_pinv @ log_decompress(mel96_log), 0)
    result.log_mel96_error = rel_error(stft_power, mel96_log_recon)

    # Band-wise for Mel-96
    result.mel96_low_error = band_error(stft_power, mel96_recon, sr, 0, 1500)
    result.mel96_mid_error = band_error(stft_power, mel96_recon, sr, 1500, 6000)
    result.mel96_high_error = band_error(stft_power, mel96_recon, sr, 6000, FMAX)

    # ── HR-Mel Original (40/32/24) ──
    hr_orig_mel = hr_orig_basis @ stft_power
    hr_orig_enc = encode_hr(hr_orig_mel, hr_orig_slices, ORIGINAL_BANDS)
    hr_orig_dec = decode_hr(hr_orig_enc, hr_orig_slices, ORIGINAL_BANDS)
    hr_orig_recon = np.maximum(hr_orig_pinv @ hr_orig_dec, 0)
    result.hr_orig_error = rel_error(stft_power, hr_orig_recon)

    # ── HR-Mel Best (48/32/16) ──
    hr_best_mel = hr_best_basis @ stft_power
    hr_best_enc = encode_hr(hr_best_mel, hr_best_slices, BEST_BANDS)
    hr_best_dec = decode_hr(hr_best_enc, hr_best_slices, BEST_BANDS)
    hr_best_recon = np.maximum(hr_best_pinv @ hr_best_dec, 0)
    result.hr_best_error = rel_error(stft_power, hr_best_recon)

    # Band-wise for HR-best
    result.hr_best_low_error = band_error(stft_power, hr_best_recon, sr, 0, 1200)
    result.hr_best_mid_error = band_error(stft_power, hr_best_recon, sr, 1200, 5000)
    result.hr_best_high_error = band_error(stft_power, hr_best_recon, sr, 5000, FMAX)

    # ── Features for classification ──
    result.hr_best_features = extract_agg_features(hr_best_enc)
    result.mel96_features = extract_agg_features(mel96_log)

    # ── Rate-Distortion ──
    hr_bins = hr_best_enc.shape[0]
    mel_bins = mel96_log.shape[0]
    rd = {}
    for bits in bits_list:
        # Mel-96 log quantized
        mel_q = uniform_quantize(mel96_log, bits)
        mel_q_recon = np.maximum(mel96_pinv @ log_decompress(mel_q), 0)
        mel_err = rel_error(stft_power, mel_q_recon)
        mel_br = (bits * mel_bins * frame_rate) / 1000.0

        # HR-best bandwise quantized
        hr_q = quantize_bandwise(hr_best_enc, bits, hr_best_slices)
        hr_q_dec = decode_hr(hr_q, hr_best_slices, BEST_BANDS)
        hr_q_recon = np.maximum(hr_best_pinv @ hr_q_dec, 0)
        hr_err = rel_error(stft_power, hr_q_recon)
        hr_br = (bits * hr_bins * frame_rate) / 1000.0

        rd[bits] = {
            "mel96_log_q_error": mel_err,
            "mel96_log_q_bitrate": mel_br,
            "hr_best_q_error": hr_err,
            "hr_best_q_bitrate": hr_br,
        }
    result.rd_data = rd

    return result


# ── Aggregation ─────────────────────────────────────────────────────────

def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(values):
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ── Genre classification ────────────────────────────────────────────────

def run_genre_classification(results: List[FileResult], out_dir: Path) -> Dict:
    """Stratified 3-fold CV with SVM for mel-96 vs HR-Mel features."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report

    le = LabelEncoder()
    genres = [r.genre for r in results]
    y = le.fit_transform(genres)

    X_mel = np.array([r.mel96_features for r in results])
    X_hr = np.array([r.hr_best_features for r in results])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    mel_accs, hr_accs = [], []
    mel_reports, hr_reports = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_mel, y)):
        scaler_mel = StandardScaler()
        X_mel_train = scaler_mel.fit_transform(X_mel[train_idx])
        X_mel_test = scaler_mel.transform(X_mel[test_idx])

        scaler_hr = StandardScaler()
        X_hr_train = scaler_hr.fit_transform(X_hr[train_idx])
        X_hr_test = scaler_hr.transform(X_hr[test_idx])

        svm_mel = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm_mel.fit(X_mel_train, y[train_idx])
        mel_pred = svm_mel.predict(X_mel_test)
        mel_acc = accuracy_score(y[test_idx], mel_pred)
        mel_accs.append(mel_acc)

        svm_hr = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm_hr.fit(X_hr_train, y[train_idx])
        hr_pred = svm_hr.predict(X_hr_test)
        hr_acc = accuracy_score(y[test_idx], hr_pred)
        hr_accs.append(hr_acc)

        mel_reports.append(classification_report(y[test_idx], mel_pred,
                                                  target_names=le.classes_, output_dict=True))
        hr_reports.append(classification_report(y[test_idx], hr_pred,
                                                 target_names=le.classes_, output_dict=True))

        print(f"  Fold {fold+1}: Mel-96 acc={mel_acc:.4f}  HR-Mel acc={hr_acc:.4f}")

    summary = {
        "n_folds": 3,
        "note": "Stratified 3-fold CV, SVM(RBF, C=10). Artist-filter not applied (see caveat).",
        "mel96_log": {
            "fold_accuracies": mel_accs,
            "mean_accuracy": float(np.mean(mel_accs)),
            "std_accuracy": float(np.std(mel_accs)),
        },
        "hr_mel_best": {
            "fold_accuracies": hr_accs,
            "mean_accuracy": float(np.mean(hr_accs)),
            "std_accuracy": float(np.std(hr_accs)),
        },
        "features": "mean + std per bin (192 dims for 96-bin representations)",
    }
    (out_dir / "genre_classification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GTZAN validation for HR-Mel")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "Data",
                        help="Path to Data/ directory containing genres_original/")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="Output directory for results")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit files (0 = all)")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip genre classification step")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HR-Mel GTZAN Validation")
    print("=" * 60)

    # ── Collect files ──
    files = collect_gtzan_files(args.data_dir)
    if args.max_files > 0:
        files = files[:args.max_files]

    n_files = len(files)
    genres_found = sorted(set(g for _, g in files))
    print(f"Files: {n_files} ({len(genres_found)} genres)")
    print(f"Excluded fault files: {FAULT_FILES}")
    print(f"Upsample: 22050 → {TARGET_SR} Hz")
    print()

    # ── Pre-compute bases (shared across files) ──
    print("Pre-computing mel bases...")
    # For basis computation we need a dummy STFT to get n_fft dimension
    mel80_basis = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=80,
                                       fmax=FMAX, norm="slaney")
    mel80_pinv = np.linalg.pinv(mel80_basis)

    mel96_basis = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=96,
                                       fmax=FMAX, norm="slaney")
    mel96_pinv = np.linalg.pinv(mel96_basis)

    hr_orig_basis, hr_orig_slices = build_hr_mel_basis(TARGET_SR, N_FFT, FMAX, ORIGINAL_BANDS)
    hr_orig_pinv = np.linalg.pinv(hr_orig_basis)

    hr_best_basis, hr_best_slices = build_hr_mel_basis(TARGET_SR, N_FFT, FMAX, BEST_BANDS)
    hr_best_pinv = np.linalg.pinv(hr_best_basis)

    bits_list = [2, 4, 6, 8, 10, 12, 16]

    # ── Process each file ──
    results: List[FileResult] = []
    t_start = time.time()

    for i, (path, genre) in enumerate(files):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / (i + 1)) * (n_files - i - 1) if i > 0 else 0
            print(f"  [{i+1:4d}/{n_files}] {genre}/{path.name}  "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        try:
            r = analyze_one_file(
                path=path, genre=genre,
                mel80_basis=mel80_basis, mel80_pinv=mel80_pinv,
                mel96_basis=mel96_basis, mel96_pinv=mel96_pinv,
                hr_orig_basis=hr_orig_basis, hr_orig_slices=hr_orig_slices,
                hr_orig_pinv=hr_orig_pinv,
                hr_best_basis=hr_best_basis, hr_best_slices=hr_best_slices,
                hr_best_pinv=hr_best_pinv,
                bits_list=bits_list,
            )
            results.append(r)
        except Exception as e:
            print(f"  WARN: skipping {path.name}: {e}")

    total_time = time.time() - t_start
    print(f"\nProcessed {len(results)} files in {total_time:.1f}s")
    print()

    # ── 1) Basic comparison aggregate ──
    print("=== 1) Basic Representation Comparison ===")
    rep_names = ["mel80", "log_mel80", "mel96", "log_mel96", "hr_orig", "hr_best"]
    for name in rep_names:
        errors = [getattr(r, f"{name}_error") for r in results]
        m, s = mean_std(errors)
        print(f"  {name:15s}: {m:.4f} ± {s:.4f}")

    # Per-file CSV
    per_file_rows = []
    for r in results:
        per_file_rows.append({
            "file": Path(r.path).name,
            "genre": r.genre,
            "mel80_error": r.mel80_error,
            "log_mel80_error": r.log_mel80_error,
            "mel96_error": r.mel96_error,
            "log_mel96_error": r.log_mel96_error,
            "hr_orig_error": r.hr_orig_error,
            "hr_best_error": r.hr_best_error,
        })
    write_csv(out_dir / "comparison_per_file.csv", per_file_rows,
              ["file", "genre"] + [f"{n}_error" for n in rep_names])

    # Aggregate JSON
    comparison_agg = {}
    for name in rep_names:
        errors = [getattr(r, f"{name}_error") for r in results]
        m, s = mean_std(errors)
        comparison_agg[name] = {"mean": m, "std": s}
    (out_dir / "comparison_aggregate.json").write_text(
        json.dumps(comparison_agg, indent=2), encoding="utf-8")

    # ── Per-genre breakdown ──
    print("\n=== Per-Genre Breakdown (HR-best vs Mel-96) ===")
    genre_rows = []
    for genre in genres_found:
        g_results = [r for r in results if r.genre == genre]
        mel96_m, mel96_s = mean_std(r.mel96_error for r in g_results)
        hr_m, hr_s = mean_std(r.hr_best_error for r in g_results)
        improvement = (mel96_m - hr_m) / mel96_m * 100 if mel96_m > 0 else 0
        print(f"  {genre:12s}: mel96={mel96_m:.4f}±{mel96_s:.4f}  "
              f"hr_best={hr_m:.4f}±{hr_s:.4f}  Δ={improvement:+.1f}%")
        genre_rows.append({
            "genre": genre, "n_files": len(g_results),
            "mel96_mean": mel96_m, "mel96_std": mel96_s,
            "hr_best_mean": hr_m, "hr_best_std": hr_s,
            "improvement_pct": improvement,
        })
    write_csv(out_dir / "per_genre_comparison.csv", genre_rows,
              ["genre", "n_files", "mel96_mean", "mel96_std",
               "hr_best_mean", "hr_best_std", "improvement_pct"])

    # ── Band-wise comparison ──
    print("\n=== Band-Wise Errors (HR-best vs Mel-96) ===")
    band_labels = [
        ("low",  "mel96_low_error",  "hr_best_low_error"),
        ("mid",  "mel96_mid_error",  "hr_best_mid_error"),
        ("high", "mel96_high_error", "hr_best_high_error"),
    ]
    band_summary = {}
    for label, mel_attr, hr_attr in band_labels:
        mel_m, mel_s = mean_std(getattr(r, mel_attr) for r in results)
        hr_m, hr_s = mean_std(getattr(r, hr_attr) for r in results)
        imp = (mel_m - hr_m) / mel_m * 100 if mel_m > 0 else 0
        print(f"  {label:5s}: mel96={mel_m:.4f}±{mel_s:.4f}  "
              f"hr_best={hr_m:.4f}±{hr_s:.4f}  Δ={imp:+.1f}%")
        band_summary[label] = {
            "mel96_mean": mel_m, "mel96_std": mel_s,
            "hr_best_mean": hr_m, "hr_best_std": hr_s,
            "improvement_pct": imp,
        }
    (out_dir / "bandwise_comparison.json").write_text(
        json.dumps(band_summary, indent=2), encoding="utf-8")

    # ── 2) Rate-Distortion curves ──
    print("\n=== 2) Rate-Distortion Curves ===")
    rd_rows = []
    for bits in bits_list:
        mel_errors = [r.rd_data[bits]["mel96_log_q_error"] for r in results]
        hr_errors = [r.rd_data[bits]["hr_best_q_error"] for r in results]
        mel_br = results[0].rd_data[bits]["mel96_log_q_bitrate"]
        hr_br = results[0].rd_data[bits]["hr_best_q_bitrate"]
        mel_m, mel_s = mean_std(mel_errors)
        hr_m, hr_s = mean_std(hr_errors)
        print(f"  {bits:2d}-bit: mel96_log_q={mel_m:.4f}±{mel_s:.4f} ({mel_br:.1f} kbps)  "
              f"hr_best_q={hr_m:.4f}±{hr_s:.4f} ({hr_br:.1f} kbps)")
        rd_rows.append({
            "bits": bits,
            "mel96_log_q_mean_error": mel_m, "mel96_log_q_std_error": mel_s,
            "mel96_log_q_bitrate_kbps": mel_br,
            "hr_best_q_mean_error": hr_m, "hr_best_q_std_error": hr_s,
            "hr_best_q_bitrate_kbps": hr_br,
        })
    write_csv(out_dir / "rate_distortion.csv", rd_rows,
              ["bits", "mel96_log_q_mean_error", "mel96_log_q_std_error",
               "mel96_log_q_bitrate_kbps",
               "hr_best_q_mean_error", "hr_best_q_std_error",
               "hr_best_q_bitrate_kbps"])

    # Per-file RD CSV
    rd_per_file = []
    for r in results:
        for bits in bits_list:
            rd_per_file.append({
                "file": Path(r.path).name,
                "genre": r.genre,
                "bits": bits,
                "mel96_log_q_error": r.rd_data[bits]["mel96_log_q_error"],
                "hr_best_q_error": r.rd_data[bits]["hr_best_q_error"],
            })
    write_csv(out_dir / "rate_distortion_per_file.csv", rd_per_file,
              ["file", "genre", "bits", "mel96_log_q_error", "hr_best_q_error"])

    # ── 3) Genre classification ──
    if not args.skip_classification:
        print("\n=== 3) Genre Classification (SVM, 3-fold CV) ===")
        cls_summary = run_genre_classification(results, out_dir)
        print(f"  Mel-96 acc: {cls_summary['mel96_log']['mean_accuracy']:.4f} "
              f"± {cls_summary['mel96_log']['std_accuracy']:.4f}")
        print(f"  HR-Mel acc: {cls_summary['hr_mel_best']['mean_accuracy']:.4f} "
              f"± {cls_summary['hr_mel_best']['std_accuracy']:.4f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mel96_m, _ = mean_std(r.mel96_error for r in results)
    hr_best_m, _ = mean_std(r.hr_best_error for r in results)
    hr_orig_m, _ = mean_std(r.hr_orig_error for r in results)
    overall_imp = (mel96_m - hr_best_m) / mel96_m * 100
    print(f"Mel-96 mean error:     {mel96_m:.4f}")
    print(f"HR-Mel orig (40/32/24): {hr_orig_m:.4f}")
    print(f"HR-Mel best (48/32/16): {hr_best_m:.4f}")
    print(f"Improvement over Mel-96: {overall_imp:+.1f}%")

    summary = {
        "dataset": "GTZAN",
        "n_files": len(results),
        "n_genres": len(genres_found),
        "genres": genres_found,
        "upsample": f"22050 → {TARGET_SR}",
        "fault_filter": list(FAULT_FILES),
        "processing_time_sec": total_time,
        "comparison": comparison_agg,
        "band_summary": band_summary,
        "improvement_over_mel96_pct": overall_imp,
        "hr_best_config": {
            "f1_hz": 1200, "f2_hz": 5000,
            "bins": [48, 32, 16],
            "high_compression": "sqrt_log1p",
        },
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults written to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
