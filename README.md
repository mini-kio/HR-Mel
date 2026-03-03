# HR-Mel (44.1 kHz)

[Korean README](README_KR.md) · [Paper (Zenodo)](https://zenodo.org/records/18831533) · [License](LICENSE)

HR-Mel is a fixed 3-band Mel front-end for 44.1 kHz music that reallocates capacity toward low frequencies with band-specific nonlinearities. Validated on the GTZAN benchmark (999 tracks, 10 genres), achieving **53.8% lower reconstruction error** than standard 96-bin log-mel at matched dimensionality.

## Configs

| Config | Bands | Bins | Use case |
|---|---|---|---|
| **v1** (original) | 0–1.5k / 1.5–6k / 6–20k | 40/32/24 | Initial design |
| **v2** (optimized) | 0–1.2k / 1.2–5k / 5–20k | 48/32/16 | Best from design-space search |

- STFT: `n_fft=2048`, `hop_length=441` (~10 ms), `win_length=2048`
- Compression: log1p (low/mid), sqrt(log1p) (high)
- Total: 96 bins

## Results

### Overall Performance

| Representation | Bins | 12-track (private) | GTZAN (999 tracks) |
|---|---|---|---|
| Mel-80 | 80 | 0.4438 ± 0.040 | 0.4514 ± 0.058 |
| Mel-96 / Log-Mel-96 | 96 | 0.3810 ± 0.040 | 0.3747 ± 0.059 |
| HR-Mel v1 (40/32/24) | 96 | 0.2845 ± 0.041 | 0.3025 ± 0.056 |
| **HR-Mel v2 (48/32/16)** | **96** | **0.2041 ± 0.041** | **0.1730 ± 0.077** |

### Rate-Distortion (GTZAN, scalar quantization)

| Bits | Log-Mel-96 | HR-Mel v2 |
|---|---|---|
| 4 | 0.4026 | **0.2532** |
| 6 | 0.3764 | **0.1790** |
| 8 | 0.3748 | **0.1733** |

### Per-Genre (GTZAN, HR-Mel v2 vs Mel-96)

| Genre | Mel-96 | HR-Mel v2 | Improvement |
|---|---|---|---|
| blues | 0.3807 | 0.1902 | +50.1% |
| classical | 0.4110 | 0.2150 | +47.7% |
| country | 0.3848 | 0.1793 | +53.4% |
| disco | 0.3278 | 0.1659 | +49.4% |
| hiphop | 0.3608 | 0.1239 | +65.7% |
| jazz | 0.4173 | 0.2019 | +51.6% |
| metal | 0.3589 | 0.1656 | +53.9% |
| pop | 0.3634 | 0.1470 | +59.5% |
| reggae | 0.3690 | 0.1547 | +58.1% |
| rock | 0.3741 | 0.1863 | +50.2% |

### Genre Classification (SVM, 3-fold CV)

| Features | Accuracy |
|---|---|
| Log-Mel-96 | 0.647 ± 0.010 |
| **HR-Mel v2** | **0.660 ± 0.029** |

### Band-Wise (GTZAN, HR-Mel v2)

| Band | Mel-96 | HR-Mel v2 | Improvement |
|---|---|---|---|
| Low (<1.2 kHz) | 0.3645 | **0.1227** | **+66.3%** |
| Mid (1.2–5 kHz) | 0.6209 | 0.6133 | +1.2% |
| High (>5 kHz) | 0.7254 | 0.7936 | -9.4% |

## Usage

```bash
# Basic analysis (single file or directory)
python -m src.analyze_features --input <audio_file_or_dir> --output-dir output

# HR-Mel extraction
python -m src.generate_mel_variants --input <audio_file> --output-dir output
```

```python
from src.hr_mel import hr_mel
encoded, meta = hr_mel(y, sr=44_100)  # y: mono waveform
```

## Research Scripts

```bash
# Paradigm studies (design-space search, RD, multi-rate, psycho-allocator, adaptive)
python research/paradigm_v1/run_research.py --input <audio_dir> --output-dir research/paradigm_v1/results

# GTZAN validation (requires GTZAN dataset in Data/genres_original/)
python research/gtzan_validation/run_gtzan_validation.py --data-dir Data --output-dir research/gtzan_validation/results
```

## Project Structure

```
HR-Mel/
├── src/                                    # Core implementation
│   ├── hr_mel.py                          # HR-Mel basis, encode/decode
│   ├── generate_mel_variants.py           # Feature extraction CLI
│   ├── analyze_features.py                # Representation comparison
│   └── utils/
│       ├── mel_utils.py                   # Mel helpers
│       └── analysis_utils.py              # Error/size utilities
├── research/
│   ├── paradigm_v1/
│   │   ├── run_research.py                # 6-study paradigm runner
│   │   ├── hrmel_spec_prototype.py        # Profile-driven API prototype
│   │   └── README.md
│   └── gtzan_validation/
│       └── run_gtzan_validation.py        # GTZAN benchmark test
├── main.tex                               # Paper (LaTeX)
├── README.md
├── README_KR.md
└── LICENSE                                # Apache 2.0
```

## GTZAN Notes

GTZAN is 22.05 kHz natively. The validation script upsamples to 44.1 kHz. Known fault file (jazz.00054.wav) is excluded per [Sturm 2013](https://arxiv.org/abs/1306.1461). Classification uses stratified random splits (not artist-filtered); relative comparisons remain valid.
