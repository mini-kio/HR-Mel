# HR-Mel (44.1 kHz)

[Korean README](README_KR.md) · [License](LICENSE)

HR-Mel is a 3-band Mel front-end that reallocates capacity toward low frequencies while preserving high-band detail. This repo ships a minimal pipeline to extract HR-Mel at fixed 44.1 kHz settings and to compare it against STFT / Mel / Log-Mel in terms of relative reconstruction error and compressed size.

## Specs
- Input: `playlist.mp3` (forced mono, default `sr=44,100`)
- STFT: `n_fft=2048`, `hop_length=441` (≈10 ms), `win_length=2048`
- Mel upper bound: `fmax=20,000 Hz`
- HR-Mel bands: `0–1.5 kHz` 40 bins log1p / `1.5–6 kHz` 32 bins log1p / `6–20 kHz` 24 bins sqrt(log1p) → total 96 bins

### Motivation
- Standard log-mel for music often drops high-band detail. HR-Mel aims to cut STFT reconstruction error while keeping dimensionality small (96 bins).
- Fits 44.1 kHz music front-ends (neural audio codecs, music LMs) where “air/attack” retention matters.

### Limitations
- Defaults are fixed for 44.1 kHz (`n_fft=2048`, `hop=441`, `win=2048`). Other configs are untested.
- HR-Mel high band is capped by `fmax` (clipped to Nyquist). Use matching params between extraction and analysis.

## Dataset
- **File count**: 12 tracks
- **Duration**: Average ~2–3 minutes per track
- **Source**: Personal collection, various styles (vocal/instrumental mix)
- **Sample rate**: 44.1 kHz
- **Format**: WAV or MP3

## Usage
```bash
cd HR-Mel
NUMBA_CACHE_DIR=/tmp python -m src.generate_mel_variants \
  --input playlist.mp3 --output-dir output --sr 44100 --fmax 20000

NUMBA_CACHE_DIR=/tmp python -m src.analyze_features \
  --input playlist.mp3 --output-dir output --sr 44100 --fmax 20000
```

Python API (minimal):
```python
from src.hr_mel import hr_mel
encoded, meta = hr_mel(y, sr=44_100)  # y: mono waveform
```

## Outputs
- `output/hr_mel.npz` : HR-Mel encoding + metadata
- `output/summary.json` : input/shape/FFT params
- `output/analysis.json` : relative reconstruction error vs STFT (Frobenius), bins/frames, compressed sizes

`analysis.json` schema (top-level):
- `input_sr`, `duration_sec`, `frames`, `n_fft`, `hop_length`, `win_length`, `fmax`
- `representations`:
  - `bins`, `frames`, `relative_recon_error`, `bytes_compressed`, `note`

## Results

### Overall Performance (12 tracks, mean ± std)
| Representation | Bins | Rel. Error ↓ | Notes |
| --- | --- | --- | --- |
| STFT | 1025 | 0.000 | reference |
| Mel-80 | 80 | 0.4438 ± 0.0404 | 80-bin power Mel |
| Log-Mel-80 | 80 | 0.4438 ± 0.0404 | log1p on 80-bin Mel |
| Mel-96 | 96 | 0.3810 ± 0.0395 | 96-bin power Mel |
| Log-Mel-96 | 96 | 0.3810 ± 0.0395 | log1p on 96-bin Mel |
| **HR-Mel (40/32/24)** | 96 | **0.2845 ± 0.0410** | 40/32/24 bins with sqrt-log top band |

### Band-wise Analysis (Frequency ranges: <1.5 kHz / 1.5–6 kHz / >6 kHz)

**40/32/24 Configuration (Original, Recommended)**
| Band | Mel-96 Error | HR-Mel Error | Delta | Improvement |
| --- | --- | --- | --- | --- |
| Low (<1.5 kHz) | 0.3737 ± 0.0391 | **0.2729 ± 0.0345** | +0.1008 | **~27.0%** |
| Mid (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| High (>6 kHz) | 0.7186 ± 0.0548 | 0.7372 ± 0.0557 | -0.0186 | -2.6% |

**32/32/32 Configuration (Balanced bins)**
| Band | Mel-96 Error | HR-Mel Error | Delta | Improvement |
| --- | --- | --- | --- | --- |
| Low (<1.5 kHz) | 0.3737 ± 0.0391 | 0.4007 ± 0.0394 | -0.0271 | -7.4% |
| Mid (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| High (>6 kHz) | 0.7186 ± 0.0548 | 0.7159 ± 0.0509 | +0.0027 | +0.3% |

**24/32/40 Configuration (High-band focused)**
| Band | Mel-96 Error | HR-Mel Error | Delta | Improvement |
| --- | --- | --- | --- | --- |
| Low (<1.5 kHz) | 0.3737 ± 0.0391 | 0.4840 ± 0.0381 | -0.1104 | -30.5% |
| Mid (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| High (>6 kHz) | 0.7186 ± 0.0548 | **0.6958 ± 0.0429** | +0.0228 | **+3.1%** |

### Key Findings
- **40/32/24 configuration** provides the best overall performance with significant low-band improvement (~27%)
- HR-Mel gains are primarily in the low-frequency range (<1.5 kHz)
- Mid and high bands show slight degradation compared to standard Mel-96
- Balanced (32/32/32) or high-focused (24/32/40) configurations sacrifice low-band performance
- To improve high-band performance while maintaining low-band gains, consider adjusting top-band compression or increasing fmax

Errors are computed by projecting STFT power → representation → pseudo-inverse back to STFT power, then taking relative Frobenius norm.

## Project Structure
```
HR-Mel/
├── src/                          # Source code
│   ├── hr_mel.py                # Core HR-Mel implementation
│   ├── generate_mel_variants.py # HR-Mel feature extractor
│   ├── analyze_features.py      # STFT/Mel/HR-Mel comparison analysis
│   └── utils/                   # Utility modules
│       ├── mel_utils.py         # Mel processing utilities
│       └── analysis_utils.py    # Analysis helper functions
├── output/                      # Generated artifacts (analysis results, encodings)
├── README.md                    # This file
├── README_KR.md                 # Korean documentation
└── LICENSE                      # Apache 2.0
```
