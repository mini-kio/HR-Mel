# HR-Mel (44.1 kHz)

[Korean README](README_KR.md) · [License](LICENSE)

HR-Mel is a 3-band Mel front-end that keeps Mel spacing but adds resolution in the highs. This repo ships a minimal pipeline to extract HR-Mel at fixed 44.1 kHz settings and to compare it against STFT / Mel / Log-Mel in terms of relative reconstruction error and compressed size.

## Specs
- Input: `playlist.mp3` (forced mono, `sr=44,100`)
- STFT: `n_fft=2048`, `hop_length=441` (≈10 ms), `win_length=2048`
- Mel upper bound: `fmax=20,000 Hz`
- HR-Mel bands: `0–1.5 kHz` 40 bins log1p / `1.5–6 kHz` 32 bins log1p / `6–20 kHz` 24 bins sqrt(log1p) → total 96 bins

## Usage
```bash
cd HR-mel
NUMBA_CACHE_DIR=/tmp python3 generate_mel_variants.py      # extract HR-Mel
NUMBA_CACHE_DIR=/tmp python3 analyze_features.py           # compare STFT/Mel/Log-Mel/HR-Mel
```

## Outputs
- `output/hr_mel.npz` : HR-Mel encoding + metadata
- `output/summary.json` : input/shape/FFT params
- `output/analysis.json` : relative reconstruction error vs STFT (Frobenius), bins/frames, compressed sizes

## Results (on `playlist.mp3`)
Baseline vs 80-bin Mel
| Rep | Bins | Rel. error ↓ | Notes |
| --- | --- | --- | --- |
| STFT | 1025 | 0.000 | reference |
| Mel (linear) | 80 | 0.461 | 80-bin power Mel |
| Log-Mel | 80 | 0.461 | log1p reversible |

Same dimensionality (96 bins)
| Rep | Bins | Rel. error ↓ | Notes |
| --- | --- | --- | --- |
| Mel-96 | 96 | 0.410 | 96-bin power Mel |
| Log-Mel-96 | 96 | 0.410 | log1p on 96-bin Mel |
| HR-Mel | 96 | 0.304 | 40/32/24 with sqrt-log top band |

Compressed size (np.savez_compressed, bytes)
| Rep | Bins | Size | Notes |
| --- | --- | --- | --- |
| STFT | 1025 | 55.6 MB | power spectrogram |
| Mel | 80 | 4.43 MB | |
| Log-Mel | 80 | 4.41 MB | |
| Mel-96 | 96 | 5.32 MB | |
| Log-Mel-96 | 96 | 5.29 MB | |
| HR-Mel | 96 | 5.26 MB | sqrt-log top band |

Errors are computed by projecting STFT power → representation → pseudo-inverse back to STFT power, then taking relative Frobenius norm. HR-Mel lowers error noticeably at the same 96-bin dimensionality while keeping compressed size on par with 96-bin Mel.

## Files
- `generate_mel_variants.py` : HR-Mel extractor (`playlist.mp3` → `output/`)
- `analyze_features.py` : STFT/Mel/Log-Mel/HR-Mel comparison report
- `output/` : generated artifacts
- `README.md`, `README_KR.md` : docs
- `LICENSE` : Apache 2.0
