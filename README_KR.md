# HR-Mel (44.1 kHz) 한국어

[English README](README.md) · [License](LICENSE)

HR-Mel은 멜 분포는 유지하면서 고역을 더 촘촘히 보는 3밴드 멜 표현입니다. 44.1 kHz 고정 파라미터로 HR-Mel을 추출하고, STFT / Mel / Log-Mel 대비 상대 복원 오차를 계산하는 최소 코드를 포함합니다.

## 스펙
- 입력: `playlist.mp3` (mono 강제, `sr=44,100`)
- STFT: `n_fft=2048`, `hop_length=441`(≈10 ms), `win_length=2048`
- 멜 상한: `fmax=20,000 Hz`
- HR-Mel 밴드: `0–1.5 kHz` 40 bin log1p / `1.5–6 kHz` 32 bin log1p / `6–20 kHz` 24 bin sqrt(log1p) → 총 96 bin

## 실행
```bash
cd HR-mel
NUMBA_CACHE_DIR=/tmp python3 generate_mel_variants.py      # HR-Mel 생성
NUMBA_CACHE_DIR=/tmp python3 analyze_features.py           # STFT/Mel/Log-Mel/HR-Mel 비교
```

## 출력물
- `output/hr_mel.npz` : HR-Mel 인코딩 + 메타데이터
- `output/summary.json` : 입력/형상/FFT 파라미터 요약
- `output/analysis.json` : STFT 대비 상대 복원 오차 (Frobenius), 표현별 bin/프레임

## 비교 결과 (`playlist.mp3` 기준)
기본 비교 (80-bin 멜 대비)
| 표현 | Bins | STFT 상대 오차↓ | 메모 |
| --- | --- | --- | --- |
| STFT | 1025 | 0.000 | 기준 |
| Mel (linear) | 80 | 0.461 | 기본 멜 파워 |
| Log-Mel | 80 | 0.461 | log1p 가역 |

동일 차원(96 bin) 비교
| 표현 | Bins | STFT 상대 오차↓ | 메모 |
| --- | --- | --- | --- |
| Mel-96 | 96 | 0.410 | 96-bin 멜 파워 |
| Log-Mel-96 | 96 | 0.410 | log1p on 96-bin 멜 |
| HR-Mel | 96 | 0.304 | 40/32/24 + 상단 sqrt-log |

압축 크기 (np.savez_compressed, bytes)
| 표현 | Bins | 압축 크기 | 비고 |
| --- | --- | --- | --- |
| STFT | 1025 | 55.6 MB | 파워 스펙트럼 |
| Mel | 80 | 4.43 MB | |
| Log-Mel | 80 | 4.41 MB | |
| Mel-96 | 96 | 5.32 MB | |
| Log-Mel-96 | 96 | 5.29 MB | |
| HR-Mel | 96 | 5.26 MB | sqrt-log 상단 밴드 |

오차 계산: STFT 파워 → 표현 → 유사역행렬로 STFT 공간 복원 → Frobenius 상대 오차. HR-Mel은 동일 96 bin 대비 오차를 크게 낮추면서 압축 크기는 거의 동일합니다.

## 파일 구성
- `generate_mel_variants.py` : HR-Mel 추출 (입력 `playlist.mp3` → `output/`)
- `analyze_features.py` : STFT/Mel/Log-Mel/HR-Mel 비교 리포트 생성
- `output/` : 실행 결과 저장 위치
- `README.md`, `README_KR.md` : 문서
- `LICENSE` : Apache 2.0
