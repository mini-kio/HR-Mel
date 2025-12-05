# HR-Mel (44.1 kHz) 한국어

[English README](README.md) · [License](LICENSE)

HR-Mel은 멜 분포는 유지하면서 고역을 더 촘촘히 보는 3밴드 멜 표현입니다. 44.1 kHz 고정 파라미터로 HR-Mel을 추출하고, STFT / Mel / Log-Mel 대비 상대 복원 오차를 계산하는 최소 코드를 포함합니다.

## 스펙
- 입력: `playlist.mp3` (mono 강제, 기본 `sr=44,100`)
- STFT: `n_fft=2048`, `hop_length=441`(≈10 ms), `win_length=2048`
- 멜 상한: `fmax=20,000 Hz`
- HR-Mel 밴드: `0–1.5 kHz` 40 bin log1p / `1.5–6 kHz` 32 bin log1p / `6–20 kHz` 24 bin sqrt(log1p) → 총 96 bin

### Motivation (왜 이걸 쓰나)
- 기본 log-mel은 고역 정보가 쉽게 사라진다. HR-Mel은 차원을 크게 늘리지 않으면서 고역 해상도를 높여 STFT 복원 오차를 줄이는 것을 목표로 한다.
- 44.1 kHz 음악 데이터의 front-end(오디오 코덱, music LM 등)로 사용할 때 어택/에어 손실을 줄이는 용도.

### Limitations (제한)
- 기본값은 44.1 kHz 고정(`n_fft=2048`, `hop=441`, `win=2048`)이며 다른 세팅은 검증되지 않았다.
- `fmax`는 나이퀴스트로 클리핑된다. 추출/분석 시 동일 파라미터를 맞추는 것을 권장.

## 데이터셋
- **파일 수**: 12 트랙
- **길이**: 트랙당 평균 ~2–3분
- **출처**: 개인 수집, 다양한 스타일 (보컬/악기 혼합)
- **샘플레이트**: 44.1 kHz
- **파일 형식**: WAV 또는 MP3

## 실행
```bash
cd HR-mel
NUMBA_CACHE_DIR=/tmp python -m src.generate_mel_variants \
  --input playlist.mp3 --output-dir output --sr 44100 --fmax 20000

NUMBA_CACHE_DIR=/tmp python -m src.analyze_features \
  --input playlist.mp3 --output-dir output --sr 44100 --fmax 20000
```

Python API 예시:
```python
from src.hr_mel import hr_mel
encoded, meta = hr_mel(y, sr=44_100)  # y: mono waveform
```

## 출력물
- `output/hr_mel.npz` : HR-Mel 인코딩 + 메타데이터
- `output/summary.json` : 입력/형상/FFT 파라미터 요약
- `output/analysis.json` : STFT 대비 상대 복원 오차 (Frobenius), 표현별 bin/프레임

`analysis.json` 구조:
- 최상단: `input_sr`, `duration_sec`, `frames`, `n_fft`, `hop_length`, `win_length`, `fmax`
- `representations`: 각 표현별 `bins`, `frames`, `relative_recon_error`, `bytes_compressed`, `note`

## 실험 결과

### 전체 성능 비교 (12곡, 평균 ± 표준편차)
| 표현 | Bins | 상대 오차 ↓ | 메모 |
| --- | --- | --- | --- |
| STFT | 1025 | 0.000 | 기준 |
| Mel-80 | 80 | 0.4438 ± 0.0404 | 80-bin 파워 멜 |
| Log-Mel-80 | 80 | 0.4438 ± 0.0404 | log1p on 80-bin 멜 |
| Mel-96 | 96 | 0.3810 ± 0.0395 | 96-bin 파워 멜 |
| Log-Mel-96 | 96 | 0.3810 ± 0.0395 | log1p on 96-bin 멜 |
| **HR-Mel (40/32/24)** | 96 | **0.2845 ± 0.0410** | 40/32/24 빈 + sqrt-log 상단 밴드 |

### 대역별 분석 (주파수 범위: <1.5 kHz / 1.5–6 kHz / >6 kHz)

**40/32/24 구성 (기본, 권장)**
| 대역 | Mel-96 오차 | HR-Mel 오차 | 차이 | 개선율 |
| --- | --- | --- | --- | --- |
| 저역 (<1.5 kHz) | 0.3737 ± 0.0391 | **0.2729 ± 0.0345** | +0.1008 | **~26.7%** |
| 중역 (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| 고역 (>6 kHz) | 0.7186 ± 0.0548 | 0.7372 ± 0.0557 | -0.0186 | -2.6% |

**32/32/32 구성 (균형 빈 배치)**
| 대역 | Mel-96 오차 | HR-Mel 오차 | 차이 | 개선율 |
| --- | --- | --- | --- | --- |
| 저역 (<1.5 kHz) | 0.3737 ± 0.0391 | 0.4007 ± 0.0394 | -0.0271 | -7.4% |
| 중역 (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| 고역 (>6 kHz) | 0.7186 ± 0.0548 | 0.7159 ± 0.0509 | +0.0027 | +0.3% |

**24/32/40 구성 (고역 집중)**
| 대역 | Mel-96 오차 | HR-Mel 오차 | 차이 | 개선율 |
| --- | --- | --- | --- | --- |
| 저역 (<1.5 kHz) | 0.3737 ± 0.0391 | 0.4840 ± 0.0381 | -0.1104 | -30.5% |
| 중역 (1.5–6 kHz) | 0.6701 ± 0.0372 | 0.6780 ± 0.0419 | -0.0079 | -1.2% |
| 고역 (>6 kHz) | 0.7186 ± 0.0548 | **0.6958 ± 0.0429** | +0.0228 | **+3.1%** |

### 핵심 발견사항
- **40/32/24 구성**이 전체적으로 가장 우수한 성능을 보이며 저역에서 약 27% 개선
- HR-Mel의 주요 이득은 저주파 대역(<1.5 kHz)에서 발생
- 중역 및 고역은 표준 Mel-96 대비 소폭 성능 저하
- 균형 잡힌(32/32/32) 또는 고역 집중(24/32/40) 구성은 저역 성능을 희생
- 저역 이득을 유지하면서 고역 성능을 개선하려면 상단 밴드 압축 함수 조정이나 fmax 증가 고려 필요

오차 계산: STFT 파워 → 표현 → 유사역행렬로 STFT 공간 복원 → Frobenius 상대 오차.

## 프로젝트 구조
```
HR-Mel/
├── src/                          # 소스 코드
│   ├── hr_mel.py                # 핵심 HR-Mel 구현
│   ├── generate_mel_variants.py # HR-Mel 특징 추출기
│   ├── analyze_features.py      # STFT/Mel/HR-Mel 비교 분석
│   └── utils/                   # 유틸리티 모듈
│       ├── mel_utils.py         # Mel 처리 유틸리티
│       └── analysis_utils.py    # 분석 헬퍼 함수
├── output/                      # 생성된 결과물 (분석 결과, 인코딩)
├── README.md                    # 영문 문서
├── README_KR.md                 # 한국어 문서 (이 파일)
└── LICENSE                      # Apache 2.0
```
