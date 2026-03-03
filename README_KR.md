# HR-Mel (44.1 kHz) 한국어

[English README](README.md) · [논문 (Zenodo)](https://zenodo.org/records/18831533) · [License](LICENSE)

HR-Mel은 44.1 kHz 음악 신호를 위한 고정 3밴드 멜 프론트엔드로, 저역에 더 많은 용량을 할당하면서 밴드별 비선형 압축을 적용합니다. GTZAN 벤치마크(999곡, 10장르)에서 표준 96-bin log-mel 대비 **53.8% 낮은 복원 오차**를 달성했습니다.

## 설정

| 설정 | 밴드 | Bins | 용도 |
|---|---|---|---|
| **v1** (초기) | 0–1.5k / 1.5–6k / 6–20k | 40/32/24 | 초기 설계 |
| **v2** (최적) | 0–1.2k / 1.2–5k / 5–20k | 48/32/16 | 디자인 스페이스 탐색 최적 |

- STFT: `n_fft=2048`, `hop_length=441` (~10 ms), `win_length=2048`
- 압축: log1p (저/중역), sqrt(log1p) (고역)
- 총 96 bins

## 실험 결과

### 전체 성능

| 표현 | Bins | 12곡 (개인) | GTZAN (999곡) |
|---|---|---|---|
| Mel-80 | 80 | 0.4438 ± 0.040 | 0.4514 ± 0.058 |
| Mel-96 / Log-Mel-96 | 96 | 0.3810 ± 0.040 | 0.3747 ± 0.059 |
| HR-Mel v1 (40/32/24) | 96 | 0.2845 ± 0.041 | 0.3025 ± 0.056 |
| **HR-Mel v2 (48/32/16)** | **96** | **0.2041 ± 0.041** | **0.1730 ± 0.077** |

### Rate-Distortion (GTZAN, 스칼라 양자화)

| Bits | Log-Mel-96 | HR-Mel v2 |
|---|---|---|
| 4 | 0.4026 | **0.2532** |
| 6 | 0.3764 | **0.1790** |
| 8 | 0.3748 | **0.1733** |

### 장르별 (GTZAN, HR-Mel v2 vs Mel-96)

| 장르 | Mel-96 | HR-Mel v2 | 개선율 |
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

### 장르 분류 (SVM, 3-fold CV)

| 특성 | 정확도 |
|---|---|
| Log-Mel-96 | 0.647 ± 0.010 |
| **HR-Mel v2** | **0.660 ± 0.029** |

### 대역별 (GTZAN, HR-Mel v2)

| 대역 | Mel-96 | HR-Mel v2 | 개선율 |
|---|---|---|---|
| 저역 (<1.2 kHz) | 0.3645 | **0.1227** | **+66.3%** |
| 중역 (1.2–5 kHz) | 0.6209 | 0.6133 | +1.2% |
| 고역 (>5 kHz) | 0.7254 | 0.7936 | -9.4% |

## 실행

```bash
# 기본 분석 (파일 또는 디렉토리)
python -m src.analyze_features --input <오디오_파일_또는_디렉토리> --output-dir output

# HR-Mel 추출
python -m src.generate_mel_variants --input <오디오_파일> --output-dir output
```

```python
from src.hr_mel import hr_mel
encoded, meta = hr_mel(y, sr=44_100)  # y: 모노 파형
```

## 연구 스크립트

```bash
# 패러다임 연구 (디자인 스페이스, RD, 멀티레이트, 심리음향, 적응형)
python research/paradigm_v1/run_research.py --input <오디오_디렉토리> --output-dir research/paradigm_v1/results

# GTZAN 검증 (Data/genres_original/에 GTZAN 데이터셋 필요)
python research/gtzan_validation/run_gtzan_validation.py --data-dir Data --output-dir research/gtzan_validation/results
```

## 프로젝트 구조

```
HR-Mel/
├── src/                                    # 핵심 구현
│   ├── hr_mel.py                          # HR-Mel 기저/인코딩/디코딩
│   ├── generate_mel_variants.py           # 특성 추출 CLI
│   ├── analyze_features.py                # 표현 비교 분석
│   └── utils/
│       ├── mel_utils.py                   # Mel 유틸리티
│       └── analysis_utils.py              # 오차/크기 유틸리티
├── research/
│   ├── paradigm_v1/
│   │   ├── run_research.py                # 6단계 패러다임 연구 러너
│   │   ├── hrmel_spec_prototype.py        # 프로파일 기반 API 프로토타입
│   │   └── README.md
│   └── gtzan_validation/
│       └── run_gtzan_validation.py        # GTZAN 벤치마크 테스트
├── main.tex                               # 논문 (LaTeX)
├── README.md                              # 영문 문서
├── README_KR.md                           # 한국어 문서 (이 파일)
└── LICENSE                                # Apache 2.0
```

## GTZAN 참고

GTZAN은 네이티브 22.05 kHz입니다. 검증 스크립트는 44.1 kHz로 업샘플합니다. [Sturm 2013](https://arxiv.org/abs/1306.1461) 권고에 따라 알려진 결함 파일(jazz.00054.wav)을 제외합니다. 분류는 계층화 랜덤 분할(아티스트 필터 미적용)을 사용하며, 상대 비교는 유효합니다.
