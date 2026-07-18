# Amazon Review Intelligence Dashboard

> 아마존 상품 리뷰 데이터를 **세 개의 시선**으로 읽는 Streamlit 대시보드.
> 상품을 **세그먼트로 나누고**(Tab 1), 리뷰의 **신뢰도를 점수화**하고(Tab 2), 불만을 **조기에 경보**합니다(Tab 3).

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20Ridge-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-3D%20Scatter-3F4F75?logo=plotly&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

![대시보드 화면](docs/dashboard-hero.png)

---

## 프로젝트 소개

**별점 하나로는 상품도, 리뷰도, 고객도 제대로 읽히지 않습니다.** 별점은 높은데 텍스트는 불만인 상품이 있고,
같은 5점이라도 근거가 뚜렷한 리뷰와 성의 없는 한 줄이 섞여 있으며, 별점이 후해도 본문엔 고장·환불 요청이 담겨 있습니다.

DACOS Team 2 는 같은 [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)(상품 1,465개)을 놓고
**별점이 놓치는 신호를 세 갈래로** 파고들었고, 그 결과를 하나의 대시보드에 통일된 디자인으로 모았습니다.

| 탭 | 담당 | 무엇을 하나 | 핵심 산출물 |
|---|---|---|---|
| **① Product Segmentation** | sohee | 가격·감정·별점-감정 괴리·리뷰 수로 상품을 3개 세그먼트로 군집화 | K-Means `k=3` · PCA 3D 맵 |
| **② Review Trust Score** | hjm00502-eng | 리뷰 텍스트를 6개 피처로 환산해 0~100 신뢰도 점수 예측 | TF-IDF + Ridge · 신뢰도 게이지 |
| **③ Early Warning** | heeseon25 | 감정·부정 키워드·길이로 불만 위험도를 3단계 경보로 분류 | Early Warning Score · 조기 경보 시뮬레이터 |

> Tab 2·3 은 허위 리뷰나 불만의 **정답 라벨이 없는 자체 정의 지표**이므로 참고용 보조 신호입니다. 자세한 한계는 각 섹션에 명시했습니다.

---

## 실행 방법

```bash
git clone https://github.com/sohee-log/amazon-segmentation-dashboard.git
cd amazon-segmentation-dashboard

pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 됩니다.
분석 결과 CSV·학습된 모델이 저장소에 포함되어 있어 **별도 데이터 준비 없이 세 탭 모두 바로 실행**됩니다.

> Tab 2 모델(`models/trust_pipeline.joblib`)이 없다면 `python export_trust_model.py` 로 먼저 생성하세요.

---

## 폴더 구조

```text
amazon-segmentation-dashboard/
├── app.py                              # Streamlit 진입점 · 공통 디자인 · Tab 1
├── segment_naming.py                   # Tab 1 — 세그먼트 이름을 군집 통계에서 유도
├── model2_trust_tab.py                 # Tab 2 — 리뷰 신뢰도 점수
├── model3_alert_tab.py                 # Tab 3 — 불만 고객 조기 경보
├── export_trust_model.py               # Tab 2 모델 학습 → models/ 생성
├── data/
│   ├── user_segments.csv               # Tab 1 분석 결과 (대시보드 메인 데이터)
│   ├── sample_user_segments.csv        # 폴백용 샘플
│   ├── model_metrics.csv               # Tab 1 — k=2~6 KMeans/GMM 성능 비교
│   └── review_trust_samples.csv        # Tab 2 — 샘플 리뷰 + 신뢰도 점수
├── models/                             # Tab 2 학습된 파이프라인·메타 (joblib/json)
├── notebooks/
│   └── 01_product_segmentation.ipynb   # Tab 1 — 전처리 → 모델링 → 검증 전 과정
├── scripts/
│   └── export_segments.py              # Tab 1 노트북 로직 재현 → data/*.csv
├── members/                            # 팀원별 원본 분석 문서
│   ├── sohee/ · member2/ · member3/     #   각 담당의 README
│   └── model3                          #   Tab 3 원본 시각화(HTML) 아카이브
├── .streamlit/config.toml              # 테마 설정
├── requirements.txt                    # 대시보드 실행용
└── requirements-pipeline.txt           # 데이터·모델 재생성용
```

---

## ① Product Segmentation — 상품 세그먼테이션

별점과 리뷰 텍스트 감정의 **괴리(rating gap)** 를 핵심 피처로 삼아 상품 1,465개를 세 갈래로 나눕니다.

| 세그먼트 | 상품 수 | 평균가 | 괴리 | 감정 | 운영 액션 |
|---|---:|---:|---:|---:|---|
| 관심 필요 상품 | 637 | 1,342 | **0.229** (최대) | 0.198 (최저) | 품질 · 배송 · 상세페이지 우선 점검 |
| 프리미엄 베스트셀러 | 343 | 17,722 | 0.219 | 0.221 | 할인보다 브랜드 · 로열티 전략 |
| 가성비 호평 상품 | 485 | 2,152 | **0.104** (최소) | 0.394 (최고) | 긍정 리뷰 노출 강화로 전환 상승 |

별점이 후한 쪽은 오히려 **저가·대량 리뷰 상품**이고, 별점이 가장 박한 중가군이 텍스트 감정은 가장 좋아 괴리가 가장 작습니다.
전체의 75.8%가 별점 4.0 이상이라 별점만으로는 변별이 어렵습니다.

### 분석 파이프라인

**1. 피처 엔지니어링**

| 피처 | 정의 |
|---|---|
| `discount_percentage` | 할인율 |
| `sentiment_score` | TextBlob 기반 리뷰 텍스트 감정 극성 (-1 ~ 1) |
| `rating_gap` | `rating/5 - (sentiment+1)/2` — **별점과 텍스트 감정의 괴리** |
| `log_actual_price` | 정가 로그 변환 (가격 왜도 보정) |
| `log_rating_count` | 리뷰 수 로그 변환 (인기도 왜도 보정) |

**2. 모델 선정** — `k=2~6` 구간에서 K-Means와 GMM을 실루엣·Davies-Bouldin·Calinski-Harabasz 세 지표로 비교했습니다.
전 구간에서 K-Means가 우세했고, 해석 가능성을 우선해 **K-Means `k=3`** 을 채택했습니다.

| 모델 | k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---:|---:|---:|---:|
| **KMeans** | **3** | **0.181** | **1.686** | **347.6** |
| GMM | 3 | 0.134 | 1.999 | 207.6 |

> k=2가 실루엣 점수는 더 높지만(0.204), 2개 군집은 "좋음/나쁨" 이분법이라 운영 액션으로 연결되지 않아 k=3을 선택했습니다.
> 전체 비교표는 [`data/model_metrics.csv`](data/model_metrics.csv) 에 있습니다.

**3. 세그먼트 이름 부여** — K-Means 클러스터 번호는 초기화에 따라 임의로 붙으므로, 이름을 번호에 고정하지 않고
[`segment_naming.py`](segment_naming.py)에서 **군집 통계로 유도**합니다. (① 평균가 최고 → 프리미엄, ② 나머지 중 괴리 큰 쪽 → 관심 필요, ③ 나머지 → 가성비 호평)

**4. 시각화** — 5개 피처를 **PCA 3차원으로 압축**(총 분산 77.5% — PC1 36.6% / PC2 21.4% / PC3 19.5%)해 회전·확대 가능한 3D 산점도로 렌더링합니다.
포인트를 클릭하면 아래 패널에 상품 상세가 연결됩니다.

---

## ② Review Trust Score — 리뷰 신뢰도 점수

리뷰 텍스트만으로 **0~100 신뢰도 점수**를 예측합니다. 같은 별점이라도 근거가 뚜렷한 리뷰와 과장·스팸을 구분하려는 시도입니다.

**피처 6종 → 가중합 → 점수.** rating 과의 상관계수로 가중치를 정하고, 리뷰 원문으로 이 점수를 재현하도록 학습합니다.

| 피처 | 방향 | 의미 |
|---|:--:|---|
| 감정 강도 (`emotion_strength`) | ↑ | VADER compound 의 절댓값 — 감정이 뚜렷할수록 |
| VADER 감성 (`vader_compound`) | ↑ | 감정의 방향과 세기 |
| 긍정 키워드 수 (`pos_keyword_count`) | ↑ | great·excellent 등 |
| 리뷰 길이 (`review_length`) | ↑ | 단어 수 |
| 부정 키워드 수 (`neg_keyword_count`) | ↓ | bad·broken 등 |
| 어휘 다양성 (`word_diversity`) | ↓ | 지나치게 단조로운 어휘는 감점 |

- **모델**: `TF-IDF(1~2gram, max 800) + Ridge(α=0.5)` 파이프라인이 리뷰 원문 → 신뢰도 점수를 예측
- **재현 성능**: **R² ≈ 0.42 · MAE ≈ 7.9** (텍스트로부터 이 지표를 얼마나 잘 재현하는지)
- 대시보드에서 리뷰를 입력하면 신뢰도 게이지, VADER 감성, 긍·부정 키워드, 피처 가중치 차트를 함께 보여줍니다.

> ⚠️ `trust_score` 는 **정답 라벨이 없는 자체 정의 지표**입니다. R²·MAE 는 리뷰의 실제 진위를 맞히는 정확도가 아니라, 정의한 점수를 텍스트로 재현하는 성능입니다.
> 학습·아티팩트 생성 로직은 [`export_trust_model.py`](export_trust_model.py) 참고.

---

## ③ Early Warning — 불만 고객 조기 경보

별점이 높아도 본문에 불만이 담긴 리뷰를 **조기에 골라내** 대응 우선순위를 제안합니다.
리뷰를 **Normal / Risk / Critical** 세 단계로 분류합니다.

**왜 별점만으로는 부족한가** — ① 별점은 높은데 본문은 불만, ② 고장·환불처럼 즉시 대응이 필요한 리뷰, ③ 별점만으로는 안 보이는 강한 부정 감정.

**분석 방법 3단계**

1. **감정 수치화** — TextBlob 으로 리뷰별 감정 극성(-1~1)을 계산하고, 음수일 때만 부정 감정 강도로 반영
2. **부정 키워드 자동 추출** — 저평점·부정 리뷰에 **TF-IDF** 를 적용해 주요 부정 키워드를 자동 추출 (일반어는 불용어 처리로 제거)
3. **조기 경보 점수 계산** — 세 요소의 가중합

```text
Early Warning Score
= 0.55 × negative_sentiment_strength   (부정 감정 강도)
+ 0.30 × negative_keyword_score        (min(부정 키워드 수 / 10, 1))
+ 0.15 × review_length_norm            (min(단어 수 / 120, 1))

Critical : 점수 >= 0.45      Risk : 점수 >= 0.25      Normal : 그 외
```

- 대시보드에서 리뷰를 입력하면 **위험도 미터, 신호별 분석(감정·키워드·길이), 근거 테이블, 감지된 불만 표현, 단계별 AI 인사이트·권장 대응**을 보여줍니다.
- 자동 추출한 부정 키워드 TOP 15 와 실제 경보 리뷰 예시도 함께 제공합니다.

> 감정 점수는 긍/부정 단어 사전 기반 간이 계산이며, 불만·허위 리뷰의 정답 라벨이 없으므로 참고용 보조 지표입니다.
> 원본 분석 노트는 [`members/member3/README.md`](members/member3/), 원본 시각화는 [`members/model3`](members/model3) 에 있습니다.

---

## 데이터 · 모델 재생성 (선택)

저장소의 CSV·모델을 직접 다시 만들고 싶다면:

```bash
pip install -r requirements-pipeline.txt

python scripts/export_segments.py     # Tab 1 — user_segments.csv, model_metrics.csv 등
python export_trust_model.py          # Tab 2 — trust_pipeline.joblib, review_trust_samples.csv
```

Kaggle 데이터셋을 자동 다운로드하므로 [Kaggle API 인증](https://www.kaggle.com/docs/api#authentication)이 필요합니다.

---

## 디자인

세 탭 모두 **하나의 에디토리얼 매거진 톤**을 공유합니다. off-white 캔버스(`#f5f5f5`)에 웜 니어블랙 잉크(`#0c0a09`),
채도 높은 CTA 색은 두지 않고, 파스텔 그라디언트 orb 가 유일한 "색" 순간입니다.
서체는 Inter + Noto Sans KR 로 통일하고, 위계는 웨이트와 크기로만 만듭니다.

### 차트 색을 따로 파생시킨 이유

브랜드의 파스텔 orb(mint · peach · lavender · sky · rose)는 **분위기 전용**이라 데이터 마크로 쓰면 off-white 위에서 대비가 무너집니다.
그래서 **같은 hue 계보는 유지하되 명도만 낮춘** 데이터 전용 팔레트를 파생시켰고, 세 탭이 일관된 규칙을 따릅니다.

| 탭 | 인코딩 대상 | 색 규칙 |
|---|---|---|
| ① 세그먼테이션 | 세그먼트(범주) | peach `#b45309` · sky `#2563a8` · mint `#0d9488` (범주형, all-pairs 분리) |
| ② 신뢰도 | 신뢰도(순서형) | sky 단일 hue 명도 램프 — 어두울수록 높은 신뢰도 |
| ③ 조기 경보 | 심각도(순서형) | warm 단일 hue 명도 램프 — 색약에서도 명도 차이는 남음 |

색만으로 정보를 나르지 않도록 범례·이름·태그를 항상 함께 띄우고, 세그먼트 배지는 **점만 색이고 글자는 잉크**를 유지하며,
중심점은 색이 아니라 형태(잉크 다이아몬드)로 구분합니다.

---

## 팀

DACOS Team 2 토이 프로젝트입니다. 각 팀원의 원본 작업은 [`members/`](members/) 폴더에 있습니다.

| 담당 | 파트 | 원본 문서 |
|---|---|---|
| sohee | ① 상품 세그먼테이션 · 대시보드 통합 | [`members/sohee/`](members/sohee/) |
| hjm00502-eng | ② 리뷰 신뢰도 점수 | [`members/member2/`](members/member2/) |
| heeseon25 | ③ 불만 고객 조기 경보 | [`members/member3/`](members/member3/) · [`members/model3`](members/model3) |

협업 규칙과 시작 가이드는 [`members/README.md`](members/README.md) 를 참고하세요.

---

## 라이선스

[MIT License](LICENSE)
