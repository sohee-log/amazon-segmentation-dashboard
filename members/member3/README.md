# 불만 고객 조기 경보 모델
### Early Warning System for Negative Feedback

Amazon 상품 리뷰의 감정과 부정 키워드를 분석하여  
불만 가능성이 높은 고객 리뷰를 조기에 탐지하고 대응 우선순위를 제안하는 모델입니다.

---

## 1. 분석 주제 / 문제 정의

온라인 쇼핑몰에는 많은 상품 리뷰가 등록되지만, 기업이 모든 리뷰를 직접 확인하기는 어렵습니다.

특히 별점만으로 고객 불만을 판단할 경우 다음과 같은 문제가 있습니다.

- 별점은 높지만 리뷰 본문에 불만이 포함된 경우
- 제품 고장, 환불, 교환 등 즉각적인 대응이 필요한 리뷰
- 강한 부정 감정이 표현됐지만 별점만으로는 확인하기 어려운 경우

따라서 본 분석에서는 리뷰 텍스트의 감정과 부정 키워드를 분석하여  
고객 불만 위험도를 **Normal / Risk / Critical** 3단계로 분류하는  
불만 고객 조기 경보 모델을 구현했습니다.

---

## 2. 사용 데이터 설명

### 데이터셋

Kaggle의 Amazon Sales Dataset을 사용했습니다.

### 주요 변수

| 변수 | 설명 |
|---|---|
| `product_name` | 상품명 |
| `category` | 상품 카테고리 |
| `rating` | 상품 평점 |
| `rating_count` | 리뷰 수 |
| `review_title` | 리뷰 제목 |
| `review_content` | 리뷰 본문 |
| `discounted_price` | 할인 가격 |
| `actual_price` | 정가 |
| `discount_percentage` | 할인율 |

### 분석을 위해 생성한 파생 변수

| 파생 변수 | 설명 |
|---|---|
| `review_length` | 리뷰의 단어 수 |
| `sentiment_score` | 리뷰의 감정 점수 |
| `emotion_strength` | 감정 점수 절댓값 |
| `negative_sentiment_strength` | 부정 감정의 강도 |
| `negative_keyword_count` | 리뷰에 포함된 부정 키워드 수 |
| `negative_keyword_score` | 정규화한 부정 키워드 점수 |
| `review_length_norm` | 정규화한 리뷰 길이 |
| `early_warning_score` | 고객 불만 조기 경보 점수 |
| `warning_level` | Normal / Risk / Critical 경보 단계 |

---

## 3. 분석 방법

### 3-1. 리뷰 텍스트 수치화

TextBlob을 이용해 리뷰별 감정 점수를 계산했습니다.

- `-1`에 가까울수록 매우 부정적
- `0`에 가까울수록 중립적
- `1`에 가까울수록 매우 긍정적

부정적인 감정만 위험 점수에 반영하기 위해  
감정 점수가 음수인 경우에만 `negative_sentiment_strength`를 계산했습니다.

### 3-2. 부정 키워드 자동 추출

기존에는 `bad`, `broken`, `refund` 등의 부정 키워드를 사람이 직접 지정했습니다.

이를 개선하기 위해 저평점 및 부정 감정 리뷰를 대상으로  
**TF-IDF를 적용하여 주요 부정 키워드를 자동 추출**했습니다.

TF-IDF 분석 과정에서 `product`, `good`, `quality`처럼  
불만을 직접 의미하지 않는 일반 단어는 사용자 정의 불용어와 후처리를 통해 제거했습니다.

### 3-3. 조기 경보 점수 계산

다음 세 가지 요소를 가중합하여 `Early Warning Score`를 계산했습니다.

```text
Early Warning Score
= 0.55 × negative_sentiment_strength
+ 0.30 × negative_keyword_score
+ 0.15 × review_length_norm
