"""
모델 2 대시보드 탭: 리뷰 텍스트 -> 신뢰도 점수

기존 app.py의 tab2 자리에 아래 render()를 연결하면 됩니다.

    from model2_trust_tab import render as render_trust_tab
    ...
    tab1, tab2, tab3 = st.tabs(["세그멘테이션", "리뷰 신뢰도 점수", "..."])
    with tab2:
        render_trust_tab()

먼저 `python export_trust_model.py`를 실행해서
models/trust_pipeline.joblib, models/trust_meta.json을 생성해 두어야 합니다.
"""
import json
import os

import joblib
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

MODEL_PATH = "models/trust_pipeline.joblib"
META_PATH = "models/trust_meta.json"
SAMPLE_PATH = "data/review_trust_samples.csv"

POS_KW = {"great", "excellent", "amazing", "perfect", "love", "best",
          "fantastic", "outstanding", "wonderful", "superb", "recommend"}
NEG_KW = {"bad", "terrible", "awful", "horrible", "worst", "broken",
          "waste", "disappointed", "poor", "defective", "scam", "fake"}


@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta


@st.cache_data
def load_samples():
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)
    return None


@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()


def score_label(score: float):
    if score >= 75:
        return "⭐ 고신뢰도 리뷰", "#2ecc71"
    elif score >= 50:
        return "✅ 보통 신뢰도 리뷰", "#3498db"
    elif score >= 25:
        return "⚠️ 낮은 신뢰도 리뷰", "#e67e22"
    return "🚨 위험 리뷰", "#e74c3c"


def gauge_chart(score: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "점", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#fdecea"},
                {"range": [25, 50], "color": "#fdf3e3"},
                {"range": [50, 75], "color": "#eaf3fc"},
                {"range": [75, 100], "color": "#eafaf1"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def render():
    st.subheader("📝 리뷰 신뢰도 점수 예측 (Model 2)")
    st.caption(
        "리뷰 텍스트를 감정 방향/강도(VADER), 리뷰 길이, 어휘 다양성, 긍부정 "
        "키워드 수로 구성한 자체 정의 '신뢰도 점수(0~100)'로 환산합니다. "
        "실제 허위 리뷰 판별 라벨은 존재하지 않으므로 참고용 보조 지표입니다."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "모델 파일이 없습니다. 먼저 터미널에서 아래 명령을 실행해 주세요:\n\n"
            "`python export_trust_model.py`"
        )
        return

    pipeline, meta = load_model()
    sia = load_sia()

    example_reviews = {
        "직접 입력": "",
        "고신뢰 긍정 예시": (
            "This product is absolutely amazing! I have been using it for 2 months "
            "and it works perfectly. The build quality is excellent and the value "
            "for money is outstanding. Highly recommend to anyone!"
        ),
        "고신뢰 부정 예시": (
            "I purchased this product 3 weeks ago and it has already stopped working. "
            "The build quality is extremely poor and customer service did not help at all."
        ),
        "스팸 의심 예시": (
            "AMAZING AMAZING BUY NOW!!! BEST PRODUCT EVER!!!! OK OK OK GOOD GOOD GOOD!!!"
        ),
    }

    choice = st.selectbox("예시 리뷰로 테스트해보기", list(example_reviews.keys()))
    review_text = st.text_area(
        "리뷰 텍스트 입력",
        value=example_reviews[choice],
        height=140,
        placeholder="분석할 리뷰 텍스트를 영어로 입력하세요...",
    )

    if st.button("신뢰도 점수 분석하기", type="primary"):
        if not review_text.strip():
            st.warning("리뷰 텍스트를 입력해 주세요.")
            return

        score = float(pipeline.predict([review_text])[0])
        score = max(0.0, min(100.0, score))
        label, color = score_label(score)

        vader = sia.polarity_scores(review_text)
        compound = vader["compound"]
        words = review_text.split()
        pos_hits = [w for w in words if w.lower().strip(".,!?") in POS_KW]
        neg_hits = [w for w in words if w.lower().strip(".,!?") in NEG_KW]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(gauge_chart(score, color), use_container_width=True)
            st.markdown(f"### {label}")
        with col2:
            st.metric("리뷰 길이", f"{len(words)}단어")
            st.metric("VADER 감성 점수", f"{compound:+.3f}")
            st.write("긍정 키워드:", ", ".join(pos_hits) if pos_hits else "없음")
            st.write("부정 키워드:", ", ".join(neg_hits) if neg_hits else "없음")

        with st.expander("이 점수는 어떻게 계산되나요?"):
            st.write(
                "리뷰 텍스트를 TF-IDF로 벡터화한 뒤, Ridge 회귀로 감정 방향·강도, "
                "어휘 다양성, 리뷰 길이, 긍/부정 키워드 수를 조합해 만든 자체 "
                "정의 신뢰도 점수를 예측합니다. 아래는 각 요소가 신뢰도 점수를 "
                "만들 때 사용된 가중치(rating과의 상관계수 기반)입니다."
            )
            st.json(meta.get("weights", {}))
            st.caption(meta.get("note", ""))

    st.divider()

    samples = load_samples()
    if samples is not None and "trust_score" in samples.columns:
        st.markdown("#### 데이터셋 내 신뢰도 점수 분포")
        st.bar_chart(samples["trust_score"].value_counts(bins=10).sort_index())

        show_cols = [c for c in
                     ["product_name", "rating", "trust_score", "review_content"]
                     if c in samples.columns]
        st.markdown("**신뢰도 상위 5개 리뷰**")
        st.dataframe(samples.sort_values("trust_score", ascending=False).head(5)[show_cols])
        st.markdown("**신뢰도 하위 5개 리뷰**")
        st.dataframe(samples.sort_values("trust_score", ascending=True).head(5)[show_cols])