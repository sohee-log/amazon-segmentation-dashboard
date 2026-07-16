"""
모델 2 대시보드 탭: 리뷰 텍스트 -> 신뢰도 점수

app.py 의 tab2 에 아래처럼 연결한다.

    from model2_trust_tab import render as render_trust_tab
    ...
    with tab2:
        render_trust_tab()

먼저 `python export_trust_model.py`를 실행해서
models/trust_pipeline.joblib, models/trust_meta.json을 생성해 두어야 합니다.
"""
import json
from pathlib import Path

import joblib
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

# 상대 경로면 저장소 루트에서 실행할 때만 동작한다. app.py 와 같은 기준으로 맞춘다.
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "trust_pipeline.joblib"
META_PATH = APP_DIR / "models" / "trust_meta.json"
SAMPLE_PATH = APP_DIR / "data" / "review_trust_samples.csv"

# --- 디자인 토큰 (app.py 와 동일 계보) ---------------------------------------
INK = "#0c0a09"
BODY = "#4e4e4e"
HAIRLINE = "#e7e5e4"
SURFACE_CARD = "#ffffff"
PLOT_FONT = "Inter, Noto Sans KR, sans-serif"

# 신뢰도 점수(0~100)는 순서형 '크기'다. 서로 다른 4가지 색(초록/파랑/주황/빨강)은
# 무지개 인코딩이므로, 브랜드 sky 계보의 단일 hue 명도 램프로 표현한다.
# 어두울수록 높은 신뢰도. 검증 통과: 명도 단조·인접 ΔL·밝은 끝 대비 2.22:1·단일 hue.
TRUST_RAMP = ["#7dabd6", "#4f86c6", "#2f66a4", "#1a4a80"]

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
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    return None


@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()


def score_label(score: float) -> tuple[str, str]:
    """점수 구간의 이름과 램프 색. 색만으로 뜻을 나르지 않도록 이름을 항상 함께 쓴다."""
    if score >= 75:
        return "고신뢰도 리뷰", TRUST_RAMP[3]
    if score >= 50:
        return "보통 신뢰도 리뷰", TRUST_RAMP[2]
    if score >= 25:
        return "낮은 신뢰도 리뷰", TRUST_RAMP[1]
    return "위험 리뷰", TRUST_RAMP[0]


def section(label: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(
        f'<div class="section"><div class="section-label">{label}</div>'
        f"<h2>{title}</h2>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def gauge_chart(score: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 44, "color": INK, "family": PLOT_FONT}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": HAIRLINE,
                    "tickfont": {"size": 11, "color": BODY, "family": PLOT_FONT},
                },
                "bar": {"color": color, "thickness": 0.62},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                # 배경 밴드는 눈금 역할만 — 램프를 옅게 깔아 어두울수록 높은 신뢰도임을 보인다.
                "steps": [
                    {"range": [0, 25], "color": "rgba(125,171,214,0.10)"},
                    {"range": [25, 50], "color": "rgba(79,134,198,0.10)"},
                    {"range": [50, 75], "color": "rgba(47,102,164,0.10)"},
                    {"range": [75, 100], "color": "rgba(26,74,128,0.10)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        # 마진이 좁으면 오른쪽 끝 눈금 "100" 이 "10" 으로 잘린다.
        margin={"l": 44, "r": 44, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": BODY, "family": PLOT_FONT},
    )
    return fig


def distribution_chart(samples: pd.DataFrame) -> go.Figure:
    counts = samples["trust_score"].value_counts(bins=10).sort_index()
    labels = [f"{int(i.left)}–{int(i.right)}" for i in counts.index]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker={"color": TRUST_RAMP[2], "line": {"width": 0}},
        )
    )
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        font={"color": BODY, "family": PLOT_FONT},
        bargap=0.32,
        xaxis={"title": None, "tickfont": {"size": 11, "color": BODY}, "showgrid": False},
        yaxis={"title": None, "gridcolor": "rgba(12,10,9,0.10)", "tickfont": {"size": 11, "color": BODY}},
    )
    return fig


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">Review Trust Score</p>
            <h1>리뷰가 스스로<br>말하는 신뢰도</h1>
            <p>
                리뷰 텍스트를 감정 방향·강도(VADER), 길이, 어휘 다양성, 긍부정 키워드 수로 환산해
                0~100 신뢰도 점수를 예측합니다. 허위 리뷰 정답 라벨이 없으므로 참고용 보조 지표입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
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

    section("Input", "리뷰 분석", "예시를 고르거나 직접 붙여넣어 점수를 확인합니다.")
    choice = st.selectbox("예시 리뷰로 테스트해보기", list(example_reviews.keys()))
    review_text = st.text_area(
        "리뷰 텍스트 입력",
        value=example_reviews[choice],
        height=140,
        placeholder="분석할 리뷰 텍스트를 영어로 입력하세요...",
    )

    if st.button("신뢰도 점수 분석하기"):
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

        section("Result", "분석 결과")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.plotly_chart(gauge_chart(score, color), use_container_width=True)
            st.markdown(
                f'<div class="badge"><span class="dot" style="background:{color}"></span>{label}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            c1, c2 = st.columns(2)
            c1.metric("리뷰 길이", f"{len(words)}")
            c2.metric("VADER 감성", f"{compound:+.3f}")
            st.markdown(
                f'<div class="card"><p class="step">Keywords</p>'
                f'<p><strong>긍정</strong> · {", ".join(pos_hits) if pos_hits else "없음"}<br>'
                f'<strong>부정</strong> · {", ".join(neg_hits) if neg_hits else "없음"}</p></div>',
                unsafe_allow_html=True,
            )

        with st.expander("이 점수는 어떻게 계산되나요?"):
            st.write(
                "리뷰 텍스트를 TF-IDF로 벡터화한 뒤, Ridge 회귀로 감정 방향·강도, "
                "어휘 다양성, 리뷰 길이, 긍/부정 키워드 수를 조합해 만든 자체 "
                "정의 신뢰도 점수를 예측합니다. 아래는 각 요소가 신뢰도 점수를 "
                "만들 때 사용된 가중치(rating과의 상관계수 기반)입니다."
            )
            st.json(meta.get("weights", {}))
            st.caption(meta.get("note", ""))

    samples = load_samples()
    if samples is not None and "trust_score" in samples.columns:
        section("Distribution", "데이터셋 내 점수 분포")
        st.plotly_chart(distribution_chart(samples), use_container_width=True)

        show_cols = [c for c in
                     ["product_name", "rating", "trust_score", "review_content"]
                     if c in samples.columns]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-label">신뢰도 상위 5개</div>', unsafe_allow_html=True)
            st.dataframe(
                samples.sort_values("trust_score", ascending=False).head(5)[show_cols],
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown('<div class="section-label">신뢰도 하위 5개</div>', unsafe_allow_html=True)
            st.dataframe(
                samples.sort_values("trust_score", ascending=True).head(5)[show_cols],
                use_container_width=True,
                hide_index=True,
            )
