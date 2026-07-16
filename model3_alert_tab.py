"""모델 3 대시보드 탭: 불만 고객 조기 경보 (Early Warning System)

members/model3 (Chart.js 단독 HTML) 의 분석을 Streamlit 탭으로 옮긴 것.
점수 계산식·키워드 사전·임계값은 원본 로직을 그대로 따른다.

    from model3_alert_tab import render as render_alert_tab
    ...
    with tab3:
        render_alert_tab()
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- 디자인 토큰 (app.py 와 동일 계보) ---------------------------------------
INK = "#0c0a09"
BODY = "#4e4e4e"
PLOT_FONT = "Inter, Noto Sans KR, sans-serif"

# 경보 단계는 Normal < Risk < Critical 로 순서가 있는 '심각도'다.
# 초록/앰버/빨강 3색은 적록색약에서 앰버와 빨강이 ΔE 4.3 으로 뭉개져 쓸 수 없다.
# 심각도를 명도로 싣는 단일 warm hue 램프를 쓴다 — 명도 차이는 색약에서도 남는다.
# 검증 통과: 명도 단조·인접 ΔL·밝은 끝 대비 2.18:1·단일 hue 5°.
LEVEL_COLORS = {
    "Normal": "#e8927a",
    "Risk": "#c9503a",
    "Critical": "#8f2415",
}
LEVEL_KO = {
    "Normal": "정상 리뷰",
    "Risk": "주의 리뷰",
    "Critical": "즉시 대응 필요",
}

# 점수 구성 요소는 서로 다른 항목(identity)이므로 검증된 카테고리 팔레트를 쓴다.
COMPONENT_COLORS = ["#b45309", "#2563a8", "#0d9488"]

# --- 원본(members/model3) 로직 그대로 ----------------------------------------
NEGATIVE_KEYWORDS = [
    "bad", "broken", "poor", "waste", "wasted", "disappointed", "damage",
    "defective", "issue", "problem", "refund", "return", "doesnt", "dont",
    "failed", "useless", "worst", "stopped", "slow", "hard", "cheap",
    "fake", "missing", "dead", "faulty", "terrible", "awful",
]
POSITIVE_WORDS = [
    "good", "great", "excellent", "love", "perfect", "nice", "best",
    "amazing", "satisfied", "recommend", "useful", "durable", "easy",
]

KEYWORD_TOP15 = [
    ("bad", 9.79), ("broken", 8.46), ("doesnt", 7.85), ("battery", 6.84),
    ("poor", 5.78), ("issue", 4.21), ("return", 3.89), ("waste", 3.62),
    ("disappointed", 3.41), ("damage", 3.08), ("refund", 2.87), ("stopped", 2.65),
    ("slow", 2.44), ("defective", 2.12), ("worst", 1.96),
]

EXAMPLES = [
    ("Battery died after 2 weeks. Doesn't charge at all and waste of money.",
     "1.0", "-0.84", "battery, doesnt, waste", "Critical"),
    ("This product is broken. It stopped working after a few days. Very poor quality.",
     "2.0", "-0.63", "broken, stopped, poor", "Critical"),
    ("Charging is too slow and often doesn't work properly.",
     "2.5", "-0.52", "slow, doesnt, work", "Risk"),
    ("Not as good as expected. The remote barely works from a distance.",
     "3.0", "-0.21", "not, works", "Risk"),
]

RISK_THRESHOLD = 0.25
CRITICAL_THRESHOLD = 0.45


def sentiment_score(text: str) -> float:
    """원본과 동일: 부분 문자열(includes) 기준으로 긍/부정 단어를 센다."""
    lower = text.lower()
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in lower)
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    raw = (pos - neg) / max(pos + neg, 1)
    return max(-1.0, min(1.0, raw))


def count_matches(text: str) -> tuple[int, list[str]]:
    """원본과 동일: 단어 경계 기준으로 부정 키워드 출현 횟수를 센다."""
    lower = text.lower()
    count = 0
    matched = []
    for word in NEGATIVE_KEYWORDS:
        hits = re.findall(r"\b" + re.escape(word) + r"\b", lower)
        if hits:
            count += len(hits)
            matched.append(word)
    return count, matched


def alert_score(text: str) -> dict:
    """원본 계산식: |부정감정|*0.55 + min(키워드/10,1)*0.30 + min(단어수/120,1)*0.15"""
    words = len(text.split()) if text.strip() else 0
    sent = sentiment_score(text)
    negative_strength = abs(sent) if sent < 0 else 0.0
    count, matched = count_matches(text)

    keyword_score = min(count / 10, 1)
    length_norm = min(words / 120, 1)

    components = [
        negative_strength * 0.55,
        keyword_score * 0.30,
        length_norm * 0.15,
    ]
    score = sum(components)

    level = "Normal"
    if score >= CRITICAL_THRESHOLD:
        level = "Critical"
    elif score >= RISK_THRESHOLD:
        level = "Risk"

    return {
        "score": score, "level": level, "sentiment": sent, "words": words,
        "count": count, "matched": matched, "components": components,
    }


def section(label: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(
        f'<div class="section"><div class="section-label">{label}</div>'
        f"<h2>{title}</h2>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def keyword_chart() -> go.Figure:
    words = [k for k, _ in KEYWORD_TOP15][::-1]
    scores = [s for _, s in KEYWORD_TOP15][::-1]
    fig = go.Figure(
        go.Bar(
            x=scores, y=words, orientation="h",
            marker={"color": LEVEL_COLORS["Risk"], "line": {"width": 0}},
            text=[f"{s:.2f}" for s in scores], textposition="outside",
            textfont={"color": BODY, "size": 11, "family": PLOT_FONT},
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 36, "t": 8, "b": 0},
        font={"color": BODY, "family": PLOT_FONT},
        bargap=0.34,
        xaxis={"visible": False},
        yaxis={"title": None, "showgrid": False, "ticksuffix": "  ",
               "tickfont": {"color": INK, "size": 12}},
    )
    return fig


def component_chart(components: list[float]) -> go.Figure:
    labels = ["부정 감정", "부정 키워드", "리뷰 길이"]
    fig = go.Figure(
        go.Bar(
            x=labels, y=components,
            marker={"color": COMPONENT_COLORS, "line": {"width": 0}},
            text=[f"{c:.3f}" for c in components], textposition="outside",
            textfont={"color": BODY, "size": 11, "family": PLOT_FONT},
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        font={"color": BODY, "family": PLOT_FONT},
        bargap=0.5, showlegend=False,
        xaxis={"title": None, "tickfont": {"color": INK, "size": 12}},
        yaxis={"title": None, "gridcolor": "rgba(12,10,9,0.10)",
               "tickfont": {"color": BODY, "size": 11}, "range": [0, 0.6]},
    )
    return fig


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">Early Warning System</p>
            <h1>별점보다 먼저<br>도착하는 불만 신호</h1>
            <p>
                리뷰의 감정, 부정 키워드, 길이를 조합해 불만 위험도를 Normal · Risk · Critical
                세 단계로 분류합니다. 별점이 높아도 본문에 불만이 담긴 리뷰를 조기에 골라내
                대응 우선순위를 제안합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 스탯 타일에는 숫자만 — 긴 한글 문구를 넣으면 잘린다.
    c1, c2, c3 = st.columns(3)
    c1.metric("부정 키워드 사전", f"{len(NEGATIVE_KEYWORDS)}")
    c2.metric("Risk 기준", f"{RISK_THRESHOLD:.2f}")
    c3.metric("Critical 기준", f"{CRITICAL_THRESHOLD:.2f}")

    section("Keywords", "자동 추출 부정 키워드 TOP 15",
            "불만 리뷰에서 상대적으로 강하게 나타난 단어입니다.")
    st.plotly_chart(keyword_chart(), use_container_width=True)

    section("Simulator", "리뷰 입력 시뮬레이터", "리뷰를 붙여넣으면 경보 단계를 계산합니다.")
    review_text = st.text_area(
        "리뷰 텍스트 입력",
        height=120,
        placeholder="분석할 리뷰 텍스트를 영어로 입력하세요...",
        key="alert_input",
    )

    if st.button("경보 단계 분석하기", key="alert_btn"):
        if not review_text.strip():
            st.warning("리뷰 텍스트를 입력해 주세요.")
        else:
            r = alert_score(review_text)
            color = LEVEL_COLORS[r["level"]]

            section("Result", "분석 결과")
            left, right = st.columns([1, 1], gap="large")
            with left:
                # 색만으로 뜻을 나르지 않도록 단계 이름을 항상 함께 쓴다.
                st.markdown(
                    f'<div class="badge"><span class="dot" style="background:{color}"></span>'
                    f'{r["level"]} · {LEVEL_KO[r["level"]]}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("경보 점수", f"{r['score']:.3f}")
                m2.metric("감정 점수", f"{r['sentiment']:+.2f}")
                m3, m4 = st.columns(2)
                m3.metric("부정 키워드", f"{r['count']}")
                m4.metric("리뷰 길이", f"{r['words']}")
            with right:
                st.markdown('<div class="section-label">경보 점수 구성</div>', unsafe_allow_html=True)
                st.plotly_chart(component_chart(r["components"]), use_container_width=True)

            st.markdown(
                f'<div class="card"><p class="step">Matched keywords</p>'
                f'<p>{", ".join(r["matched"]) if r["matched"] else "없음"}</p></div>',
                unsafe_allow_html=True,
            )

    section("Examples", "실제 조기 경보 리뷰 예시",
            "높은 경보 점수를 받은 리뷰는 감정이 낮거나 부정 키워드가 많습니다.")
    st.dataframe(
        pd.DataFrame(EXAMPLES, columns=["리뷰 내용", "별점", "감정 점수", "부정 키워드", "경보 단계"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "이 표의 감정 점수·경보 단계는 원본 노트북 분석 기준 값입니다. "
        "위 시뮬레이터는 단어 사전 기반의 간이 계산이라 같은 리뷰라도 값이 다를 수 있습니다."
    )

    with st.expander("모델 계산식"):
        st.markdown(
            "```\n"
            "경보 점수 = |부정 감정| × 0.55\n"
            "          + min(부정 키워드 수 / 10, 1) × 0.30\n"
            "          + min(단어 수 / 120, 1) × 0.15\n"
            "\n"
            f"Critical : 점수 >= {CRITICAL_THRESHOLD}\n"
            f"Risk     : 점수 >= {RISK_THRESHOLD}\n"
            "Normal   : 그 외\n"
            "```"
        )
        st.caption(
            "감정 점수는 긍/부정 단어 사전 기반이며, 허위·불만 리뷰의 정답 라벨이 없으므로 "
            "참고용 보조 지표입니다."
        )
