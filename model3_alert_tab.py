"""모델 3 대시보드 탭: 불만 고객 조기 경보 (Early Warning System)

members/model3 (Chart.js 단독 HTML) + members/member3/README.md 의 분석을
대시보드 에디토리얼 톤으로 옮긴 것. 점수 계산식·키워드 사전·임계값은 원본
로직을 그대로 따르고, 원본 페이지에 있던 데이터 인사이트 · 작동 방식 플로우 ·
위험도 미터 · 근거 테이블 · 단계별 AI 인사이트/권장 대응까지 모두 복원한다.

    from model3_alert_tab import render as render_alert_tab
    ...
    with tab3:
        render_alert_tab()
"""
from __future__ import annotations

import html
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
# 결과 헤드라인 — 원본 HTML 의 문구를 그대로 살린다.
LEVEL_HEAD = {
    "Normal": "정상 리뷰",
    "Risk": "불만 가능성 감지",
    "Critical": "불만 고객 경보",
}
LEVEL_SUB = {
    "Normal": "현재 기준으로는 불만 위험도가 낮습니다.",
    "Risk": "모니터링이 필요한 주의 리뷰입니다.",
    "Critical": "즉각적인 확인이 필요한 고위험 리뷰입니다.",
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


# --- 원본 renderResult() 의 태그·근거·인사이트 로직 이식 ----------------------
def sentiment_tag(sent: float) -> str:
    if sent <= -0.6:
        return "매우 부정"
    if sent < -0.2:
        return "부정"
    if sent < 0.2:
        return "중립"
    return "긍정"


def keyword_tag(count: int) -> str:
    if count >= 3:
        return "높음"
    if count >= 1:
        return "보통"
    return "낮음"


def length_tag(words: int) -> str:
    if words >= 50:
        return "상세"
    if words >= 20:
        return "보통"
    return "짧음"


def risk_word(kind: str, r: dict) -> str:
    """근거 테이블의 '위험 수준' 칸 — 색약 대비를 위해 색이 아닌 낱말로 나른다."""
    if kind == "sentiment":
        s = r["sentiment"]
        return "위험" if s <= -0.6 else "주의" if s < -0.2 else "낮음"
    if kind == "keyword":
        c = r["count"]
        return "위험" if c >= 3 else "주의" if c >= 1 else "낮음"
    w = r["words"]
    return "상세 불만 가능" if w >= 50 else "보통" if w >= 20 else "짧음"


def insight_action(r: dict) -> tuple[str, str]:
    """단계별 AI 분석 인사이트 · 권장 대응 (원본 HTML 문구 이식)."""
    level = r["level"]
    if level == "Critical":
        hits = ", ".join(r["matched"][:3]) if r["matched"] else "강한 부정 표현"
        insight = (
            f"리뷰에서 <strong>{html.escape(hits)}</strong>이(가) 감지되었고 감정 점수도 "
            "부정적으로 나타났습니다. 제품 결함, 환불 요청, 작동 불량 가능성이 있는 리뷰로 해석할 수 있습니다."
        )
        action = (
            "고객 문의를 <strong>우선 확인</strong>하고, 환불·교환 요청 또는 제품 결함 여부를 "
            "빠르게 점검하는 것이 좋습니다."
        )
    elif level == "Risk":
        insight = (
            "일부 부정 표현이 포함되어 불만 가능성이 있습니다. 아직 심각한 수준은 아니지만 "
            "반복 발생 시 제품 개선 신호로 볼 수 있습니다."
        )
        action = (
            "동일 상품군에서 유사 키워드가 반복되는지 <strong>모니터링</strong>하고, 필요 시 "
            "고객 응대를 준비하는 것이 좋습니다."
        )
    else:
        insight = (
            "강한 부정 감정이나 핵심 불만 키워드가 크게 나타나지 않았습니다. 현재 기준에서는 "
            "일반 리뷰로 분류됩니다."
        )
        action = (
            "즉각적인 대응보다는 <strong>일반 리뷰 데이터로 누적</strong>하여 상품 만족도 분석에 "
            "활용할 수 있습니다."
        )
    return insight, action


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


def render_result(r: dict) -> None:
    """원본 result-box 를 에디토리얼 톤으로 복원한다.

    st.markdown 은 4칸 이상 들여쓴 줄을 코드블록으로 삼키므로, HTML 은 줄마다
    들여쓰기 없이 이어 붙인다.
    """
    color = LEVEL_COLORS[r["level"]]
    ko = LEVEL_KO[r["level"]]
    head = LEVEL_HEAD[r["level"]]
    sub = LEVEL_SUB[r["level"]]
    # 위험도 % — 원본과 동일하게 0.70 을 상한으로 환산한다.
    risk_pct = min(round(r["score"] / 0.70 * 100), 100)
    insight, action = insight_action(r)

    matched = r["matched"]
    chips = (
        "".join(f'<span class="chip">{html.escape(w)}</span>' for w in matched)
        if matched else '<span class="chip">탐지된 부정 키워드 없음</span>'
    )

    parts = [
        '<div class="badge" style="margin-bottom:16px">',
        f'<span class="dot" style="background:{color}"></span>{r["level"]} · {ko}</div>',
        f'<h3 style="font-family:var(--display);font-weight:700;font-size:1.6rem;'
        f'letter-spacing:-0.6px;color:var(--ink);margin:0 0 4px">{head}</h3>',
        f'<p style="color:var(--body);font-size:14px;margin:0 0 18px">{sub}</p>',
        # 위험도 미터
        '<div class="risk-meter"><div class="meter-row">',
        '<span class="meter-label">고객 불만 위험도</span>',
        f'<span class="meter-value">{risk_pct}%</span></div>',
        f'<div class="meter-track"><div class="meter-fill" '
        f'style="width:{risk_pct}%;background:{color}"></div></div></div>',
        # 시그널 카드 3장
        '<div class="signal-grid">',
        f'<div class="signal-card"><span class="lbl">감정 점수</span>'
        f'<span class="val">{r["sentiment"]:.3f}</span>'
        f'<span class="tag">{sentiment_tag(r["sentiment"])}</span></div>',
        f'<div class="signal-card"><span class="lbl">부정 키워드 수</span>'
        f'<span class="val">{r["count"]}</span>'
        f'<span class="tag">{keyword_tag(r["count"])}</span></div>',
        f'<div class="signal-card"><span class="lbl">리뷰 길이</span>'
        f'<span class="val">{r["words"]}</span>'
        f'<span class="tag">{length_tag(r["words"])}</span></div>',
        "</div>",
        # 근거 테이블
        '<table class="reason-table"><thead><tr>'
        "<th>감지 신호</th><th>분석 결과</th><th>위험 수준</th></tr></thead><tbody>",
        f"<tr><td>고객 감정</td><td>"
        f'{"부정 감정 " + format(r["sentiment"], ".3f") + " 감지" if r["sentiment"] < 0 else "부정 감정 낮음 (" + format(r["sentiment"], ".3f") + ")"}'
        f"</td><td>{risk_word('sentiment', r)}</td></tr>",
        f"<tr><td>불만 키워드</td><td>"
        f'{str(r["count"]) + "개의 불만 키워드 감지" if r["count"] > 0 else "불만 키워드 없음"}'
        f"</td><td>{risk_word('keyword', r)}</td></tr>",
        f"<tr><td>리뷰 길이</td><td>{r['words']}단어로 작성됨</td>"
        f"<td>{risk_word('length', r)}</td></tr>",
        "</tbody></table>",
        # 감지된 불만 표현
        '<div class="section-label" style="margin:18px 0 4px">감지된 불만 표현</div>',
        f'<div class="chip-row">{chips}</div>',
        # AI 인사이트 · 권장 대응
        f'<div class="insight-box"><strong>AI 분석 인사이트</strong><br>{insight}</div>',
        f'<div class="action-box"><strong>권장 대응</strong><br>{action}</div>',
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


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

    # --- 분석 배경 / 문제 정의 (member3 README §1) ----------------------------
    section("Background", "왜 별점만으로는 부족한가",
            "별점은 후하지만 본문은 불만인 리뷰가 존재합니다. 텍스트를 읽어야 진짜 신호가 보입니다.")
    st.markdown(
        '<div class="card-grid">'
        '<div class="card"><h3>별점-본문 괴리</h3>'
        '<p>별점은 4~5점이지만 리뷰 본문에는 고장·환불·불만이 담긴 경우가 많습니다. '
        '별점만 보면 이런 리뷰를 놓칩니다.</p></div>'
        '<div class="card"><h3>즉시 대응 리뷰</h3>'
        '<p>제품 고장, 환불·교환 요청처럼 빠른 대응이 필요한 리뷰를 사람이 전부 읽어 골라내기는 '
        '어렵습니다.</p></div>'
        '<div class="card"><h3>감정 강도의 사각지대</h3>'
        '<p>강한 부정 감정이 표현됐어도 별점만으로는 확인되지 않습니다. 텍스트 감정을 수치화해야 '
        '드러납니다.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- 자동 추출 부정 키워드 + 데이터 인사이트 -----------------------------
    section("Keywords", "자동 추출 부정 키워드 TOP 15",
            "TF-IDF 로 저평점·부정 리뷰에서 상대적으로 강하게 나타난 단어를 자동 추출했습니다.")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(keyword_chart(), use_container_width=True)
    with right:
        st.markdown(
            '<div class="card"><p class="step">데이터 분석 인사이트</p>'
            '<ul class="insight-list">'
            '<li><strong>bad</strong>, <strong>broken</strong> 은 강한 불만을 직접적으로 나타내는 핵심 키워드입니다.</li>'
            '<li><strong>battery</strong>, <strong>charging</strong> 은 제품 기능·성능 문제와 연결됩니다.</li>'
            '<li><strong>return</strong>, <strong>refund</strong> 는 고객 대응 우선순위를 높이는 신호입니다.</li>'
            '</ul>'
            '<div class="mini-kpi-row">'
            '<div class="mini-kpi"><b>15</b><span>핵심 키워드</span></div>'
            '<div class="mini-kpi"><b>3단계</b><span>경보 분류</span></div>'
            '<div class="mini-kpi"><b>실시간</b><span>리뷰 판정</span></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # --- 작동 방식 플로우 ----------------------------------------------------
    section("How it works", "조기 경보 시스템 작동 방식",
            "리뷰 한 건이 경보 단계로 판정되기까지의 다섯 단계.")
    st.markdown(
        '<div class="flow">'
        '<div class="flow-step"><span class="ico">✍️</span><span class="n">01</span>리뷰 입력</div>'
        '<div class="flow-step"><span class="ico">😡</span><span class="n">02</span>감정 분석</div>'
        '<div class="flow-step"><span class="ico">🔎</span><span class="n">03</span>키워드 탐지</div>'
        '<div class="flow-step"><span class="ico">📊</span><span class="n">04</span>위험 점수</div>'
        '<div class="flow-step"><span class="ico">🚨</span><span class="n">05</span>경보 판단</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- 시뮬레이터 ----------------------------------------------------------
    section("Simulator", "리뷰 입력 시뮬레이터", "리뷰를 붙여넣으면 경보 단계를 계산합니다.")
    review_text = st.text_area(
        "리뷰 텍스트 입력",
        value="The product stopped working after two days. Battery is terrible and I want a refund.",
        height=120,
        placeholder="분석할 리뷰 텍스트를 영어로 입력하세요...",
        key="alert_input",
    )

    if st.button("불만 고객 조기 경보 분석하기", key="alert_btn"):
        if not review_text.strip():
            st.warning("리뷰 텍스트를 입력해 주세요.")
        else:
            r = alert_score(review_text)

            section("Result", "분석 결과")
            left, right = st.columns([1.15, 1], gap="large")
            with left:
                render_result(r)
            with right:
                st.markdown('<div class="section-label">경보 점수 구성</div>', unsafe_allow_html=True)
                st.plotly_chart(component_chart(r["components"]), use_container_width=True)
                st.caption(
                    "위험 점수가 어떤 요소에서 나왔는지 보여줍니다. 부정 감정이 강하고, 부정 "
                    "키워드가 많고, 리뷰가 상세할수록 불만 가능성이 높은 리뷰로 판단합니다."
                )

    # --- 실제 예시 ----------------------------------------------------------
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

    # --- 분석 방법 (member3 README §3) --------------------------------------
    section("Method", "분석 방법", "감정 수치화 → 키워드 자동 추출 → 가중합 점수, 세 단계.")
    render_method_cards()

    with st.expander("모델 계산식 자세히 보기"):
        st.markdown(
            "```\n"
            "Early Warning Score\n"
            "= 0.55 × negative_sentiment_strength   (부정 감정 강도)\n"
            "+ 0.30 × negative_keyword_score        (min(부정 키워드 수 / 10, 1))\n"
            "+ 0.15 × review_length_norm            (min(단어 수 / 120, 1))\n"
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


def render_method_cards() -> None:
    st.markdown(
        '<div class="card-grid">'
        '<div class="card"><p class="step">01 · Sentiment</p>'
        '<h3>리뷰 텍스트 수치화</h3>'
        '<p>TextBlob 으로 리뷰별 감정 극성을 -1(매우 부정)~1(매우 긍정)로 계산합니다. '
        '감정이 음수일 때만 부정 감정 강도로 반영해 불만 신호에 집중합니다.</p></div>'
        '<div class="card"><p class="step">02 · Keywords</p>'
        '<h3>부정 키워드 자동 추출</h3>'
        '<p>사람이 지정하던 방식 대신, 저평점·부정 감정 리뷰에 TF-IDF 를 적용해 주요 부정 '
        '키워드를 자동으로 뽑습니다. product·good 같은 일반어는 불용어 처리로 걸러냅니다.</p></div>'
        '<div class="card"><p class="step">03 · Score</p>'
        '<h3>조기 경보 점수 계산</h3>'
        '<p>부정 감정 강도(0.55) · 부정 키워드 점수(0.30) · 리뷰 길이(0.15)를 가중합해 '
        'Early Warning Score 를 만들고 Normal / Risk / Critical 로 나눕니다.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )
