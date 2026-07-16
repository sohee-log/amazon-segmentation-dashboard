from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from model2_trust_tab import render as render_trust_tab
from model3_alert_tab import render as render_alert_tab
from segment_naming import ATTENTION, PREMIUM, VALUE, assign_segment_names

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "user_segments.csv"
SAMPLE_PATH = APP_DIR / "data" / "sample_user_segments.csv"

# --- 디자인 토큰 -------------------------------------------------------------
# 에디토리얼 톤: off-white 캔버스, 웜 니어블랙 잉크, 채도 높은 CTA 색 없음.
INK = "#0c0a09"
BODY = "#4e4e4e"
MUTED = "#777169"
HAIRLINE = "#e7e5e4"
CANVAS = "#f5f5f5"
SURFACE_CARD = "#ffffff"

# 브랜드의 파스텔 orb(mint/peach/lavender/sky/rose)는 '분위기 전용'이라
# 데이터 마크로 쓸 수 없다(off-white 위에서 대비가 무너짐).
# 그래서 같은 hue 계보를 유지한 채 명도만 낮춘 데이터 전용 팔레트를 파생시킨다.
# scripts/validate_palette 기준 전 항목 통과(대비·채도·색각 분리).
# 색은 세그먼트명에만 건다 — 클러스터 번호는 임의로 붙으므로 번호에 색을 고정하면
# 재학습 때 색이 뒤바뀐다(이름이 뒤바뀐 것과 같은 원인).
CLUSTER_COLORS = {
    ATTENTION: "#b45309",  # peach 계보
    PREMIUM: "#2563a8",    # sky 계보
    VALUE: "#0d9488",      # mint 계보
}

# orb 는 순수 분위기 — 마크·텍스트에는 절대 쓰지 않는다.
ORB_MINT = "#a7e5d3"
ORB_PEACH = "#f4c5a8"
ORB_LAVENDER = "#c8b8e0"
ORB_SKY = "#a8c8e8"

GRID = "rgba(12, 10, 9, 0.10)"
AXIS_TEXT = "#777169"
# 차트 내 한글도 시스템 고딕으로 떨어지지 않게 본문 스택과 동일하게 맞춘다.
PLOT_FONT = "Inter, Noto Sans KR, sans-serif"


st.set_page_config(
    page_title="Amazon Product Segmentation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* 모던 고딕 하나로 통일 — Inter(라틴) + Noto Sans KR(한글).
           헤드라인은 굵은 웨이트(700)에 타이트한 자간으로 세련된 인상을 주고,
           본문은 400/500 으로 가독성을 지킨다. OS 별 폴백 편차를 없애기 위해
           한글 글리프를 Noto Sans KR 로 고정한다. */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;700;800&display=swap');

        :root {
            --ink: #0c0a09;
            --body: #4e4e4e;
            --muted: #777169;
            --muted-soft: #a8a29e;
            --canvas: #f5f5f5;
            --canvas-soft: #fafafa;
            --surface-card: #ffffff;
            --surface-strong: #f0efed;
            --hairline: #e7e5e4;
            --hairline-soft: #f0efed;
            --hairline-strong: #d6d3d1;
            --primary: #292524;
            --primary-active: #0c0a09;
            --on-primary: #ffffff;

            --orb-mint: #a7e5d3;
            --orb-peach: #f4c5a8;
            --orb-lavender: #c8b8e0;
            --orb-sky: #a8c8e8;
            --orb-rose: #e8b8c4;

            /* display 와 body 모두 같은 고딕 스택 — 위계는 웨이트와 크기로 만든다. */
            --display: 'Inter', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
            --sans: 'Inter', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;

            --rounded-md: 8px;
            --rounded-lg: 12px;
            --rounded-xl: 16px;
            --rounded-xxl: 24px;
            --rounded-pill: 9999px;

            --soft-drop: 0 4px 16px rgba(0, 0, 0, 0.04);
        }

        header[data-testid="stHeader"] {
            height: 0;
            background: transparent;
        }

        .stApp {
            background: var(--canvas);
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: var(--sans);
            color: var(--body);
            letter-spacing: 0.16px;
        }

        /* ── 사이드바 ── */
        [data-testid="stSidebar"] {
            background: var(--canvas-soft);
            border-right: 1px solid var(--hairline);
        }

        [data-testid="stSidebar"] h1 {
            font-family: var(--sans) !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.96px !important;
            text-transform: uppercase;
            color: var(--muted) !important;
        }

        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 96px;
            max-width: 1200px;
        }

        /* ── 타이포 ── */
        h1, h2, h3 {
            font-family: var(--display);
            font-weight: 700;
            color: var(--ink);
        }

        p, span, label, div {
            font-family: var(--sans);
        }

        /* Streamlit 은 heading 텍스트를 앵커 링크용 자식 span 으로 감싸고,
           metric 값도 자식 div 에 넣는다. 위의 포괄 선택자가 그 자식들을 잡아
           display 폰트를 덮어쓰므로 부모를 상속하도록 되돌린다.
           (이게 없으면 헤드라인이 세리프로 안 보인다.) */
        h1 span, h2 span, h3 span,
        div[data-testid="stMetricValue"] div,
        div[data-testid="stMetricValue"] span {
            font-family: inherit;
            font-weight: inherit;
            letter-spacing: inherit;
        }

        /* heading 앵커 링크 아이콘은 에디토리얼 톤에 불필요 */
        span[data-testid="stHeaderActionElements"] {
            display: none;
        }

        /* ── 탭: 밑줄만 남긴 에디토리얼 내비 ── */
        div[data-testid="stTabs"] {
            position: relative;
            z-index: 5;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 32px;
            background: transparent;
            border-bottom: 1px solid var(--hairline);
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            background: transparent;
            padding: 0 0 14px;
            font-family: var(--sans);
            font-size: 15px;
            font-weight: 500;
            letter-spacing: 0;
            color: var(--muted);
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
            color: var(--ink);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--ink);
            height: 1px;
        }

        /* ── 스탯 타일 ── */
        div[data-testid="stMetric"] {
            background: var(--surface-card);
            border: 1px solid var(--hairline);
            border-radius: var(--rounded-xl);
            padding: 20px 24px;
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.96px !important;
            text-transform: uppercase;
            color: var(--muted) !important;
        }

        div[data-testid="stMetricValue"] {
            font-family: var(--display);
            font-weight: 600;
            font-size: 2.2rem;
            letter-spacing: -0.6px;
            color: var(--ink);
        }

        /* ── 히어로 ── */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 72px 0 64px;
            margin-bottom: 48px;
            border-bottom: 1px solid var(--hairline);
        }

        /* 파스텔 orb — 순수 분위기. 콘텐츠를 담지 않는다. */
        .hero::before {
            content: "";
            position: absolute;
            top: -180px;
            left: 42%;
            width: 620px;
            height: 620px;
            border-radius: 50%;
            background:
                radial-gradient(circle at 32% 34%, var(--orb-mint) 0%, transparent 62%),
                radial-gradient(circle at 68% 40%, var(--orb-sky) 0%, transparent 60%),
                radial-gradient(circle at 50% 72%, var(--orb-peach) 0%, transparent 64%);
            filter: blur(64px);
            opacity: 0.55;
            pointer-events: none;
        }

        .hero > * {
            position: relative;
            z-index: 1;
        }

        .hero .eyebrow {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.96px;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0 0 20px;
        }

        .hero h1 {
            font-family: var(--display);
            font-weight: 800;
            font-size: clamp(2.2rem, 5.2vw, 4rem);
            line-height: 1.08;
            letter-spacing: -1.6px;
            color: var(--ink);
            margin: 0 0 20px;
        }

        .hero p {
            max-width: 620px;
            margin: 0;
            font-size: 16px;
            line-height: 1.5;
            letter-spacing: 0.16px;
            color: var(--body);
        }

        /* ── 섹션 ── */
        .section {
            margin: 96px 0 24px;
        }

        .section-label {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.96px;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 10px;
        }

        .section h2 {
            font-family: var(--display);
            font-weight: 700;
            font-size: 2rem;
            line-height: 1.2;
            letter-spacing: -0.8px;
            color: var(--ink);
            margin: 0;
        }

        .section-sub {
            margin: 10px 0 0;
            max-width: 620px;
            font-size: 16px;
            line-height: 1.5;
            letter-spacing: 0.16px;
            color: var(--body);
        }

        /* ── 카드 ── */
        /* Streamlit 컬럼에 카드를 하나씩 넣으면 중간 래퍼에 명시적 높이가 없어
           height:100% 가 해소되지 않아 카드 밑단이 어긋난다. 카드 묶음은
           st.columns 대신 이 grid 하나로 렌더해 높이를 자동 정렬한다.
           auto-fit 덕에 좁은 화면에서 2단→1단으로 자연히 접힌다. */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            align-items: stretch;
        }

        .card {
            height: 100%;
            box-sizing: border-box;
            background: var(--surface-card);
            border: 1px solid var(--hairline);
            border-radius: var(--rounded-xl);
            padding: 24px;
            transition: box-shadow 160ms ease;
        }

        .card:hover {
            box-shadow: var(--soft-drop);
        }

        .card h3 {
            font-family: var(--display);
            font-weight: 600;
            font-size: 1.3rem;
            line-height: 1.25;
            letter-spacing: -0.4px;
            color: var(--ink);
            margin: 0 0 10px;
        }

        .card p {
            margin: 0;
            font-size: 15px;
            line-height: 1.55;
            letter-spacing: 0.15px;
            color: var(--body);
        }

        .card .step {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.96px;
            text-transform: uppercase;
            color: var(--muted-soft);
            margin: 0 0 12px;
        }

        /* 세그먼트 배지 — 색 점이 정체성을 나르고, 글자는 잉크를 유지한다. */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px 4px 10px;
            border-radius: var(--rounded-pill);
            background: var(--surface-strong);
            color: var(--ink);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.96px;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }


        /* ── 폼: 필 CTA, 8px 인풋 ── */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: var(--rounded-md);
            border-color: var(--hairline-strong);
            background: var(--surface-card);
        }

        .stButton > button {
            border-radius: var(--rounded-pill);
            background: var(--primary);
            color: var(--on-primary);
            border: none;
            font-family: var(--sans);
            font-size: 15px;
            font-weight: 500;
            padding: 10px 20px;
            height: 40px;
        }

        .stButton > button:hover {
            background: var(--primary-active);
            color: var(--on-primary);
        }

        /* ── 데이터프레임 ── */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--hairline);
            border-radius: var(--rounded-lg);
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--hairline);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_segments() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
    df = pd.read_csv(path)
    rename_map = {"PC1": "x", "PC2": "y", "PC3": "z"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"cluster", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
        st.stop()

    df["cluster"] = df["cluster"].astype(str)
    if "segment_name" not in df.columns:
        # 번호로 이름을 찍지 않고 군집 통계에서 유도한다.
        try:
            df["segment_name"] = df["cluster"].map(assign_segment_names(df))
        except ValueError:
            df["segment_name"] = "Segment " + df["cluster"]
    if "product_id" not in df.columns:
        df["product_id"] = df.index.astype(str)
    if "product_name" not in df.columns:
        df["product_name"] = "Product " + df["product_id"].astype(str)
    return df


def fmt_rs(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"Rs.{value:,.0f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def mean_value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def section(label: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(
        f"""
        <div class="section">
            <div class="section-label">{label}</div>
            <h2>{title}</h2>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_html(eyebrow: str, title: str, body: str) -> str:
    """카드 한 장을 만든다.

    HTML 은 반드시 들여쓰기·줄바꿈 없이 한 줄로 만든다. st.markdown 은 4칸 이상
    들여쓴 줄을 코드 블록으로 해석해서, 들여쓴 카드는 <pre> 로 삼켜진다.
    """
    return f'<div class="card">{eyebrow}<h3>{title}</h3><p>{body}</p></div>'


def render_card_grid(cards: list[str]) -> None:
    st.markdown(f'<div class="card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def axis_style(title: str) -> dict:
    return {
        "title": {"text": title, "font": {"size": 12, "color": AXIS_TEXT}},
        "showbackground": False,
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "color": AXIS_TEXT,
        "tickfont": {"size": 11, "color": AXIS_TEXT},
    }


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": cluster,
                "segment": group["segment_name"].iloc[0],
                "products": len(group),
                "avg_price": mean_value(group, "actual_price"),
                "avg_discount": mean_value(group, "discount_percentage"),
                "sentiment": mean_value(group, "sentiment_score"),
                "rating_gap": mean_value(group, "rating_gap"),
                "rating": mean_value(group, "rating"),
                "review_count": mean_value(group, "rating_count"),
            }
        )
    return pd.DataFrame(rows)


def make_3d_figure(df: pd.DataFrame) -> go.Figure:
    hover_cols = [
        col
        for col in [
            "product_id",
            "product_name",
            "segment_name",
            "actual_price",
            "discount_percentage",
            "sentiment_score",
            "rating_gap",
            "rating",
            "rating_count",
            "category",
        ]
        if col in df.columns
    ]
    ev = [mean_value(df, "pca1_variance"), mean_value(df, "pca2_variance"), mean_value(df, "pca3_variance")]
    labels = {
        "x": f"PC1 ({ev[0]:.0%})" if not pd.isna(ev[0]) else "PC1",
        "y": f"PC2 ({ev[1]:.0%})" if not pd.isna(ev[1]) else "PC2",
        "z": f"PC3 ({ev[2]:.0%})" if not pd.isna(ev[2]) else "PC3",
        "segment_name": "Segment",
    }
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="segment_name",
        color_discrete_map=CLUSTER_COLORS,
        hover_data=hover_cols,
        custom_data=["product_id"],
        opacity=0.82,
        height=620,
        labels=labels,
    )
    # 겹치는 마크는 서피스 링으로 분리한다.
    fig.update_traces(marker={"size": 3.4, "line": {"width": 0.5, "color": SURFACE_CARD}})

    # 중심점은 잉크 다이아몬드 — 색이 아니라 형태로 구분한다.
    centroids = df.groupby("segment_name")[["x", "y", "z"]].mean()
    fig.add_trace(
        go.Scatter3d(
            x=centroids["x"],
            y=centroids["y"],
            z=centroids["z"],
            mode="markers",
            marker={
                "size": 7,
                "color": INK,
                "symbol": "diamond",
                "line": {"color": SURFACE_CARD, "width": 1},
            },
            name="Centroid",
            hovertext=list(centroids.index),
            hoverinfo="text",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
            "font": {"color": BODY, "size": 13, "family": PLOT_FONT},
        },
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": axis_style(labels["x"]),
            "yaxis": axis_style(labels["y"]),
            "zaxis": axis_style(labels["z"]),
            "camera": {"eye": {"x": 1.7, "y": 1.55, "z": 1.15}},
        },
        font={"color": BODY, "family": PLOT_FONT},
        hoverlabel={
            "bgcolor": SURFACE_CARD,
            "bordercolor": HAIRLINE,
            "font": {"color": INK, "family": PLOT_FONT, "size": 12},
        },
    )
    return fig


def insight_text(segment: str) -> tuple[str, str]:
    mapping = {
        ATTENTION: (
            "별점 인플레이션 위험군",
            "저가·대량 리뷰 상품군입니다. 별점은 4.1대를 유지하지만 텍스트 감정이 가장 낮아 괴리가 가장 큽니다. "
            "품질, 배송, 상세 설명을 우선 점검해야 합니다.",
        ),
        PREMIUM: (
            "고가·고인기 신뢰 상품",
            "가격과 리뷰 수가 모두 높은 핵심 상품군입니다. 괴리도 큰 편이라 과도한 할인보다 "
            "브랜드 신뢰와 로열티 전략이 더 적합합니다.",
        ),
        VALUE: (
            "정직한 만족 신호",
            "별점은 가장 낮지만 텍스트 감정이 가장 높아 괴리가 가장 작은 상품군입니다. "
            "리뷰 신뢰도가 높으므로 긍정 리뷰 노출과 추천 강화로 전환을 높일 수 있습니다.",
        ),
    }
    return mapping.get(segment, ("Segment strategy", "세그먼트별 가격, 감정, 괴리 지표를 기준으로 운영 전략을 분리할 수 있습니다."))


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], tuple[float, float], tuple[float, float]]:
    st.sidebar.title("Filters")

    segments = list(df["segment_name"].dropna().unique())
    selected = st.sidebar.multiselect("상품 세그먼트", options=segments, default=segments)

    price_min = float(pd.to_numeric(df["actual_price"], errors="coerce").min()) if "actual_price" in df else 0.0
    price_max = float(pd.to_numeric(df["actual_price"], errors="coerce").max()) if "actual_price" in df else 1.0
    price_range = st.sidebar.slider("가격 범위", price_min, price_max, (price_min, price_max))

    gap_min = float(pd.to_numeric(df["rating_gap"], errors="coerce").min()) if "rating_gap" in df else -1.0
    gap_max = float(pd.to_numeric(df["rating_gap"], errors="coerce").max()) if "rating_gap" in df else 1.0
    gap_range = st.sidebar.slider("별점-감정 괴리", gap_min, gap_max, (gap_min, gap_max), format="%.3f")

    st.sidebar.divider()
    st.sidebar.caption("Data source")
    st.sidebar.caption("user_segments.csv" if DATA_PATH.exists() else "sample_user_segments.csv")
    return selected, price_range, gap_range


def apply_filters(
    df: pd.DataFrame,
    selected_segments: list[str],
    price_range: tuple[float, float],
    gap_range: tuple[float, float],
) -> pd.DataFrame:
    filtered = df[df["segment_name"].isin(selected_segments)].copy()
    if "actual_price" in filtered.columns:
        price = pd.to_numeric(filtered["actual_price"], errors="coerce")
        filtered = filtered[price.between(price_range[0], price_range[1])]
    if "rating_gap" in filtered.columns:
        gap = pd.to_numeric(filtered["rating_gap"], errors="coerce")
        filtered = filtered[gap.between(gap_range[0], gap_range[1])]
    return filtered


def render_hero(df: pd.DataFrame) -> None:
    algo = df["model_algo"].iloc[0] if "model_algo" in df.columns and not df.empty else "KMeans"
    variance = mean_value(df, "pca_total_variance")
    variance_text = f"{variance:.0%}" if not pd.isna(variance) else "PCA"
    st.markdown(
        f"""
        <div class="hero">
            <p class="eyebrow">Product Segmentation</p>
            <h1>별점이 말하지 않는<br>상품의 진짜 신호</h1>
            <p>
                가격, 텍스트 감정, 별점-감정 괴리, 리뷰 수를 기반으로 상품을 세 세그먼트로 나눕니다.
                {algo} 결과를 PCA 3D 공간({variance_text} variance)에 펼쳐 탐색합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 상품", f"{len(df):,}")
    c2.metric("평균 가격", fmt_rs(mean_value(df, "actual_price")))
    c3.metric("평균 감정 점수", f"{mean_value(df, 'sentiment_score'):.3f}")
    c4.metric("평균 괴리", f"{mean_value(df, 'rating_gap'):.3f}")


def render_cluster_cards(summary: pd.DataFrame) -> None:
    section("Playbook", "세그먼트별 운영 전략", "각 군집이 무엇을 뜻하고, 무엇을 해야 하는지.")
    cards = []
    for _, row in summary.iterrows():
        title, body = insight_text(row["segment"])
        color = CLUSTER_COLORS.get(row["segment"], INK)
        cards.append(card_html(f'<div class="badge"><span class="dot" style="background:{color}"></span>{row["segment"]}</div>', title, body))
    render_card_grid(cards)


def render_detail_panel(df: pd.DataFrame, selected_product_id: str | None) -> None:
    section("Detail", "상품 상세", "3D 맵에서 점을 클릭하면 해당 상품이 여기에 연결됩니다.")
    if df.empty:
        st.info("필터 조건에 맞는 상품이 없습니다.")
        return

    labels = (df["product_id"].astype(str) + " | " + df["product_name"].astype(str).str.slice(0, 60)).tolist()
    ids = df["product_id"].astype(str).tolist()
    index = ids.index(selected_product_id) if selected_product_id in ids else 0
    label = st.selectbox("상세 상품 선택", labels, index=index)
    product_id = label.split(" | ", 1)[0]
    product = df[df["product_id"].astype(str) == product_id].iloc[0]

    # 세그먼트명은 배지로 — 스탯 타일에 넣으면 긴 한글이 잘린다(타일은 숫자용).
    segment = str(product.get("segment_name", "-"))
    color = CLUSTER_COLORS.get(segment, INK)
    st.markdown(
        f'<div class="badge"><span class="dot" style="background:{color}"></span>{segment}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", fmt_rs(pd.to_numeric(product.get("actual_price", np.nan), errors="coerce")))
    c2.metric("Rating", f"{float(product.get('rating', np.nan)):.2f}")
    c3.metric("Rating Gap", f"{float(product.get('rating_gap', np.nan)):.3f}")
    c4.metric("Reviews", f"{pd.to_numeric(product.get('rating_count', np.nan), errors='coerce'):,.0f}")

    show_cols = [
        col
        for col in [
            "product_id",
            "product_name",
            "category",
            "segment_name",
            "actual_price",
            "discount_percentage",
            "sentiment_score",
            "rating_gap",
            "rating",
            "rating_count",
        ]
        if col in df.columns
    ]
    st.dataframe(pd.DataFrame([product[show_cols]]), use_container_width=True, hide_index=True)


def render_model_notes() -> None:
    section("Method", "모델 이야기", "피처 설계에서 운영 액션까지 세 단계.")
    steps = [
        (
            "01",
            "Feature Engineering",
            "할인율, 텍스트 감정, 별점-감정 괴리, 로그 가격, 로그 리뷰 수를 사용해 상품의 가격·만족도·신뢰도 구조를 만들었습니다.",
        ),
        (
            "02",
            "Model Selection",
            "K-Means와 GMM을 세 지표로 비교하고, 발표 해석성을 위해 k=3 세그먼트를 사용했습니다. 최종 결과는 PCA 3D로 압축했습니다.",
        ),
        (
            "03",
            "Business Action",
            "각 세그먼트를 품질 점검, 프리미엄 유지, 리뷰 노출 강화 전략으로 연결해 실무 의사결정에 바로 사용할 수 있게 했습니다.",
        ),
    ]
    render_card_grid(
        [card_html(f'<p class="step">{step}</p>', title, body) for step, title, body in steps]
    )


def render_first_tab(filtered: pd.DataFrame) -> None:
    render_hero(filtered)
    render_metrics(filtered)

    section("Map", "3D 세그먼테이션 맵", "회전·확대해 군집 구조를 살펴보고, 점을 클릭해 상품을 선택합니다.")
    left, right = st.columns([1.55, 1], gap="large")
    selected_product = None
    with left:
        fig = make_3d_figure(filtered)
        try:
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="cluster_map",
            )
            points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
            if points and points[0].get("customdata"):
                selected_product = str(points[0]["customdata"][0])
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)

    with right:
        summary = cluster_summary(filtered)
        if summary.empty:
            st.info("필터 조건에 맞는 세그먼트가 없습니다.")
        else:
            chart = px.bar(
                summary,
                x="products",
                y="segment",
                orientation="h",
                color="segment",
                color_discrete_map=CLUSTER_COLORS,
                text="products",
            )
            chart.update_traces(
                textposition="outside",
                textfont={"color": BODY, "family": PLOT_FONT, "size": 12},
                marker={"line": {"width": 0}},
                width=0.5,
                cliponaxis=False,
            )
            chart.update_layout(
                height=260,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 28, "t": 8, "b": 0},
                font={"color": BODY, "family": PLOT_FONT},
                bargap=0.42,
                xaxis={"visible": False},
                yaxis={
                    "title": None,
                    "showgrid": False,
                    "ticksuffix": "  ",
                    "tickfont": {"color": INK, "size": 13},
                },
            )
            st.plotly_chart(chart, use_container_width=True)

            table = summary.copy()
            table["avg_price"] = table["avg_price"].map(fmt_rs)
            table["avg_discount"] = table["avg_discount"].map(fmt_pct)
            table["review_count"] = table["review_count"].map(lambda v: f"{v:,.0f}")
            st.dataframe(table, use_container_width=True, hide_index=True)

    if not filtered.empty:
        render_cluster_cards(cluster_summary(filtered))

    render_detail_panel(filtered, selected_product)
    render_model_notes()

    section("Data", "필터링된 상품 데이터")
    show_cols = [
        col
        for col in [
            "product_id",
            "product_name",
            "segment_name",
            "actual_price",
            "discount_percentage",
            "sentiment_score",
            "rating_gap",
            "rating",
            "rating_count",
            "category",
        ]
        if col in filtered.columns
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    df = load_segments()
    selected_segments, price_range, gap_range = render_sidebar(df)
    filtered = apply_filters(df, selected_segments, price_range, gap_range)

    tab1, tab2, tab3 = st.tabs(["Product Segmentation", "Review Trust Score", "Early Warning"])
    with tab1:
        render_first_tab(filtered)
    with tab2:
        render_trust_tab()
    with tab3:
        render_alert_tab()


if __name__ == "__main__":
    main()
