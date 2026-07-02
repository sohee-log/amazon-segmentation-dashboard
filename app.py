from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "user_segments.csv"
SAMPLE_PATH = APP_DIR / "data" / "sample_user_segments.csv"

SEGMENT_NAMES = {
    "0": "관심 필요 상품",
    "1": "프리미엄 베스트셀러",
    "2": "가성비 호평 상품",
}

CLUSTER_COLORS = {
    "0": "#ff5a5f",
    "1": "#ff9900",
    "2": "#00a8e1",
    "관심 필요 상품": "#ff5a5f",
    "프리미엄 베스트셀러": "#ff9900",
    "가성비 호평 상품": "#00a8e1",
}


st.set_page_config(
    page_title="Amazon Product Segmentation",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --amazon-orange: #ff9900;
            --amazon-blue: #00a8e1;
            --panel: rgba(255, 255, 255, 0.055);
            --panel-strong: rgba(255, 255, 255, 0.092);
            --stroke: rgba(255, 255, 255, 0.13);
            --muted: rgba(245, 247, 251, 0.68);
        }

        header[data-testid="stHeader"] {
            height: 0;
            background: transparent;
        }

        .stApp {
            background:
                radial-gradient(circle at 16% 4%, rgba(255,153,0,0.14), transparent 24%),
                linear-gradient(135deg, #0f1115 0%, #151b22 48%, #10151b 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12161d 0%, #0d1117 100%);
            border-right: 1px solid var(--stroke);
        }

        .block-container {
            padding-top: 3.8rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        div[data-testid="stTabs"] {
            position: relative;
            z-index: 5;
        }

        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            min-height: 46px;
        }

        div[data-testid="stMetric"] {
            min-height: 112px;
            background: var(--panel);
            border: 1px solid var(--stroke);
            border-radius: 8px;
            padding: 16px 18px;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.72rem;
        }

        .hero {
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            padding: 24px 28px;
            background:
                linear-gradient(120deg, rgba(255,153,0,0.18), rgba(0,168,225,0.08)),
                rgba(255, 255, 255, 0.052);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
            margin: 8px 0 18px;
        }

        .hero h1 {
            margin: 0 0 8px;
            font-size: clamp(2.1rem, 4vw, 4rem);
            line-height: 1.04;
            letter-spacing: 0;
            color: #ffffff;
        }

        .hero p {
            max-width: 980px;
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .section-title {
            margin: 22px 0 10px;
            color: #ffffff;
            font-size: 1.24rem;
            font-weight: 760;
        }

        .insight-card {
            height: 100%;
            border: 1px solid var(--stroke);
            border-radius: 8px;
            padding: 18px;
            background: var(--panel);
        }

        .insight-card h3 {
            margin: 0 0 10px;
            color: #ffffff;
            font-size: 1.02rem;
        }

        .insight-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .cluster-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid var(--stroke);
            background: var(--panel-strong);
            color: #f5f7fb;
            font-size: 0.86rem;
            margin-bottom: 10px;
        }

        .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }

        .placeholder {
            border: 1px dashed rgba(255, 255, 255, 0.24);
            border-radius: 8px;
            padding: 28px;
            background: rgba(255, 255, 255, 0.04);
            color: var(--muted);
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
        df["segment_name"] = df["cluster"].map(SEGMENT_NAMES).fillna("Segment " + df["cluster"])
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


def axis_style(title: str) -> dict:
    return {
        "title": title,
        "showbackground": True,
        "backgroundcolor": "rgba(255,255,255,0.035)",
        "gridcolor": "rgba(255,255,255,0.12)",
        "zerolinecolor": "rgba(255,153,0,0.5)",
        "color": "rgba(245,247,251,0.78)",
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
        symbol="segment_name",
        color_discrete_map=CLUSTER_COLORS,
        hover_data=hover_cols,
        custom_data=["product_id"],
        opacity=0.74,
        height=650,
        labels=labels,
    )
    fig.update_traces(marker={"size": 4, "line": {"width": 0.3, "color": "rgba(255,255,255,0.45)"}})

    centroids = df.groupby("segment_name")[["x", "y", "z"]].mean()
    for segment, row in centroids.iterrows():
        fig.add_trace(
            go.Scatter3d(
                x=[row["x"]],
                y=[row["y"]],
                z=[row["z"]],
                mode="markers",
                marker={
                    "size": 10,
                    "color": "#ffffff",
                    "symbol": "diamond",
                    "line": {"color": "#0f1115", "width": 2},
                },
                name=f"Centroid - {segment}",
                showlegend=True,
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#f5f7fb"},
        },
        scene={
            "bgcolor": "rgba(255,255,255,0.02)",
            "xaxis": axis_style(labels["x"]),
            "yaxis": axis_style(labels["y"]),
            "zaxis": axis_style(labels["z"]),
            "camera": {"eye": {"x": 1.7, "y": 1.55, "z": 1.15}},
        },
        font={"color": "#f5f7fb"},
    )
    return fig


def insight_text(segment: str) -> tuple[str, str]:
    mapping = {
        "관심 필요 상품": (
            "별점 인플레이션 위험군",
            "별점은 높지만 텍스트 감정이 낮고 괴리가 큰 상품군입니다. 품질, 배송, 상세 설명 문제를 우선 점검해야 합니다.",
        ),
        "프리미엄 베스트셀러": (
            "고가·고인기 신뢰 상품",
            "가격과 리뷰 수가 높은 핵심 상품군입니다. 과도한 할인보다 브랜드 신뢰와 로열티 전략이 더 적합합니다.",
        ),
        "가성비 호평 상품": (
            "정직한 만족 신호",
            "별점은 상대적으로 보수적이지만 텍스트 감성이 좋은 상품군입니다. 긍정 리뷰 노출과 추천 강화로 전환을 높일 수 있습니다.",
        ),
    }
    return mapping.get(segment, ("Segment strategy", "세그먼트별 가격, 감정, 괴리 지표를 기준으로 운영 전략을 분리할 수 있습니다."))


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], tuple[float, float], tuple[float, float]]:
    st.sidebar.title("Control Panel")
    st.sidebar.caption("상품 세그먼트, 가격대, 별점-감정 괴리 기준으로 발표 중 바로 필터링할 수 있습니다.")

    segments = list(df["segment_name"].dropna().unique())
    selected = st.sidebar.multiselect("상품 세그먼트", options=segments, default=segments)

    price_min = float(pd.to_numeric(df["actual_price"], errors="coerce").min()) if "actual_price" in df else 0.0
    price_max = float(pd.to_numeric(df["actual_price"], errors="coerce").max()) if "actual_price" in df else 1.0
    price_range = st.sidebar.slider("가격 범위", price_min, price_max, (price_min, price_max))

    gap_min = float(pd.to_numeric(df["rating_gap"], errors="coerce").min()) if "rating_gap" in df else -1.0
    gap_max = float(pd.to_numeric(df["rating_gap"], errors="coerce").max()) if "rating_gap" in df else 1.0
    gap_range = st.sidebar.slider("별점-감정 괴리", gap_min, gap_max, (gap_min, gap_max), format="%.3f")

    st.sidebar.divider()
    st.sidebar.markdown("**Data source**")
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
            <h1>Amazon Product Segmentation</h1>
            <p>
                가격, 텍스트 감정, 별점-감정 괴리, 가격 로그, 리뷰 수 로그를 기반으로 상품을 3개 세그먼트로 나누고,
                {algo} 결과를 PCA 3D 공간({variance_text} variance)에서 발표용으로 시각화합니다.
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
    st.markdown('<div class="section-title">Segment Playbook</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, max(1, len(summary))))
    for i, row in summary.iterrows():
        title, body = insight_text(row["segment"])
        color = CLUSTER_COLORS.get(row["segment"], "#ff9900")
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="cluster-chip"><span class="dot" style="background:{color}"></span>{row["segment"]}</div>
                    <h3>{title}</h3>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_detail_panel(df: pd.DataFrame, selected_product_id: str | None) -> None:
    st.markdown('<div class="section-title">Product Detail</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("필터 조건에 맞는 상품이 없습니다.")
        return

    labels = (df["product_id"].astype(str) + " | " + df["product_name"].astype(str).str.slice(0, 60)).tolist()
    ids = df["product_id"].astype(str).tolist()
    index = ids.index(selected_product_id) if selected_product_id in ids else 0
    label = st.selectbox("상세 상품 선택", labels, index=index)
    product_id = label.split(" | ", 1)[0]
    product = df[df["product_id"].astype(str) == product_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", str(product.get("segment_name", "-")))
    c2.metric("Price", fmt_rs(pd.to_numeric(product.get("actual_price", np.nan), errors="coerce")))
    c3.metric("Rating", f"{float(product.get('rating', np.nan)):.2f}")
    c4.metric("Rating Gap", f"{float(product.get('rating_gap', np.nan)):.3f}")

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
    st.markdown('<div class="section-title">Model Story</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="insight-card">
                <h3>1. Feature Engineering</h3>
                <p>할인율, 텍스트 감정, 별점-감정 괴리, 로그 가격, 로그 리뷰 수를 사용해 상품의 가격·만족도·신뢰도 구조를 만들었습니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="insight-card">
                <h3>2. Model Selection</h3>
                <p>K-Means와 GMM을 비교하고, 발표 해석성을 위해 k=3 세그먼트를 사용했습니다. 최종 결과는 PCA 3D로 압축해 보여줍니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="insight-card">
                <h3>3. Business Action</h3>
                <p>각 세그먼트를 품질 점검, 프리미엄 유지, 리뷰 노출 강화 전략으로 연결해 실무 의사결정에 바로 사용할 수 있게 했습니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_first_tab(filtered: pd.DataFrame) -> None:
    render_hero(filtered)
    render_metrics(filtered)

    left, right = st.columns([1.55, 1], gap="large")
    selected_product = None
    with left:
        st.markdown('<div class="section-title">3D Product Segmentation Map</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title">Segment Comparison</div>', unsafe_allow_html=True)
        if summary.empty:
            st.info("필터 조건에 맞는 세그먼트가 없습니다.")
        else:
            chart = px.bar(
                summary,
                x="segment",
                y="products",
                color="segment",
                color_discrete_map=CLUSTER_COLORS,
                text="products",
            )
            chart.update_layout(
                height=300,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                font={"color": "#f5f7fb"},
                xaxis={"title": None, "tickangle": -14},
                yaxis={"title": "Products", "gridcolor": "rgba(255,255,255,0.12)"},
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

    st.markdown('<div class="section-title">Filtered Product Data</div>', unsafe_allow_html=True)
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


def render_teammate_tab(number: int) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Analysis Track {number}</h1>
            <p>팀원이 자신의 분석 주제, 핵심 시각화, 주요 지표, 실무 적용 방안을 채우는 공간입니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="placeholder">
            <strong>Recommended structure</strong><br>
            1. 분석 질문과 데이터 설명<br>
            2. 핵심 지표 3-4개<br>
            3. 메인 시각화<br>
            4. 인사이트와 비즈니스 액션<br>
            5. 한계점과 다음 실험
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    df = load_segments()
    selected_segments, price_range, gap_range = render_sidebar(df)
    filtered = apply_filters(df, selected_segments, price_range, gap_range)

    tab1, tab2, tab3 = st.tabs(["1. Product Segmentation", "2. Teammate Analysis", "3. Teammate Analysis"])
    with tab1:
        render_first_tab(filtered)
    with tab2:
        render_teammate_tab(2)
    with tab3:
        render_teammate_tab(3)


if __name__ == "__main__":
    main()
