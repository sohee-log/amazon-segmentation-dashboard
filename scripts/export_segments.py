from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "user_segments.csv"
METRICS_PATH = REPO_ROOT / "data" / "model_metrics.csv"
SAMPLE_PATH = REPO_ROOT / "data" / "sample_user_segments.csv"
SAMPLE_ROWS = 40


def clean_money_percent(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in ["discounted_price", "actual_price", "discount_percentage", "rating_count"]:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned["discount_percentage"] = cleaned["discount_percentage"] / 100
    cleaned["rating_count"] = cleaned["rating_count"].fillna(cleaned["rating_count"].median())
    cleaned["rating"] = cleaned["rating"].fillna(cleaned["rating"].median())
    return cleaned


def sentiment(text: object) -> float:
    return TextBlob(str(text)).sentiment.polarity


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset"))
    csv_files = sorted(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    amazon = pd.read_csv(csv_files[0])
    model_df = clean_money_percent(amazon)
    model_df["sentiment_score"] = model_df["review_content"].apply(sentiment)
    model_df["log_actual_price"] = np.log1p(model_df["actual_price"])
    model_df["log_rating_count"] = np.log1p(model_df["rating_count"])
    model_df["rating_gap"] = model_df["rating"] / 5.0 - (model_df["sentiment_score"] + 1) / 2.0

    candidate_features = [
        "discount_percentage",
        "sentiment_score",
        "rating_gap",
        "log_actual_price",
        "log_rating_count",
    ]
    cluster_base = model_df[candidate_features].dropna()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cluster_base)

    rows = []
    for k in range(2, 7):
        km_lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x_scaled)
        gm_lab = GaussianMixture(n_components=k, random_state=42).fit_predict(x_scaled)
        for algo, lab in [("KMeans", km_lab), ("GMM", gm_lab)]:
            rows.append(
                {
                    "algo": algo,
                    "k": k,
                    "silhouette": round(silhouette_score(x_scaled, lab), 3),
                    "davies_bouldin": round(davies_bouldin_score(x_scaled, lab), 3),
                    "calinski_harabasz": round(calinski_harabasz_score(x_scaled, lab), 1),
                }
            )
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(METRICS_PATH, index=False)

    final_k = 3
    km3 = KMeans(n_clusters=final_k, random_state=42, n_init=10).fit(x_scaled)
    gm3 = GaussianMixture(n_components=final_k, random_state=42).fit(x_scaled)
    candidates = {"KMeans": km3.labels_, "GMM": gm3.predict(x_scaled)}
    final_algo = max(candidates, key=lambda a: silhouette_score(x_scaled, candidates[a]))
    labels = candidates[final_algo]

    pca3 = PCA(n_components=3)
    coords3 = pca3.fit_transform(x_scaled)

    result = model_df.loc[cluster_base.index].copy()
    result["x"] = coords3[:, 0]
    result["y"] = coords3[:, 1]
    result["z"] = coords3[:, 2]
    result["cluster"] = labels.astype(str)
    result["segment_name"] = result["cluster"].map(
        {
            "0": "관심 필요 상품",
            "1": "프리미엄 베스트셀러",
            "2": "가성비 호평 상품",
        }
    )
    result["product_id"] = result.get("product_id", pd.Series(result.index, index=result.index)).astype(str)
    result["product_name"] = result.get("product_name", pd.Series("", index=result.index)).astype(str)
    result["category"] = result.get("category", pd.Series("", index=result.index)).astype(str)
    result["model_algo"] = final_algo
    result["pca_total_variance"] = pca3.explained_variance_ratio_.sum()
    result["pca1_variance"] = pca3.explained_variance_ratio_[0]
    result["pca2_variance"] = pca3.explained_variance_ratio_[1]
    result["pca3_variance"] = pca3.explained_variance_ratio_[2]

    columns = [
        "product_id",
        "product_name",
        "category",
        "cluster",
        "segment_name",
        "x",
        "y",
        "z",
        "discount_percentage",
        "sentiment_score",
        "rating_gap",
        "actual_price",
        "rating",
        "rating_count",
        "log_actual_price",
        "log_rating_count",
        "model_algo",
        "pca_total_variance",
        "pca1_variance",
        "pca2_variance",
        "pca3_variance",
    ]
    export = result[columns]
    export.to_csv(OUT_PATH, index=False)

    # 대시보드는 user_segments.csv가 없으면 이 샘플로 폴백하므로 스키마가 같아야 한다.
    export.groupby("cluster", group_keys=False).head(SAMPLE_ROWS // final_k).to_csv(SAMPLE_PATH, index=False)

    print(f"Wrote {len(export):,} rows to {OUT_PATH}")
    print(f"Wrote sample rows to {SAMPLE_PATH}")
    print(f"Final algorithm: {final_algo}")


if __name__ == "__main__":
    main()
