"""
모델 2: 리뷰 신뢰도 점수 - 학습 및 아티팩트 생성 스크립트
teammate의 export_segments.py와 동일한 패턴을 따름.

실행:
    python export_trust_model.py

생성물:
    models/trust_pipeline.joblib      -> TF-IDF + Ridge 파이프라인 (학습된 모델)
    models/trust_meta.json            -> 가중치, 성능 지표 등 메타데이터
    data/review_trust_samples.csv     -> 대시보드용 샘플 리뷰 + 신뢰도 점수

주의:
    trust_score는 실제 정답 라벨이 없는 자체 정의 지표입니다.
    여기서 계산되는 R²/MAE는 "텍스트로부터 이 지표를 얼마나 잘 재현하는가"를
    의미하며, "실제 리뷰 신뢰도를 얼마나 잘 맞히는가"가 아닙니다.
"""
import json
import os

import joblib
import kagglehub
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

nltk.download("vader_lexicon", quiet=True)

# ------------------------------------------------------------
# 1. 데이터 로드 (teammate와 동일한 Kaggle 데이터셋 사용 -> 대시보드 정합성 유지)
# ------------------------------------------------------------
path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
csv_name = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_name))

df["review_content"] = df["review_content"].astype(str)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["review_content", "rating"]).reset_index(drop=True)

# ------------------------------------------------------------
# 2. VADER 감성분석 + 피처 엔지니어링
# ------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
df["vader_compound"] = df["review_content"].apply(
    lambda t: sia.polarity_scores(str(t))["compound"]
)
df["emotion_strength"] = df["vader_compound"].abs()

POS_KW = {"great", "excellent", "amazing", "perfect", "love", "best",
          "fantastic", "outstanding", "wonderful", "superb", "recommend"}
NEG_KW = {"bad", "terrible", "awful", "horrible", "worst", "broken",
          "waste", "disappointed", "poor", "defective", "scam", "fake"}


def extract_features(text):
    words = str(text).split()
    return {
        "review_length": len(words),
        "word_diversity": len(set(w.lower() for w in words)) / max(len(words), 1),
        "pos_keyword_count": sum(1 for w in words if w.lower() in POS_KW),
        "neg_keyword_count": sum(1 for w in words if w.lower() in NEG_KW),
    }


feat_df = df["review_content"].apply(extract_features).apply(pd.Series)
df = pd.concat([df, feat_df], axis=1)

# ------------------------------------------------------------
# 3. train/test 분리 후 trust_score 산출
#    (스케일러·가중치는 train 데이터 기준으로만 fit -> 누수 방지)
# ------------------------------------------------------------
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

trust_cols = ["vader_compound", "emotion_strength", "word_diversity",
              "review_length", "pos_keyword_count", "neg_keyword_count"]

corrs = df.loc[train_idx, trust_cols + ["rating"]].corr()["rating"].drop("rating")
weights = (corrs / corrs.abs().sum()).to_dict()

scaler_trust = MinMaxScaler()
scaler_trust.fit(df.loc[train_idx, trust_cols])
trust_norm = pd.DataFrame(
    scaler_trust.transform(df[trust_cols]), columns=trust_cols, index=df.index
)

df["raw_trust"] = sum(trust_norm[c] * weights[c] for c in trust_cols)
t_min = df.loc[train_idx, "raw_trust"].min()
t_max = df.loc[train_idx, "raw_trust"].max()
df["trust_score"] = np.clip(((df["raw_trust"] - t_min) / (t_max - t_min)) * 100, 0, 100)

# ------------------------------------------------------------
# 4. TF-IDF + Ridge 학습 (리뷰 원문 -> trust_score)
# ------------------------------------------------------------
X_train, y_train = df.loc[train_idx, "review_content"], df.loc[train_idx, "trust_score"]
X_test, y_test = df.loc[test_idx, "review_content"], df.loc[test_idx, "trust_score"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=800, ngram_range=(1, 2),
                               stop_words="english", sublinear_tf=True)),
    ("ridge", Ridge(alpha=0.5)),
])
pipeline.fit(X_train, y_train)

y_pred = np.clip(pipeline.predict(X_test), 0, 100)
metrics = {
    "r2": float(r2_score(y_test, y_pred)),
    "mae": float(np.mean(np.abs(y_test.values - y_pred))),
    "mse": float(mean_squared_error(y_test, y_pred)),
}

# ------------------------------------------------------------
# 5. 아티팩트 저장
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

joblib.dump(pipeline, "models/trust_pipeline.joblib")

meta = {
    "weights": weights,
    "metrics": metrics,
    "note": (
        "trust_score는 정답 라벨이 없는 자체 정의 지표입니다. "
        "위 metrics는 텍스트로부터 이 지표를 재현하는 성능을 의미하며, "
        "실제 리뷰의 신뢰도(진위)를 검증하는 정확도가 아닙니다."
    ),
}
with open("models/trust_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

sample_cols = [c for c in
               ["product_id", "product_name", "review_content", "rating",
                "trust_score", "vader_compound", "category"]
               if c in df.columns]
df[sample_cols].sample(n=min(500, len(df)), random_state=42) \
    .to_csv("data/review_trust_samples.csv", index=False)

print("완료:")
print(f"  R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.2f}")
print("  -> models/trust_pipeline.joblib")
print("  -> models/trust_meta.json")
print("  -> data/review_trust_samples.csv")