"""세그먼트 이름을 군집의 실제 통계에서 유도한다.

K-Means 의 클러스터 번호는 초기화에 따라 임의로 붙는 값이라 이름을 번호에
고정({"0": "관심 필요 상품", ...})하면 재학습 때 조용히 어긋난다. 실제로
0번과 2번의 이름이 뒤바뀐 채 배포된 적이 있어, 이름을 번호가 아니라
군집의 성격에서 유도하도록 한 곳에 모았다.
"""
from __future__ import annotations

import pandas as pd


PREMIUM = "프리미엄 베스트셀러"
ATTENTION = "관심 필요 상품"
VALUE = "가성비 호평 상품"


def assign_segment_names(frame: pd.DataFrame) -> dict:
    """cluster -> 세그먼트명 매핑을 군집 통계로 유도한다.

    규칙:
      1. 평균 정가가 가장 높은 군집 → 프리미엄 베스트셀러
      2. 나머지 중 별점-감정 괴리가 큰 군집 → 관심 필요 상품 (별점 인플레이션)
      3. 나머지 → 가성비 호평 상품 (별점이 보수적이고 텍스트 감정이 좋음)

    frame 에는 cluster, actual_price, rating_gap 컬럼이 있어야 한다.
    """
    required = {"cluster", "actual_price", "rating_gap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"이름 유도에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")

    stats = frame.groupby("cluster").agg(
        price=("actual_price", "mean"),
        gap=("rating_gap", "mean"),
    )
    if len(stats) != 3:
        raise ValueError(f"3개 군집을 전제로 한 규칙입니다. 현재 군집 수: {len(stats)}")

    premium = stats["price"].idxmax()
    rest = stats.drop(index=premium)
    attention = rest["gap"].idxmax()
    value = rest["gap"].idxmin()
    return {premium: PREMIUM, attention: ATTENTION, value: VALUE}
