# Amazon Product Segmentation Dashboard

Streamlit dashboard for presenting a three-person data analysis project. Tab 1 is a polished Amazon product segmentation view based on price, review sentiment, rating gap, price level, and review count. Tabs 2 and 3 are placeholders for teammates.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Generate the Notebook Result

The dashboard reads:

```text
data/user_segments.csv
```

This file is generated from the Kaggle dataset used in the notebook:

```powershell
python export_segments.py
```

The script follows the final notebook logic:

- Download `karkavelrajaj/amazon-sales-dataset`
- Clean price, discount, rating, and rating count columns
- Create `sentiment_score`, `rating_gap`, `log_actual_price`, `log_rating_count`
- Compare K-Means and GMM
- Use final `k=3`
- Export PCA 3D coordinates as `x`, `y`, `z`

Required dashboard columns:

```text
product_id, product_name, cluster, segment_name, x, y, z
```

Recommended columns:

```text
actual_price, discount_percentage, sentiment_score, rating_gap, rating, rating_count, category
```

안녕하세요!
