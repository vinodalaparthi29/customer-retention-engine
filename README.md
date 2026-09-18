# Smart E-Commerce Customer Analytics & Retention Engine

Predicts which customers are likely to stop buying, and recommends products to keep
them engaged — both returned from a single customer lookup.

Built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the Olist dataset from Kaggle and put the CSVs directly in `data/`:

```
data/
  olist_orders_dataset.csv
  olist_customers_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_products_dataset.csv
```

## Run the pipeline (in this order)

```bat
python src\features.py       :: builds outputs\customer_features.csv
python src\train_churn.py    :: trains models -> models\churn_model.joblib
python src\recommender.py    :: builds models\recommender.joblib
```

`train_churn.py` prints PR-AUC, F1, precision and recall for each model and saves
them to `outputs\churn_metrics.json`. **Copy those numbers into the Results slide.**

## Run the app

Terminal 1:
```bat
uvicorn src.api:app --reload
```
Terminal 2:
```bat
streamlit run src\dashboard.py
```

API docs: http://127.0.0.1:8000/docs

---

## How it works

| Stage | File | What it does |
|---|---|---|
| Data pipeline | `src/features.py` | Joins the Olist tables, aggregates items to order level, builds one row per customer |
| Churn model | `src/train_churn.py` | Logistic Regression + XGBoost, SMOTE on the training split only |
| Recommender | `src/recommender.py` | Item-based collaborative filtering over customer × category |
| API | `src/api.py` | `GET /predict/{customer_id}` → churn probability, risk band, recommendations |
| Dashboard | `src/dashboard.py` | Streamlit UI that calls the API |

### Two things worth knowing

**1. Use `customer_unique_id`, not `customer_id`.**
Olist generates a fresh `customer_id` for every order. Only `customer_unique_id`
identifies the same person across orders — essential for anything about retention.

**2. Churn labelling is time-aware (avoids target leakage).**
The naive label — "no purchase in the last 180 days" — is computed from the same
snapshot date as the `recency_days` feature, which makes recency a perfect mirror of
the answer. That produces PR-AUC near 1.0 and a model that has learned nothing.

Instead the timeline is split:

```
|<---- observation window ---->|<---- outcome window (180d) ---->|
         features from here            label from here
```

A customer is churned if they were active during observation but placed no order in
the outcome window. Features never see the outcome window.

---

## Next steps

- Hyper-parameter tuning; threshold selection to prioritise recall
- SHAP explainability so each churn score shows its drivers
- Product-level (not just category-level) recommendations + precision@k
- Dockerise and deploy
