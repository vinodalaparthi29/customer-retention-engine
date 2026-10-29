# 🛒 Smart E-Commerce Customer Retention Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vinodalaparthi29-customer-retention-engine-srcdashboard-ngoici.streamlit.app/)

🔗 **Live Streamlit Dashboard:** [customer-retention-engine.streamlit.app](https://vinodalaparthi29-customer-retention-engine-srcdashboard-ngoici.streamlit.app/)  
⚡ **FastAPI Backend (Render):** `https://customer-retention-engine.onrender.com`

A full-stack, end-to-end Machine Learning solution designed for **e-commerce customer retention & repeat buyer prediction** using the Brazilian Olist E-Commerce dataset (~100k transactions).

The system addresses extreme class imbalance (**98.8% one-time buyers vs. 1.2% repeat buyers**) by leveraging feature engineering, leakage-free temporal splitting, propensity ranking, and a deployed FastAPI backend connected to an interactive Streamlit dashboard.

---

## 🚀 Key Features

* **Non-Leaky Temporal Windowing:** Splits customer history into Observation and Outcome windows to prevent data leakage in recency features.
* **RFM + Behavioral Signals:** Engineers recency, frequency, monetary value, freight ratios, delivery delay rates, and review scores.
* **Propensity Ranking & Risk Tiers:** Ranks customers into actionable marketing tiers (*High Retention Potential*, *Moderate Retention Potential*, *High Churn Risk*).
* **Production Architecture:** Decoupled architecture featuring a **FastAPI** backend on Render and a **Streamlit** dashboard frontend.

---

## 📊 Model Performance & Metric Selection

On an extreme 1.2% minority baseline, standard accuracy (98.8%) and static 0.50 decision thresholds fail. Models were evaluated primarily on **PR-AUC (Precision-Recall Area Under Curve)** to capture true minority signal:

| Model | PR-AUC (Primary) | ROC-AUC | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Selected)** | **0.0312** | **0.5800** | **1.53%** | **8.50%** |
| **Logistic Regression** | 0.0276 | 0.5811 | 100.0% | 1.37% |
| **LightGBM** | 0.0241 | 0.5662 | 2.29% | 3.64% |

*Note: **XGBoost** achieved a PR-AUC of 0.0312, nearly **2.6x higher than random guessing baseline (0.0120)**.*

---

## 🛠️ Tech Stack & Architecture

* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy
* **Backend API:** FastAPI, Uvicorn, Joblib, Pydantic (Deployed on Render)
* **Frontend UI:** Streamlit (Deployed on Streamlit Community Cloud)

---

## 📂 Repository Structure

```text
├── data/                         # Raw Olist CSV files (Git-ignored)
├── models/                       # Trained model artifact (churn_model.joblib)
├── outputs/                      # Processed features & performance metrics
├── src/
│   ├── api.py                    # FastAPI backend endpoints
│   ├── dashboard.py              # Streamlit dashboard script
│   ├── features.py               # Feature engineering pipeline
│   ├── recommender.py            # Category recommendation logic
│   └── train_churn.py            # Model training & benchmark script
├── requirements.txt              # Dependencies
└── README.md

HOW TO RUN LOCALLY

    1. Clone Repository & Install Dependencies
    
    git clone [https://github.com/vinodalaparthi29/customer-retention-engine.git](https://github.com/vinodalaparthi29/customer-retention-engine.git)
    cd customer-retention-engine
    pip install -r requirements.txt
    2. Run Pipeline & Train Model
    Bash
    python src/features.py
    python src/train_churn.py
    3. Launch FastAPI Backend
    Bash
    uvicorn src/api.py:app --reload
    4. Launch Streamlit Dashboard
    Bash
    streamlit run src/dashboard.py
