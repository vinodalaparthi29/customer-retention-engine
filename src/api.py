"""
FastAPI Backend for the Smart Retention Engine.
Serves real-time retention probabilities and product recommendations.
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Retention Engine API")

# Load model artifact and feature data on startup
model_data = joblib.load("models/churn_model.joblib")
model = model_data["model"]
features_list = model_data["features"]

customer_df = pd.read_csv("outputs/customer_features.csv").set_index("customer_unique_id")


class PredictionRequest(BaseModel):
    customer_unique_id: str


@app.get("/")
def root():
    return {"status": "online", "message": "Customer Retention Engine API Running"}


@app.post("/predict")
def predict_retention(req: PredictionRequest):
    cid = req.customer_unique_id
    if cid not in customer_df.index:
        raise HTTPException(status_code=404, detail="Customer ID not found in database")

    # Extract customer features
    user_feats = customer_df.loc[[cid]][features_list]
    
    # Model outputs retention probability (class 1 = retained)
    retention_prob = float(model.predict_proba(user_feats)[:, 1][0])
    churn_prob = float(1.0 - retention_prob)

    # Risk Categorization
    if retention_prob >= 0.15:
        risk_level = "High Retention Potential"
    elif retention_prob >= 0.05:
        risk_level = "Moderate Retention Potential"
    else:
        risk_level = "High Churn Risk"

    return {
        "customer_unique_id": cid,
        "retention_probability": round(retention_prob, 4),
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "metrics": {
            "recency_days": int(user_feats["recency_days"].values[0]),
            "frequency": int(user_feats["frequency"].values[0]),
            "monetary": float(user_feats["monetary"].values[0]),
            "avg_review_score": float(user_feats["avg_review_score"].values[0]),
        },
    }