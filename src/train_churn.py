"""
Trains the customer retention prediction model with probability calibration.
"""
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_freight",
    "freight_ratio",
    "avg_items_per_order",
    "avg_delivery_delay",
    "late_delivery_rate",
    "avg_review_score",
    "min_review_score",
    "review_count",
]


def get_models():
    """Baseline + gradient boosting models wrapped in CalibratedClassifierCV."""
    models = {}
    
    # Logistic Regression
    lr_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    models["LogisticRegression"] = CalibratedClassifierCV(estimator=lr_pipe, cv=5, method="sigmoid")

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,  # Prevents joblib core warning on Windows
        )
        models["XGBoost"] = CalibratedClassifierCV(estimator=xgb, cv=5, method="sigmoid")
    except ImportError:
        print("[warn] xgboost not installed")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            random_state=42, 
            verbose=-1,
            n_jobs=1,
        )
        models["LightGBM"] = CalibratedClassifierCV(estimator=lgb, cv=5, method="sigmoid")
    except ImportError:
        print("[warn] lightgbm not installed")

    return models


def evaluate(name, model, X_te, y_te, eval_threshold=0.10):
    """Predicts probability of RETAINING (class 1) and evaluates metrics."""
    proba = model.predict_proba(X_te)[:, 1]
    
    # Evaluate hard metrics at custom decision threshold (e.g. 10%)
    pred = (proba >= eval_threshold).astype(int)
    
    m = {
        "model": name,
        "pr_auc": round(average_precision_score(y_te, proba), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "f1": round(f1_score(y_te, pred, zero_division=0), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred, zero_division=0), 4),
    }
    m["confusion_matrix"] = confusion_matrix(y_te, pred).tolist()
    return m


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("outputs/customer_features.csv")
    
    # Target: Predict retention (minority class)
    df["retained"] = (df["churned"] == 0).astype(int)
    
    # Ensure all features exist in DataFrame
    available_features = [col for col in FEATURES if col in df.columns]
    X, y = df[available_features], df["retained"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results, fitted = [], {}
    for name, calibrated_model in get_models().items():
        # Fit calibrated model strictly on training data
        calibrated_model.fit(X_tr, y_tr)
        
        fitted[name] = calibrated_model
        
        # Evaluate strictly on untouched test holdout
        r = evaluate(name, calibrated_model, X_te, y_te)
        results.append(r)
        print(
            f"{name:20s} PR-AUC={r['pr_auc']}  ROC-AUC={r['roc_auc']}  "
            f"F1={r['f1']}  Recall={r['recall']}"
        )

    best = max(results, key=lambda r: r["pr_auc"])
    joblib.dump(
        {"model": fitted[best["model"]], "features": available_features, "target": "retained"},
        "models/churn_model.joblib",
    )
    json.dump(results, open("outputs/churn_metrics.json", "w"), indent=2)
    print(
        f"\nBest model for Retention: {best['model']} (PR-AUC {best['pr_auc']}, "
        f"ROC-AUC {best['roc_auc']}) -> models/churn_model.joblib"
    )


if __name__ == "__main__":
    main()