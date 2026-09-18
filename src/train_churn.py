"""
Trains the customer retention prediction model.

Target: retained = 1 (1.2% minority class).
Evaluating PR-AUC on retention measures the model's true capability
to spot rare, high-value repeat customers rather than defaulting to churn.
"""
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Non-leaky feature set
FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_freight",
    "avg_items_per_order",
    "avg_delivery_delay",
    "late_delivery_rate",
    "avg_review_score",
    "min_review_score",
]


def get_models():
    """Baseline + gradient boosting models."""
    models = {
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        print("[warn] xgboost not installed - skipping")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1
        )
    except ImportError:
        print("[warn] lightgbm not installed - skipping")
    return models


def evaluate(name, model, X_te, y_te):
    # Predict probability of RETAINING (class 1)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
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


def main(apply_smote=True):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("outputs/customer_features.csv")
    
    # FLIP TARGET: Predict retention (the 1.2% minority class)
    df["retained"] = (df["churned"] == 0).astype(int)
    X, y = df[FEATURES], df["retained"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE

            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            print(f"SMOTE applied on 'retained' minority class -> training rows: {len(X_tr):,}")
        except ImportError:
            print("[warn] imbalanced-learn not installed - training without SMOTE")

    results, fitted = [], {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        r = evaluate(name, model, X_te, y_te)
        results.append(r)
        print(
            f"{name:20s} PR-AUC={r['pr_auc']}  ROC-AUC={r['roc_auc']}  "
            f"F1={r['f1']}  Recall={r['recall']}"
        )

    best = max(results, key=lambda r: r["pr_auc"])
    joblib.dump(
        {"model": fitted[best["model"]], "features": FEATURES, "target": "retained"},
        "models/churn_model.joblib",
    )
    json.dump(results, open("outputs/churn_metrics.json", "w"), indent=2)
    print(
        f"\nBest model for Retention: {best['model']} (PR-AUC {best['pr_auc']}, "
        f"ROC-AUC {best['roc_auc']}) -> models/churn_model.joblib"
    )


if __name__ == "__main__":
    main()