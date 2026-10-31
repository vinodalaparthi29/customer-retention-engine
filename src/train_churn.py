"""
Trains the customer retention prediction models and evaluates them.

Target: retained = 1 (customer placed a repeat order in the outcome window), ~1.2% of customers.
Imbalance handling: class_weight="balanced" for Logistic Regression only.
XGBoost and LightGBM use default class weighting.
No probability calibration and no resampling (SMOTE) is applied.

Outputs
  models/churn_model.joblib        best model by PR-AUC
  outputs/churn_metrics.json       per-model metrics (unchanged format)
  outputs/lift_metrics.json        NEW: top-K% capture / lift table for every model
  outputs/top_decile_lift.png      NEW: chart of repeat buyers captured by the top-scored customers
"""
import json
import os

import joblib
import numpy as np
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

# Hard-label metrics (F1, precision, recall, confusion matrix) use this probability threshold.
EVAL_THRESHOLD = 0.10
# Share of top-scored customers to examine for the lift table.
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)


def get_models():
    models = {}

    # Logistic Regression (the only model with class weighting)
    models["LogisticRegression"] = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    # XGBoost (default class weighting)
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        pass

    # LightGBM (default class weighting)
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        pass

    return models


def evaluate(name, model, X_te, y_te, eval_threshold=EVAL_THRESHOLD):
    """Predicts probability of RETAINING (class 1) and evaluates metrics."""
    proba = model.predict_proba(X_te)[:, 1]

    # Hard metrics at the custom decision threshold
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


def lift_table(y_true, proba, fractions=TOP_FRACTIONS):
    """
    Rank test customers by predicted score (best first) and report, for the top K%,
    what share of ALL repeat buyers is captured. A random ranking captures about K%.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    total_pos = int(y_true.sum())
    rows = []
    for k in fractions:
        n = max(1, int(round(len(y_true) * k)))
        caught = int(y_true[order[:n]].sum())
        capture = caught / total_pos if total_pos else 0.0
        rows.append(
            {
                "top_fraction": k,
                "customers_contacted": n,
                "repeat_buyers_caught": caught,
                "repeat_buyers_total": total_pos,
                "capture_rate": round(capture, 4),
                "random_capture_rate": k,
                "lift": round(capture / k, 3),
            }
        )
    return rows


def bootstrap_capture_ci(y_true, proba, k=0.10, n_boot=1000, seed=42):
    """95% bootstrap interval for the capture rate of the top-K% (test set has few positives)."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n = len(y_true)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pb = y_true[idx], proba[idx]
        if yt.sum() == 0:
            continue
        top = np.argsort(-pb, kind="stable")[: max(1, int(round(n * k)))]
        rates.append(yt[top].sum() / yt.sum())
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def save_lift_chart(rows, model_name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"Top {int(r['top_fraction'] * 100)}%" for r in rows]
    model_vals = [r["capture_rate"] * 100 for r in rows]
    random_vals = [r["random_capture_rate"] * 100 for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    ax.bar(x - 0.2, model_vals, 0.4, label=f"{model_name}", color="#2a7de1")
    ax.bar(x + 0.2, random_vals, 0.4, label="Random targeting", color="#9aa6b8")
    for xi, v in zip(x, model_vals):
        ax.text(xi - 0.2, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of all repeat buyers reached")
    ax.set_title("Repeat buyers reached by contacting the top-scored customers")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
    print(
        f"Rows: {len(df)} | train: {len(X_tr)} | test: {len(X_te)} | "
        f"test repeat buyers: {int(y_te.sum())} ({y_te.mean():.2%}) "
        f"-> random-guess PR-AUC baseline = {y_te.mean():.4f}"
    )

    results, fitted, lifts = [], {}, {}
    for name, model in get_models().items():
        # Fit on training data only
        model.fit(X_tr, y_tr)
        fitted[name] = model

        # Evaluate on the untouched test holdout
        r = evaluate(name, model, X_te, y_te)
        results.append(r)
        print(
            f"{name:20s} PR-AUC={r['pr_auc']}  ROC-AUC={r['roc_auc']}  "
            f"F1={r['f1']}  Recall={r['recall']}  (threshold={EVAL_THRESHOLD})"
        )

        proba = model.predict_proba(X_te)[:, 1]
        lifts[name] = lift_table(y_te, proba)

    best = max(results, key=lambda r: r["pr_auc"])
    joblib.dump(
        {"model": fitted[best["model"]], "features": available_features, "target": "retained"},
        "models/churn_model.joblib",
    )
    json.dump(results, open("outputs/churn_metrics.json", "w"), indent=2)

    # ---- NEW: top-K% capture / lift ----
    best_proba = fitted[best["model"]].predict_proba(X_te)[:, 1]
    lo, hi = bootstrap_capture_ci(y_te, best_proba, k=0.10)
    json.dump(
        {
            "test_customers": int(len(y_te)),
            "test_repeat_buyers": int(y_te.sum()),
            "best_model": best["model"],
            "top10_capture_95ci": [lo, hi],
            "tables": lifts,
        },
        open("outputs/lift_metrics.json", "w"),
        indent=2,
    )
    save_lift_chart(lifts[best["model"]], best["model"], "outputs/top_decile_lift.png")

    print(f"\nTop-K% capture for {best['model']} (random targeting captures about K%):")
    for r in lifts[best["model"]]:
        print(
            f"  Top {r['top_fraction']:.0%} ({r['customers_contacted']} customers): caught "
            f"{r['repeat_buyers_caught']}/{r['repeat_buyers_total']} = {r['capture_rate']:.1%}  "
            f"(lift {r['lift']}x)"
        )
    print(f"  Top 10% capture, 95% bootstrap interval: {lo:.1%} to {hi:.1%}")

    print(
        f"\nBest model for Retention: {best['model']} (PR-AUC {best['pr_auc']}, "
        f"ROC-AUC {best['roc_auc']}) -> models/churn_model.joblib"
    )


if __name__ == "__main__":
    main()
