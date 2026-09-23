"""
Feature engineering for the Smart E-Commerce Customer Analytics & Retention Engine.

Turns the raw Olist multi-table dataset into ONE ROW PER CUSTOMER, with
RFM + delivery + review features, plus the churn label the model will learn.
"""
import os
import pandas as pd

RAW = "data"


def load_tables(raw_dir=RAW):
    return {
        "orders": pd.read_csv(f"{raw_dir}/olist_orders_dataset.csv"),
        "customers": pd.read_csv(f"{raw_dir}/olist_customers_dataset.csv"),
        "items": pd.read_csv(f"{raw_dir}/olist_order_items_dataset.csv"),
        "payments": pd.read_csv(f"{raw_dir}/olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(f"{raw_dir}/olist_order_reviews_dataset.csv"),
    }


def build_order_level(t):
    """One row per order, enriched with customer identity, spend, delivery, review."""
    orders = t["orders"].copy()
    for c in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    # Only completed orders: cancelled/unavailable orders are not a churn signal.
    orders = orders[orders["order_status"] == "delivered"]

    # IMPORTANT: use customer_unique_id (the real person), not customer_id,
    # which Olist regenerates for every single order.
    orders = orders.merge(
        t["customers"][["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # order_items is one row PER ITEM, so aggregate to order level before joining.
    spend = (
        t["items"]
        .groupby("order_id")
        .agg(
            order_value=("price", "sum"),
            freight=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
        )
        .reset_index()
    )
    orders = orders.merge(spend, on="order_id", how="left")

    rev = (
        t["reviews"]
        .sort_values("review_creation_date")
        .groupby("order_id")["review_score"]
        .first()
        .reset_index()
    )
    orders = orders.merge(rev, on="order_id", how="left")

    # Positive delay = delivered later than promised.
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days
    orders["is_late"] = (orders["delivery_delay_days"] > 0).astype(int)

    return orders


def build_customer_features(order_level, outcome_days=180):
    """
    Collapse order-level rows into one row per customer.

    LEAKAGE WARNING:
    Split the timeline into two windows:
      * OBSERVATION window  (start .. cutoff) -> all features computed here
      * OUTCOME window      (cutoff .. snapshot) -> label computed here
    A customer is churned if they bought during observation but placed NO order
    in the outcome window. Features never see the outcome window, so recency is
    measured up to the cutoff only and can no longer give the answer away.
    """
    snapshot = order_level["order_purchase_timestamp"].max()
    cutoff = snapshot - pd.Timedelta(days=outcome_days)

    obs = order_level[order_level["order_purchase_timestamp"] < cutoff]
    out = order_level[order_level["order_purchase_timestamp"] >= cutoff]
    active_later = set(out["customer_unique_id"].unique())

    # Features are built ONLY from the observation window
    g = obs.groupby("customer_unique_id")

    f = pd.DataFrame(
        {
            # --- RFM core --- (recency measured up to the CUTOFF, not the snapshot)
            "recency_days": (cutoff - g["order_purchase_timestamp"].max()).dt.days.clip(lower=0),
            "frequency": g["order_id"].nunique(),
            "monetary": g["order_value"].sum(),
            # --- spend shape & ratios ---
            "avg_order_value": g["order_value"].mean(),
            "total_freight": g["freight"].sum(),
            "freight_ratio": (g["freight"].sum() / (g["order_value"].sum() + 1e-5)).round(4),
            "avg_items_per_order": g["n_items"].mean(),
            # --- delivery experience ---
            "avg_delivery_delay": g["delivery_delay_days"].mean(),
            "late_delivery_rate": g["is_late"].mean(),
            # --- satisfaction ---
            "avg_review_score": g["review_score"].mean(),
            "min_review_score": g["review_score"].min(),
            "review_count": g["review_score"].count(),
            # --- tenure ---
            "customer_lifespan_days": (
                g["order_purchase_timestamp"].max() - g["order_purchase_timestamp"].min()
            ).dt.days,
        }
    )

    # Churned = active in observation window, but placed no order afterwards.
    f["churned"] = (~f.index.isin(active_later)).astype(int)
    f["is_one_time_buyer"] = (f["frequency"] == 1).astype(int)

    # Fill missing values
    for col in ["avg_review_score", "min_review_score"]:
        f[col] = f[col].fillna(f[col].median())
    f["avg_delivery_delay"] = f["avg_delivery_delay"].fillna(0)
    f["late_delivery_rate"] = f["late_delivery_rate"].fillna(0)
    f["monetary"] = f["monetary"].fillna(0)
    f["avg_order_value"] = f["avg_order_value"].fillna(0)
    f["total_freight"] = f["total_freight"].fillna(0)

    return f.reset_index()


def main():
    os.makedirs("outputs", exist_ok=True)
    t = load_tables()
    ol = build_order_level(t)
    feats = build_customer_features(ol)

    ol.to_csv("outputs/order_level.csv", index=False)
    feats.to_csv("outputs/customer_features.csv", index=False)

    print(f"order-level rows   : {len(ol):,}")
    print(f"customers          : {len(feats):,}")
    print(f"churn rate         : {feats['churned'].mean():.1%}")
    print(f"retained rate      : {(1 - feats['churned'].mean()):.1%}")
    print(f"one-time buyer rate: {feats['is_one_time_buyer'].mean():.1%}")


if __name__ == "__main__":
    main()