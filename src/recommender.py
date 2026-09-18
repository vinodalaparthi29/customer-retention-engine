"""
Product recommendation module.

Item-based collaborative filtering over the customer x product-category
purchase matrix. Uses cosine similarity from scikit-learn so the project has
no dependency on Scikit-Surprise (which is awkward to install on Windows).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CategoryRecommender:
    """Recommends product categories a customer has not bought yet."""

    def __init__(self):
        self.matrix = None          # customers x categories
        self.similarity = None      # categories x categories
        self.categories = None
        self.popularity = None      # fallback for cold-start customers

    def fit(self, interactions: pd.DataFrame):
        """interactions: columns [customer_unique_id, category, qty]"""
        self.matrix = interactions.pivot_table(
            index="customer_unique_id", columns="category",
            values="qty", aggfunc="sum", fill_value=0)
        self.categories = list(self.matrix.columns)
        self.similarity = pd.DataFrame(
            cosine_similarity(self.matrix.T.values),
            index=self.categories, columns=self.categories)
        self.popularity = self.matrix.sum().sort_values(ascending=False)
        return self

    def recommend(self, customer_id, n=5):
        # Cold start: unseen customer -> most popular categories overall.
        if self.matrix is None or customer_id not in self.matrix.index:
            return [{"category": c, "score": None, "reason": "popular overall"}
                    for c in self.popularity.head(n).index]

        bought = self.matrix.loc[customer_id]
        bought_cats = bought[bought > 0].index
        if len(bought_cats) == 0:
            return [{"category": c, "score": None, "reason": "popular overall"}
                    for c in self.popularity.head(n).index]

        # Score every category by similarity to what this customer already bought.
        scores = self.similarity[bought_cats].mul(bought[bought_cats], axis=1).sum(axis=1)
        scores = scores.drop(labels=bought_cats, errors="ignore")
        scores = scores[scores > 0].sort_values(ascending=False).head(n)

        if scores.empty:
            return [{"category": c, "score": None, "reason": "popular overall"}
                    for c in self.popularity.head(n).index]

        return [{"category": c, "score": round(float(s), 4),
                 "reason": "similar to past purchases"} for c, s in scores.items()]

    # Persist plain DataFrames rather than the object itself. Pickling the class
    # ties the file to the module path it was saved from, which breaks when the
    # API loads it from a different entry point.
    def save(self, path):
        joblib.dump({"matrix": self.matrix,
                     "similarity": self.similarity,
                     "popularity": self.popularity}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        r = cls()
        r.matrix, r.similarity, r.popularity = d["matrix"], d["similarity"], d["popularity"]
        r.categories = list(r.matrix.columns)
        return r


def build_interactions(order_level: pd.DataFrame, items: pd.DataFrame,
                       products: pd.DataFrame) -> pd.DataFrame:
    df = (items.merge(products[["product_id", "product_category_name"]],
                      on="product_id", how="left")
          .merge(order_level[["order_id", "customer_unique_id"]],
                 on="order_id", how="inner"))
    df["category"] = df["product_category_name"].fillna("unknown")
    return (df.groupby(["customer_unique_id", "category"])
            .size().reset_index(name="qty"))


def main():
    order_level = pd.read_csv("outputs/order_level.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")

    interactions = build_interactions(order_level, items, products)
    rec = CategoryRecommender().fit(interactions)
    rec.save("models/recommender.joblib")

    print(f"customers   : {rec.matrix.shape[0]:,}")
    print(f"categories  : {rec.matrix.shape[1]:,}")
    sample = rec.matrix.index[0]
    print(f"\nSample recommendations for {sample}:")
    for r in rec.recommend(sample):
        print("  -", r["category"], r["score"])


if __name__ == "__main__":
    main()
