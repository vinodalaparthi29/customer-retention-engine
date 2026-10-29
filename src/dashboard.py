"""
Smart E-Commerce Customer Analytics & Retention Engine  -  Streamlit dashboard (v2)

Drop-in replacement for src/dashboard.py.

What's new
  * Fast customer finder: type any 3+ characters of an ID, pick from a searchable list of
    short IDs, browse by segment filters, jump to a random / top-value customer, or share a
    link (?cid=...) that opens a specific customer.
  * Customer 360 view: activity KPIs, percentile of the customer vs. everyone, risk tier,
    lift vs. the 1.2% baseline, recommendations and a suggested retention action.
  * Model Insights tab: honest story of the churn -> retention reframing.
  * Handles Render free-tier cold starts (retry + wake-up message).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DEFAULT_API = os.getenv("API_URL", "https://customer-retention-engine.onrender.com")
BASELINE = 0.0117  # random-guess PR-AUC == share of repeat buyers in the test set

st.set_page_config(
    page_title="Retention Engine | Olist",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1250px; }
#MainMenu, footer { visibility: hidden; }

.hero {
  background: linear-gradient(120deg, #1e3a8a 0%, #4f46e5 55%, #7c3aed 100%);
  border-radius: 18px; padding: 26px 32px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; }
.hero p  { margin: 6px 0 0; opacity: .9; font-size: .98rem; }
.chips span {
  display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px;
  padding: 3px 12px; margin: 12px 6px 0 0; font-size: .78rem; font-weight: 500;
}

.card {
  border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px 18px;
  background: rgba(128,128,128,.05); height: 100%;
}
.kpi-label { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; font-weight: 600; }
.kpi-value { font-size: 1.65rem; font-weight: 800; margin-top: 2px; letter-spacing: -.02em; }
.kpi-sub   { font-size: .78rem; opacity: .65; margin-top: 2px; }

.tier {
  display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700;
  font-size: .95rem; color: #fff;
}
.tier-high { background: #16a34a; } .tier-mid { background: #d97706; } .tier-low { background: #dc2626; }

.action {
  border-left: 5px solid #4f46e5; background: rgba(79,70,229,.08);
  padding: 14px 18px; border-radius: 0 12px 12px 0; margin-top: 8px;
}
.reco {
  display: inline-block; padding: 7px 14px; margin: 4px 6px 4px 0; border-radius: 10px;
  background: rgba(124,58,237,.12); border: 1px solid rgba(124,58,237,.35);
  font-weight: 600; font-size: .9rem;
}
.idpill { font-family: ui-monospace, monospace; font-size: .8rem; opacity: .8; word-break: break-all; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Data loading + schema detection (works with whatever column names features.py produced)
# --------------------------------------------------------------------------------------
CANDIDATES = {
    "id": ["customer_unique_id", "customer_id"],
    "recency": ["recency", "recency_days", "days_since_last_order"],
    "frequency": ["frequency", "order_count", "n_orders", "num_orders", "total_orders"],
    "monetary": ["monetary", "total_spent", "total_payment", "payment_value", "total_spend"],
    "review": ["avg_review_score", "review_score", "avg_review", "mean_review_score"],
    "delay": ["avg_delivery_delay", "delivery_delay", "avg_delay_days", "delay_days", "avg_delivery_delay_days"],
    "late": ["late_delivery_rate", "late_rate", "pct_late", "is_late"],
    "freight": ["freight_ratio", "avg_freight_ratio"],
    "state": ["customer_state", "state"],
    "city": ["customer_city", "city"],
    "retained": ["retained", "is_retained", "repeat_buyer"],
    "churned": ["churned", "churn"],
}

CSV_PATHS = [
    "outputs/customer_features.csv",
    "customer_features.csv",
    "../outputs/customer_features.csv",
    str(Path(__file__).resolve().parent.parent / "outputs" / "customer_features.csv"),
]


def detect_schema(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    return {k: next((cols[c] for c in cands if c in cols), None) for k, cands in CANDIDATES.items()}


@st.cache_data(show_spinner=False)
def load_features(uploaded_bytes: bytes | None = None) -> pd.DataFrame | None:
    if uploaded_bytes is not None:
        import io

        return pd.read_csv(io.BytesIO(uploaded_bytes))
    for p in CSV_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def short(cid: str, n: int = 8) -> str:
    return str(cid)[:n]


# --------------------------------------------------------------------------------------
# API client (handles Render cold starts)
# --------------------------------------------------------------------------------------
def api_get(base: str, path: str, timeout: int = 70, retries: int = 2):
    last = None
    for i in range(retries + 1):
        try:
            r = requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)
            if r.status_code == 404:
                return None, "Customer not found by the API."
            r.raise_for_status()
            return r.json(), None
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(1.5 * (i + 1))
    return None, last


@st.cache_data(ttl=600, show_spinner=False)
def predict(base: str, cid: str):
    return api_get(base, f"/predict/{cid}")


@st.cache_data(ttl=120, show_spinner=False)
def health(base: str):
    data, err = api_get(base, "/health", timeout=70, retries=1)
    return (data is not None), err


def pick(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_prediction(resp: dict) -> dict:
    prob = pick(resp, ["retention_probability", "retention_prob", "probability", "propensity", "score", "churn_probability"], 0.0)
    tier = pick(resp, ["risk_band", "retention_tier", "tier", "segment", "risk_tier"], "Unknown")
    recs = pick(resp, ["recommendations", "recommended_categories", "recommended_products", "recs"], []) or []
    if isinstance(recs, dict):
        recs = list(recs.keys())
    recs = [r if isinstance(r, str) else str(pick(r, ["category", "product_category", "name"], r)) for r in recs]
    return {"prob": float(prob), "tier": str(tier), "recs": recs}


def tier_style(tier: str):
    t = tier.lower()
    if "high" in t and "churn" in t:
        return "tier-low", "At risk of dropping off"
    if "high" in t and ("reten" in t or "loyal" in t):
        return "tier-high", "Strong repeat-purchase potential"
    if "low" in t:
        return "tier-high", "Low risk"
    if "mod" in t or "med" in t:
        return "tier-mid", "Worth nurturing"
    if "high" in t:
        return "tier-low", "High risk"
    return "tier-mid", ""


ACTIONS = {
    "tier-high": ("Reward & upsell", "Offer loyalty perks or early access, and cross-sell the recommended categories. This customer is likely to come back, so protect margin and avoid heavy discounts."),
    "tier-mid": ("Nudge with a targeted offer", "Send a personalised email featuring the recommended categories with a time-limited, modest incentive (free shipping or a small voucher) within the next 2 weeks."),
    "tier-low": ("Win-back campaign", "Trigger a win-back sequence: a stronger discount on the recommended categories, plus a proactive support check-in if delivery or review signals were poor."),
}

# --------------------------------------------------------------------------------------
# Visual helpers
# --------------------------------------------------------------------------------------

def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def lift_gauge(prob: float) -> go.Figure:
    lift = prob / BASELINE if BASELINE else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(lift, 2),
            number={"suffix": "×", "font": {"size": 44}},
            title={"text": "Repeat-purchase likelihood vs. average customer", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 5], "tickvals": [0, 1, 2, 3, 4, 5]},
                "bar": {"color": "#4f46e5"},
                "steps": [
                    {"range": [0, 1], "color": "rgba(220,38,38,.28)"},
                    {"range": [1, 2], "color": "rgba(217,119,6,.28)"},
                    {"range": [2, 5], "color": "rgba(22,163,74,.28)"},
                ],
                "threshold": {"line": {"color": "#111", "width": 3}, "thickness": .8, "value": 1},
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def percentile_chart(vals: dict[str, float]) -> go.Figure:
    labels = list(vals.keys())
    pct = [vals[k] for k in labels]
    colors = ["#4f46e5" if p >= 50 else "#9ca3af" for p in pct]
    fig = go.Figure(go.Bar(x=pct, y=labels, orientation="h", marker_color=colors, text=[f"{p:.0f}th" for p in pct], textposition="outside"))
    fig.update_layout(
        height=270, margin=dict(l=10, r=30, t=30, b=10),
        xaxis=dict(range=[0, 115], title="Percentile vs. all customers"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pct_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s < value).mean() * 100) if len(s) else 0.0

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API base URL", DEFAULT_API)
    up = st.file_uploader("Features CSV (only if not bundled)", type="csv")
    ok, err = health(api_url)
    st.markdown(f"**API status:** {'🟢 online' if ok else '🟠 waking up / offline'}")
    if not ok:
        st.caption("The free Render tier sleeps when idle; the first request can take ~30-60 s. Click rerun once it's awake.")
        if st.button("Retry connection"):
            health.clear()
            st.rerun()
    st.divider()
    st.caption("Smart E-Commerce Customer Analytics & Retention Engine\n\nOlist dataset · FastAPI · Streamlit")

df = load_features(up.getvalue() if up else None)

st.markdown(
    """
<div class="hero">
  <h1>🛒 Customer Analytics &amp; Retention Engine</h1>
  <p>Find any customer in seconds, see their activity, how likely they are to buy again, and what to do about it.</p>
  <div class="chips"><span>RFM + delivery + review features</span><span>Logistic Regression + SMOTE</span><span>Item-based collaborative filtering</span><span>FastAPI · Streamlit</span></div>
</div>
""",
    unsafe_allow_html=True,
)

if df is None:
    st.error("Could not find `outputs/customer_features.csv`. Upload it in the sidebar or place it in the repo's `outputs/` folder.")
    st.stop()

S = detect_schema(df)
if S["id"] is None:
    st.error(f"No customer ID column found. Columns available: {list(df.columns)}")
    st.stop()

ID = S["id"]
df[ID] = df[ID].astype(str)

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab_find, tab_overview, tab_model = st.tabs(["🔎 Customer Finder", "📊 Portfolio Overview", "🧪 Model Insights"])

# ========================================== FINDER ======================================
with tab_find:
    qp = st.query_params
    if "selected" not in st.session_state:
        st.session_state.selected = qp.get("cid")

    left, right = st.columns([1, 1.9], gap="large")

    with left:
        st.markdown("#### Find a customer")
        mode = st.radio(
            "How do you want to search?",
            ["Type part of an ID", "Browse by segment", "Quick picks"],
            horizontal=False,
            label_visibility="collapsed",
        )

        candidates = df

        if mode == "Type part of an ID":
            q = st.text_input("Any 3+ characters of the ID", placeholder="e.g. 9a3f  (first, middle or last chars all work)").strip().lower()
            if len(q) >= 3:
                candidates = df[df[ID].str.lower().str.contains(q, regex=False)]
                st.caption(f"{len(candidates):,} match(es)")
            else:
                candidates = df.head(0)
                st.caption("Start typing, or paste a full ID.")

        elif mode == "Browse by segment":
            f = df
            if S["frequency"]:
                lo, hi = int(df[S["frequency"]].min()), int(df[S["frequency"]].max())
                if hi > lo:
                    rng = st.slider("Number of orders", lo, hi, (lo, min(hi, lo + 3)))
                    f = f[f[S["frequency"]].between(*rng)]
            if S["monetary"]:
                m_lo, m_hi = float(df[S["monetary"]].min()), float(df[S["monetary"]].quantile(0.99))
                m = st.slider("Total spend", m_lo, m_hi, (m_lo, m_hi))
                f = f[f[S["monetary"]].between(*m)]
            if S["recency"]:
                r_hi = float(df[S["recency"]].max())
                r = st.slider("Days since last order", 0.0, r_hi, (0.0, r_hi))
                f = f[f[S["recency"]].between(*r)]
            if S["review"]:
                rv = st.slider("Avg review score", 1.0, 5.0, (1.0, 5.0), step=0.5)
                f = f[f[S["review"]].between(*rv)]
            if S["state"]:
                states = sorted(df[S["state"]].dropna().unique())
                chosen = st.multiselect("State", states)
                if chosen:
                    f = f[f[S["state"]].isin(chosen)]
            sort_col = S["monetary"] or ID
            candidates = f.sort_values(sort_col, ascending=False)
            st.caption(f"{len(candidates):,} customers match (sorted by spend)")
            st.download_button("⬇ Download this segment (CSV)", candidates.to_csv(index=False), "segment.csv", "text/csv")

        else:  # Quick picks
            c1, c2 = st.columns(2)
            if c1.button("🎲 Random customer", width="stretch"):
                st.session_state.selected = df.sample(1)[ID].iloc[0]
            if S["frequency"] and c2.button("🔁 Random repeat buyer", width="stretch"):
                rep = df[df[S["frequency"]] > 1]
                if len(rep):
                    st.session_state.selected = rep.sample(1)[ID].iloc[0]
            if S["monetary"] and st.button("💎 A top-spending customer", width="stretch"):
                st.session_state.selected = df.nlargest(200, S["monetary"]).sample(1)[ID].iloc[0]
            if S["review"] and st.button("😠 A low-review customer", width="stretch"):
                low = df[df[S["review"]] <= 2]
                if len(low):
                    st.session_state.selected = low.sample(1)[ID].iloc[0]
            candidates = df.head(0)

        if len(candidates):
            show = candidates.head(300)

            def fmt(cid):
                row = show[show[ID] == cid].iloc[0]
                bits = [short(cid)]
                if S["frequency"]:
                    bits.append(f"{int(row[S['frequency']])} order(s)")
                if S["monetary"]:
                    bits.append(f"R$ {row[S['monetary']]:,.0f}")
                return "  ·  ".join(bits)

            current = st.session_state.selected if st.session_state.selected in set(show[ID]) else None
            options = list(show[ID])
            idx = options.index(current) if current in options else 0
            choice = st.selectbox(
                "Results (type to filter)", options, index=idx, format_func=fmt,
                help="Shows the first 300 matches. Narrow the search to see others.",
            )
            if choice != st.session_state.selected:
                st.session_state.selected = choice

        with st.expander("Paste a full ID"):
            pasted = st.text_input("Full customer ID", key="paste").strip()
            if pasted and pasted in set(df[ID]):
                st.session_state.selected = pasted
            elif pasted:
                st.warning("ID not found in the features file.")

    # ----------------------------------- Customer 360 ----------------------------------
    with right:
        cid = st.session_state.selected
        if not cid or cid not in set(df[ID]):
            st.markdown("#### Customer 360")
            st.info("Pick a customer on the left. Try **Quick picks → Random repeat buyer** for a quick demo.")
        else:
            st.query_params["cid"] = cid
            row = df[df[ID] == cid].iloc[0]
            st.markdown(f"#### Customer 360 &nbsp; <span class='idpill'>{cid}</span>", unsafe_allow_html=True)
            if S["city"] or S["state"]:
                loc = ", ".join(str(row[S[k]]) for k in ("city", "state") if S[k])
                st.caption(f"📍 {loc.title()}")

            # Activity KPIs
            cols = st.columns(4)
            with cols[0]:
                kpi("Orders", f"{int(row[S['frequency']])}" if S["frequency"] else "n/a", "lifetime")
            with cols[1]:
                kpi("Total spend", f"R$ {row[S['monetary']]:,.0f}" if S["monetary"] else "n/a", "all orders")
            with cols[2]:
                kpi("Last order", f"{int(row[S['recency']])} d ago" if S["recency"] else "n/a", "recency")
            with cols[3]:
                kpi("Avg review", f"{row[S['review']]:.1f} ★" if S["review"] else "n/a", "out of 5")

            st.write("")

            # Prediction
            with st.spinner("Scoring customer (the API may be waking up)..."):
                resp, perr = predict(api_url, cid)

            if resp is None:
                st.error(f"Could not get a prediction: {perr}")
                st.caption("If the API was asleep, wait a few seconds and press Retry in the sidebar.")
            else:
                p = parse_prediction(resp)
                css, blurb = tier_style(p["tier"])
                g1, g2 = st.columns([1.1, 1])
                with g1:
                    st.plotly_chart(lift_gauge(p["prob"]), width="stretch")
                    st.markdown(
                        f"<div style='text-align:center'><span class='tier {css}'>{p['tier']}</span>"
                        f"<div class='kpi-sub' style='margin-top:6px'>{blurb} · model probability {p['prob']*100:.2f}% "
                        f"(avg customer ≈ {BASELINE*100:.1f}%)</div></div>",
                        unsafe_allow_html=True,
                    )
                with g2:
                    pcts = {}
                    for key, label, invert in [("frequency", "Orders", False), ("monetary", "Spend", False),
                                               ("recency", "Recent activity", True), ("review", "Satisfaction", False)]:
                        if S[key]:
                            v = pct_rank(df[S[key]], row[S[key]])
                            pcts[label] = 100 - v if invert else v
                    if pcts:
                        st.plotly_chart(percentile_chart(pcts), width="stretch")

                st.markdown("##### 🎯 Recommended categories")
                if p["recs"]:
                    st.markdown("".join(f"<span class='reco'>{str(r).replace('_', ' ').title()}</span>" for r in p["recs"]), unsafe_allow_html=True)
                else:
                    st.caption("No recommendations returned for this customer.")

                title, body = ACTIONS[css]
                st.markdown(f"<div class='action'><b>Suggested action: {title}</b><br>{body}</div>", unsafe_allow_html=True)

            # Signals
            sig = []
            if S["delay"] and row[S["delay"]] > 0:
                sig.append(f"⚠️ Deliveries ran **{row[S['delay']]:.1f} days late** on average, a known driver of bad reviews.")
            if S["review"] and row[S["review"]] <= 2:
                sig.append("⚠️ Low review score, so consider proactive support outreach.")
            if S["frequency"] and row[S["frequency"]] > 1:
                sig.append("✅ Already a repeat buyer, so prioritise loyalty actions.")
            if sig:
                st.markdown("##### Signals")
                for s in sig:
                    st.markdown(f"- {s}")

            with st.expander("All raw features for this customer"):
                st.dataframe(row.to_frame("value"), width="stretch")

# ========================================== OVERVIEW ====================================
with tab_overview:
    st.markdown("#### Portfolio at a glance")
    a, b, c, d = st.columns(4)
    with a:
        kpi("Customers", f"{len(df):,}", "unique customers")
    if S["monetary"]:
        with b:
            kpi("Avg spend", f"R$ {df[S['monetary']].mean():,.0f}", f"median R$ {df[S['monetary']].median():,.0f}")
    if S["frequency"]:
        with c:
            rep = (df[S["frequency"]] > 1).mean() * 100
            kpi("Repeat buyers", f"{rep:.2f}%", "the rare, valuable segment")
    if S["review"]:
        with d:
            kpi("Avg review", f"{df[S['review']].mean():.2f} ★", "across customers")
    st.write("")
    g1, g2 = st.columns(2)
    if S["frequency"]:
        vc = df[S["frequency"]].clip(upper=5).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f"{i}+" if i == 5 else str(i) for i in vc.index], y=vc.values, marker_color="#4f46e5"))
        fig.update_layout(title="Orders per customer (extreme one-time-buyer skew)", height=320, yaxis_type="log",
                          yaxis_title="Customers (log scale)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        g1.plotly_chart(fig, width="stretch")
    if S["review"]:
        vc = df[S["review"]].round().value_counts().sort_index()
        fig = go.Figure(go.Bar(x=vc.index.astype(int).astype(str), y=vc.values, marker_color="#7c3aed"))
        fig.update_layout(title="Review score distribution", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        g2.plotly_chart(fig, width="stretch")
    if S["delay"] and S["review"]:
        tmp = df[[S["delay"], S["review"]]].dropna()
        tmp["bucket"] = pd.cut(tmp[S["delay"]], [-1000, -5, 0, 5, 10, 1000], labels=["5+ d early", "early", "0-5 d late", "5-10 d late", "10+ d late"])
        agg = tmp.groupby("bucket", observed=True)[S["review"]].mean()
        fig = go.Figure(go.Bar(x=agg.index.astype(str), y=agg.values, marker_color="#dc2626"))
        fig.update_layout(title="Late delivery hurts reviews", yaxis_title="Avg review score", height=320, yaxis_range=[1, 5],
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

# ========================================== MODEL =======================================
with tab_model:
    st.markdown("#### What we learned: reframing churn → retention")
    st.markdown(
        """
**The pitfall.** 98.8% of Olist customers never place a second order, so `churned = 1` is the *majority* class.
A model that always says "will churn" is right 98.8% of the time, and PR-AUC looks excellent (~0.99) even though ROC-AUC sits at
0.52-0.58 (near a coin flip). The model had learned only the class prior.

**The fix.** Predict the *rare, valuable* behaviour instead: **`retained = 1`** (the 1.2% who actually buy again).
PR-AUC is now read against a random-guess baseline of **0.0117**, which is honest and actionable.
"""
    )
    res = pd.DataFrame(
        {
            "Model": ["Logistic Regression ✅", "XGBoost + SMOTE", "LightGBM + SMOTE"],
            "PR-AUC": [0.0224, 0.0144, 0.0135],
            "ROC-AUC": [0.576, 0.532, 0.496],
            "F1": [0.0267, 0.0101, 0.0260],
            "Recall": [0.4962, 0.0076, 0.0153],
        }
    )
    res["Lift vs random"] = (res["PR-AUC"] / BASELINE).round(2)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=res["Model"], y=res["PR-AUC"], marker_color=["#4f46e5", "#9ca3af", "#9ca3af"], name="PR-AUC")
        fig.add_hline(y=BASELINE, line_dash="dash", line_color="#dc2626", annotation_text="Random baseline 0.0117")
        fig.update_layout(title="PR-AUC vs random baseline", height=340, yaxis_title="PR-AUC",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(res.set_index("Model"), width="stretch")
        st.success("Selected: **Logistic Regression**. About **2× lift** over random and **~50% recall** of repeat buyers.")
    st.info(
        "**Honest takeaway:** the signal in a one-time-buyer marketplace is weak. The value of this project is the "
        "end-to-end, leakage-free system, plus the discovery that the naive churn framing is misleading. Next steps: add "
        "time-aware features, richer product/seller signals, and calibrate probabilities."
    )

    st.markdown("#### System architecture")
    st.markdown(
        "`Raw Olist CSVs` → `features.py (RFM + delivery + review)` → `customer_features.csv` → "
        "`train_churn.py / recommender.py` → `joblib artifacts` → `FastAPI /predict/{id}` → `Streamlit dashboard`"
    )
