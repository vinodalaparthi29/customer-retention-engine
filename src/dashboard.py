"""
Streamlit Dashboard for the Smart Customer Retention Engine.
Connects to the FastAPI backend to display retention scores and product recommendations.
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Retention Engine", layout="wide")

st.title("Smart E-Commerce Customer Retention Dashboard")
st.markdown("Predict high-value repeat buyers and deliver targeted product recommendations.")

# Customer ID Input
customer_id = st.text_input(
    "Enter Customer Unique ID:",
    value="0000366f3b9a7992bf8c76cfdf3221e2",
)

if st.button("Analyze Customer"):
    try:
        response = requests.post(API_URL, json={"customer_unique_id": customer_id})
        
        if response.status_code == 200:
            data = response.json()
            
            # Top Metrics Row
            col1, col2, col3 = st.columns(3)
            col1.metric("Retention Score", f"{data['retention_probability'] * 100:.2f}%")
            col2.metric("Churn Risk", f"{data['churn_probability'] * 100:.2f}%")
            col3.metric("Status Level", data["risk_level"])

            st.divider()

            # Customer Profiles & Behavioral Data
            st.subheader("Customer Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Recency (Days)", data["metrics"]["recency_days"])
            m_col2.metric("Total Spend ($)", f"${data['metrics']['monetary']:.2f}")
            m_col3.metric("Order Count", data["metrics"]["frequency"])
            m_col4.metric("Avg Review Score", data["metrics"]["avg_review_score"])

            st.divider()

            # Recommended Retention Categories
            st.subheader("Personalized Category Recommendations")
            st.info("Top product categories recommended for retargeting campaigns:")
            
            # Fallback sample categories generated from recommender.py
            sample_recs = ["casa_conforto", "moveis_decoracao", "utilidades_domesticas"]
            for item in sample_recs:
                st.markdown(f"* **Category Offer:** `{item}`")

        else:
            st.error("Customer ID not found in customer_features.csv.")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Ensure `uvicorn src.api:app --reload` is running!")