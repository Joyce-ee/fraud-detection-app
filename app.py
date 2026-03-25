import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💼",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
}
.main {
    background-color: rgba(0,0,0,0);
}
h1, h2, h3 {
    color: #f9fafb;
}
.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #4f46e5);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/fraud_pipeline.pkl")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("💼 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "ℹ️ About"])

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>💼 Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-powered financial fraud detection platform</p>", unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="card">
    <h3>⚡ Fast Predictions</h3>
    <p>Instant fraud detection using trained ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="card">
    <h3>🧠 Smart Analysis</h3>
    <p>Analyzes transaction patterns and risk factors.</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="card">
    <h3>🔐 Secure System</h3>
    <p>Built with modern ML security techniques.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICT PAGE ----------------
elif page == "🔍 Predict":

    st.markdown("<h2>🔍 Transaction Analysis</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💳 Transaction Info")

        step = st.number_input("Step", value=1)
        transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", value=1000.0)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏦 Account Info")

        oldbalanceOrg = st.number_input("Old Balance Origin", value=5000.0)
        newbalanceOrig = st.number_input("New Balance Origin", value=4000.0)
        oldbalanceDest = st.number_input("Old Balance Destination", value=1000.0)
        newbalanceDest = st.number_input("New Balance Destination", value=2000.0)
        isFlaggedFraud = st.selectbox("Flagged Fraud", [0, 1])

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    if st.button("🚀 Analyze Transaction"):

        input_df = pd.DataFrame([{
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud
        }])

        input_df["amount_vs_oldbalance_orig"] = input_df["amount"] / (input_df["oldbalanceOrg"] + 1e-5)
        input_df["amount_vs_newbalance_orig"] = input_df["amount"] / (input_df["newbalanceOrig"] + 1e-5)
        input_df["balance_diff_orig"] = input_df["newbalanceOrig"] - input_df["oldbalanceOrg"]
        input_df["balance_diff_dest"] = input_df["newbalanceDest"] - input_df["oldbalanceDest"]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.write("")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{probability:.2%}")

        st.markdown("### 📊 Risk Insights")

        reasons = []
        if amount > 200000:
            reasons.append("High transaction amount")
        if transaction_type in ["TRANSFER", "CASH_OUT"]:
            reasons.append("Risky transaction type")
        if oldbalanceOrg > 0 and newbalanceOrig == 0:
            reasons.append("Sudden balance drop")
        if isFlaggedFraud == 1:
            reasons.append("Transaction flagged")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No major risks detected")

# ---------------- ABOUT PAGE ----------------
elif page == "ℹ️ About":
    st.markdown("<h2>ℹ️ About This Project</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <p>This system uses Machine Learning to detect fraudulent financial transactions in real-time.</p>
    <ul>
    <li>Model: Random Forest Classifier</li>
    <li>Framework: Streamlit</li>
    <li>Features: Real-time prediction, risk analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("---")
st.markdown("<p style='text-align:center;'>© 2026 Fraud Detection System | Built with ML & Streamlit</p>", unsafe_allow_html=True)
