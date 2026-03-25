import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/fraud_pipeline.pkl")

model = load_model()

st.title("🛡️ Fraud Detection System")
st.write("Enter transaction details below to check whether a transaction may be fraudulent.")

with st.form("fraud_form"):
    step = st.number_input("Step", min_value=1, value=1)
    transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=2000.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud", [0, 1])

    submitted = st.form_submit_button("Predict")

if submitted:
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

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This transaction is predicted as FRAUD")
    else:
        st.success("✅ This transaction is predicted as LEGITIMATE")

    st.write(f"Fraud Probability: {probability:.2%}")

    reasons = []
    if amount > 200000:
        reasons.append("High transaction amount")
    if transaction_type in ["TRANSFER", "CASH_OUT"]:
        reasons.append("Risky transaction type")
    if oldbalanceOrg > 0 and newbalanceOrig == 0:
        reasons.append("Origin balance dropped sharply")
    if isFlaggedFraud == 1:
        reasons.append("Transaction is flagged")

    if reasons:
        st.write("Possible risk factors:")
        for reason in reasons:
            st.write("-", reason)