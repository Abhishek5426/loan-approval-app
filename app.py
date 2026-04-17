import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction")

income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
credit_history = st.selectbox("Credit History", [0, 1])

if st.button("Predict"):
    data = np.array([[income, co_income, loan_amount, loan_term, credit_history]])
    
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)
    
    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
