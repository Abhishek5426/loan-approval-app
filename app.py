import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval System")

# Inputs (your format)
employee = st.selectbox("Employee", ["Yes", "No"])
salary = st.number_input("Monthly Salary")
cibil = st.number_input("CIBIL Score")
loan = st.number_input("Loan Amount")
property_area = st.selectbox("Property", ["Rural", "Semiurban", "Urban"])

# Convert inputs
employee_val = 1 if employee == "Yes" else 0
property_map = {"Rural":0, "Semiurban":1, "Urban":2}

if st.button("Predict"):
    data = np.array([[employee_val, salary, cibil, loan, property_map[property_area]]])
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
