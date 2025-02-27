import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
with open('loantap_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('loantap_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset to infer correct feature names
df = pd.read_csv('logistic_regression.csv')

# Extract expected feature names from scaler
if hasattr(scaler, 'feature_names_in_'):
    feature_columns = scaler.feature_names_in_
else:
    feature_columns = df.drop(columns=['loan_status']).columns  # Backup method if scaler doesn't have feature names

# Define categorical feature options
term_options = [36, 60]
purpose_options = ['vacation', 'debt_consolidation', 'credit_card', 'home_improvement', 'small_business',
                   'major_purchase', 'other', 'medical', 'wedding', 'car', 'moving', 'house', 'educational', 'renewable_energy']
verification_status_options = ['Source Verified', 'Verified']
grade_options = ['B', 'C', 'E', 'D', 'F', 'G']
home_ownership_options = ['RENT', 'OWN', 'OTHER']

# Title and Header
st.title("🏦 Loan Approval Prediction System")
st.markdown("### 💡 Predict whether your loan will be approved or not!")

# Sidebar Inputs
st.sidebar.header("📋 **Enter Loan Details**")
loan_amnt = st.sidebar.number_input("💰 Loan Amount", min_value=500, max_value=50000, step=500, help="Enter the loan amount in USD.")
term = st.sidebar.selectbox("📆 Loan Term", term_options, help="Select the loan term (36 or 60 months).")
int_rate = st.sidebar.number_input("📊 Interest Rate (%)", min_value=1.0, max_value=30.0, step=0.1, help="Enter the loan interest rate.")
dti = st.sidebar.number_input("📉 Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=50.0, step=0.1, help="Percentage of monthly income spent on debt payments.")
purpose = st.sidebar.selectbox("🎯 Purpose of Loan", purpose_options, help="Select the loan purpose.")
verification_status = st.sidebar.selectbox("✅ Verification Status", verification_status_options, help="Lender's verification status of income.")
grade = st.sidebar.selectbox("🏅 Credit Grade", grade_options, help="Credit grade assigned to the borrower.")
annual_inc = st.sidebar.number_input("💵 Annual Income ($)", min_value=10000, max_value=500000, step=1000, help="Enter the applicant's annual income.")
home_ownership = st.sidebar.selectbox("🏠 Home Ownership", home_ownership_options, help="Select the applicant's home ownership status.")
Credit_History_Years = st.sidebar.number_input("📜 Credit History (Years)", min_value=0, max_value=50, step=1, help="Years since first credit account.")

# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt],
    'term': [term],
    'int_rate': [int_rate],
    'dti': [dti],
    'purpose': [purpose],
    'verification_status': [verification_status],
    'grade': [grade],
    'annual_inc': [annual_inc],
    'home_ownership': [home_ownership],
    'Credit_History_Years': [Credit_History_Years]
})

# One-Hot Encoding for Categorical Variables
input_data = pd.get_dummies(input_data)

# Ensure input data matches the model's expected features
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Scale numerical values
input_data_scaled = scaler.transform(input_data)

# Prediction function
def predict_loan_status(input_data_scaled):
    prediction = model.predict(input_data_scaled)
    return 'Approved ✅' if prediction[0] == 1 else 'Rejected ❌'

# Predict button
if st.sidebar.button("🔍 Predict Loan Approval"):
    result = predict_loan_status(input_data_scaled)
    st.sidebar.success(f"**Prediction Result:** {result}")

# Footer
st.markdown("""
---
💡 **Disclaimer:** This prediction is based on historical data and statistical modeling. Actual loan approval depends on the lender's discretion.
""")
