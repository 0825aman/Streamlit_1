import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
with open('loantap_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('loantap_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for insights
df = pd.read_csv('logistic_regression.csv')

df.loc[(df.home_ownership == 'ANY') | (df.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['issue_d'] = pd.to_datetime(df['issue_d'])
df['Credit_History_Years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days/365.25
df['issue_year'] = df['issue_d'].dt.year
df['issue_month'] = df['issue_d'].dt.strftime('%B')
df['earliest_cr_year'] = df['earliest_cr_line'].dt.year
df2 = df.copy()
# 1. Flag for pub_rec
df2['pub_rec_flag'] = df2['pub_rec'].apply(lambda x: 1 if x > 1.0 else 0)

# 2. Flag for mort_acc
df2['mort_acc_flag'] = df2['mort_acc'].apply(lambda x: 1 if x > 1.0 else 0)

# 3. Flag for pub_rec_bankruptcies
df2['pub_rec_bankruptcies_flag'] = df2['pub_rec_bankruptcies'].apply(lambda x: 1 if x > 1.0 else 0)

sub_grade_default_rate = df2.groupby('sub_grade')['loan_status'].apply(lambda x: (x == 'Charged Off').mean())
df2['sub_grade_encoded'] = df2['sub_grade'].map(sub_grade_default_rate)

purpose_default_rate = df2.groupby('purpose')['loan_status'].apply(lambda x: (x == 'Charged Off').mean())
df2['purpose_encoded'] = df2['purpose'].map(purpose_default_rate)

issue_month_default_rate = df2.groupby('issue_month')['loan_status'].apply(lambda x: (x == 'Charged Off').mean())
df2['issue_month_encoded'] = df2['issue_month'].map(issue_month_default_rate)

term_values={' 36 months': 36, ' 60 months':60}
df2['term'] = df2.term.map(term_values)

list_status = {'w': 0, 'f': 1}
df2['initial_list_status'] = df2.initial_list_status.map(list_status)

df2['loan_status']=df2.loan_status.map({'Fully Paid':0, 'Charged Off':1})

df2['zip_code'] = df2.address.apply(lambda x: x[-5:])

df2 = pd.get_dummies(df2, columns=['purpose', 'zip_code', 'grade', 'verification_status', 'application_type', 'home_ownership'], drop_first=True)
df2.replace({True: 1, False: 0}, inplace=True)

df2.drop(columns=['issue_d','installment', 'emp_title', 'title', 'sub_grade',
                   'address', 'earliest_cr_line', 'emp_length','issue_year','issue_month','earliest_cr_year'],
                   axis=1, inplace=True)

total_acc_avg=df2.groupby(by='total_acc').mean().mort_acc
# saving mean of mort_acc according to total_acc_avg
def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc
df2['mort_acc']=df2.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)

df2.dropna(inplace=True)

# Extract expected feature names from scaler
feature_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df2.drop(columns=['loan_status']).columns

# Define categorical feature options
term_options = [36, 60]
purpose_options = df['purpose'].unique().tolist()
verification_status_options = df['verification_status'].unique().tolist()
grade_options = df['grade'].unique().tolist()
home_ownership_options = df['home_ownership'].unique().tolist()

# ---- UI Enhancements ----
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

# Custom CSS for better fonts and design
st.markdown("""
    <style>
        body { font-family: 'Poppins', sans-serif; }
        .stButton > button { background-color: #004AAD; color: white; font-size: 18px; padding: 10px; border-radius: 10px; }
        .stNumberInput input, .stSelectbox select { font-size: 16px; padding: 5px; }
        .stAlert { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown(
    "<h1 style='text-align: center; color: #004AAD;'>🏦 Loan Approval Prediction</h1>"
    "<p style='text-align: center; font-size: 18px;'>💡 Get an instant prediction on your loan approval status!</p>",
    unsafe_allow_html=True
)

# ---- Layout with Columns ----
col1, col2 = st.columns([1, 1])  # Two equal columns

# ---- Left Column: User Inputs ----
with col1:
    st.markdown("### 📋 **Enter Loan Details**")
    loan_amnt = st.number_input("💰 **Loan Amount (USD)**", min_value=500, max_value=50000, step=500, help="Enter the amount you wish to borrow.")
    term = st.selectbox("📆 **Loan Term (Months)**", term_options, help="Select the duration of the loan.")
    int_rate = st.number_input("📊 **Interest Rate (%)**", min_value=1.0, max_value=30.0, step=0.1, help="Enter the interest rate for your loan.")
    dti = st.number_input("📉 **Debt-to-Income Ratio (DTI)**", min_value=0.0, max_value=50.0, step=0.1, help="DTI measures your monthly debt payments against your income.")
    purpose = st.selectbox("🎯 **Purpose of Loan**", purpose_options, help="Select the purpose of your loan.")
    verification_status = st.selectbox("✅ **Verification Status**", verification_status_options, help="Indicates whether your income is verified.")
    grade = st.selectbox("🏅 **Credit Grade**", grade_options, help="Your creditworthiness level assigned by the lender.")
    annual_inc = st.number_input("💵 **Annual Income (USD)**", min_value=10000, max_value=500000, step=1000, help="Enter your yearly income before tax.")
    home_ownership = st.selectbox("🏠 **Home Ownership**", home_ownership_options, help="Your current housing situation.")
    Credit_History_Years = st.number_input("📜 **Credit History (Years)**", min_value=0, max_value=50, step=1, help="Years since your first credit account.")

# Prepare input data
input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt], 'term': [term], 'int_rate': [int_rate], 'dti': [dti],
    'purpose': [purpose], 'verification_status': [verification_status], 'grade': [grade],
    'annual_inc': [annual_inc], 'home_ownership': [home_ownership], 'Credit_History_Years': [Credit_History_Years]
})

# One-Hot Encoding for Categorical Variables
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Scale numerical values
input_data_scaled = scaler.transform(input_data)

# Prediction function
def predict_loan_status(input_data_scaled):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    return ('Approved ✅' if prediction[0] == 1 else 'Rejected ❌', probability)

# ---- Right Column: Loan Summary & Prediction ----
with col2:
    st.markdown("### 📊 **Loan Summary**")
    st.info(f"""
    - **Loan Amount:** ${loan_amnt}
    - **Term:** {term} months
    - **Interest Rate:** {int_rate}%
    - **Debt-to-Income Ratio:** {dti}%
    - **Purpose:** {purpose}
    - **Verification Status:** {verification_status}
    - **Credit Grade:** {grade}
    - **Annual Income:** ${annual_inc}
    - **Home Ownership:** {home_ownership}
    - **Credit History:** {Credit_History_Years} years
    """)

    if st.button("🔍 **Predict Loan Approval**"):
        result, probability = predict_loan_status(input_data_scaled)

        if result == "Approved ✅":
            st.success(f"🎉 **Congratulations! Your loan is likely to be {result}**")
        else:
            st.error(f"⚠️ **Unfortunately, your loan is likely to be {result}**")
#
#         # Probability Bar
#         st.markdown("### 📈 **Approval Probability**")
#         fig, ax = plt.subplots(figsize=(5, 1))
#         ax.barh(["Approval Chance"], [probability], color="green" if result == "Approved ✅" else "red")
#         ax.set_xlim(0, 1)
#         ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
#         ax.set_yticklabels([])
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         ax.bar_label(ax.containers[0], fmt="%.2f", label_type="center", color="white", fontsize=12)
#         st.pyplot(fig)
#
# # ---- Insights Section ----
# st.markdown("## 📊 **Insights from Historical Data**")
#
# # 1. Comparison with Approved Loans
# approved_loans = df[df['loan_status'] == 1]
# avg_approved_income = approved_loans['annual_inc'].mean()
# avg_approved_dti = approved_loans['dti'].mean()
#
# st.info(f"""
# 🔹 **Average Annual Income for Approved Loans:** ${avg_approved_income:.2f}
# 🔹 **Average DTI for Approved Loans:** {avg_approved_dti:.2f}%
# """)
#
# # 2. Approval Rate by Loan Purpose
# # Ensure 'loan_status' is numeric
# df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
#
# # Approval rate by loan purpose
# st.markdown("### 📌 Loan Approval Rate by Purpose")
# purpose_approval = df.groupby("purpose")["loan_status"].mean().sort_values()

# fig, ax = plt.subplots(figsize=(8, 4))
# sns.barplot(y=purpose_approval.index, x=purpose_approval.values, palette="Blues_r", ax=ax)
# ax.set_xlabel("Approval Rate")
# st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>🚀 Created by <b>Aman Shrivastava</b></p>
        <p>📧 Contact: <a href="mailto:amanshrivastava26266@gmail.com">amanshrivastava26266@gmail.com</a></p>
        <p>🔗 <a href="https://www.linkedin.com/in/aman0802/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/0825aman" target="_blank">GitHub</a> | 
        <a href="https://www.kaggle.com/the0aman0shrivastava" target="_blank">Kaggle</a></p>
    </div>
""", unsafe_allow_html=True)
