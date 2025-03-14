# import streamlit as st
# import pandas as pd
import pickle
# import numpy as np
from PIL import Image
import time
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm  # Import statsmodels to use sm.add_constant
# Load assets
with open('OLA_LGB_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('OLA_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<p class="main-title">Driver Attrition Prediction 🚗</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict whether a driver will stay or leave using LightGBM Model</p>', unsafe_allow_html=True)

# Sidebar Content
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/0/0f/Ola_Cabs_logo.svg", width=200)
st.sidebar.markdown("## Problem Statement")
st.sidebar.write("Recruiting and retaining drivers is a major challenge for Ola. High attrition leads to increased costs in hiring and training new drivers. This model predicts if a driver will leave or stay based on demographic, performance, and tenure data.")

st.sidebar.write("**Features Used:**")
st.sidebar.write("- Age, Gender, Education Level")
st.sidebar.write("- Income, Joining Designation")
st.sidebar.write("- Tenure (Months), Total Business Value")
st.sidebar.write("- Quarterly Rating & Rating Increase")
st.sidebar.write("- Income Increase, City Encoding")

st.sidebar.markdown("## Model Information")
st.sidebar.write("**Algorithm:** LightGBM")
st.sidebar.write("**Scaling:** StandardScaler")
st.sidebar.write("**Prediction Output:** 1 - Churned, 0 - Active")

# Input fields in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('📅 Age', min_value=18, max_value=80, value=30, help='Enter the driver’s age')
    gender = st.radio('⚤ Gender', ['Male', 'Female'], help='Select the driver’s gender')
    education_level = st.selectbox('🎓 Education Level', [0, 1, 2], help='Encoded Education Levels: 0 - High School, 1 - Diploma, 2 - Graduate')
    income = st.number_input('💰 Income ($)', min_value=1000, max_value=100000, value=10000, step=500, help='Monthly income in USD')
    joining_designation = st.selectbox('🏅 Joining Designation', [1, 2, 3, 4, 5], help='Encoded: 1 - Trainee, 2 - Sub Junior Driver, 3 - Junior Driver, (4,5) - Senior Driver ')
    tenure_months = st.number_input('📆 Tenure in Months', min_value=0, max_value=100, value=12, step=1, help='How long has the driver worked in months?')

with col2:
    quarterly_rating = st.slider('⭐ Quarterly Rating', min_value=1, max_value=4, value=3, step=1, help='Performance rating (1 - Poor, 4 - Excellent)')
    total_business_value = st.number_input('💼 Total Business Value ($)', min_value=-1385530, max_value=95331060, value=5000, help='The total business value acquired by the driver in a month (negative business indicates cancellation/refund or car EMI adjustments)')
    quarterly_rating_increase = st.radio('⚤ Quarterly Rating Increase', ['Yes', 'No'], help='Is there any increment in Quarterly rating')
    income_increase = st.radio("⚤ Driver's Income Increase", ['Yes', 'No'], help="Is there any increment in Driver's Salary")
    city_encoded = st.slider('🏙️ City Encoded', min_value=0.0, max_value=1.0, value=0.5, help='City impact factor that is to be decided by company')

# Convert categorical inputs
gender_encoded = 1 if gender == 'Male' else 0
Quarterly_rating_increase_encoded = 1 if quarterly_rating_increase == 'Yes' else 0
Income_increase_encoded = 1 if income_increase == 'Yes' else 0
# Create input array
input_data = np.array([[
    age, gender_encoded, education_level, income, joining_designation,
    quarterly_rating, tenure_months, total_business_value,
    Quarterly_rating_increase_encoded, Income_increase_encoded, city_encoded
]])

# Scale input data
input_scaled = scaler.transform(input_data)

# Prediction button with loading effect
if st.button('🚀 Predict Attrition'):
    with st.spinner('Running prediction...'):
        time.sleep(2)  # Simulate delay
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction]
        result = '🚨 **Churned (Leaving)**' if prediction == 1 else '✅ **Active (Staying)**'
        st.success(f'### Prediction: {result}')
        st.info(f'**Confidence Level:** {confidence:.2%}')

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
