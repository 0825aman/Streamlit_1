import pickle
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm  # Import statsmodels to use sm.add_constant

# Title of the app
st.title("🎓 **Admission Prediction Model**")

# Header
st.header("Welcome to the Admission Predictor")

# Introduction and description with better formatting
st.write("""
Column Profiling:

This prediction model uses the following columns from the dataset:

- **Serial No.**: Unique row ID for each record
- **GRE Scores**: Standardized test score out of 340
- **TOEFL Scores**: English proficiency score out of 120
- **University Rating**: Rating of the university (out of 5)
- **SOP (Statement of Purpose)**: Score indicating strength of the SOP (out of 5)
- **LOR (Letter of Recommendation)**: Strength of the recommendation (out of 5)
- **Undergraduate GPA**: GPA from undergraduate studies (out of 10)
- **Research Experience**: 1 if the student has research experience, 0 otherwise
- **Chance of Admit**: The likelihood of being admitted, ranging from 0 to 1
""")

# Load dataset and show first few rows
st.subheader("### Dataset Overview")
admis_df = pd.read_csv("./Jamboree_Admission.csv")
st.dataframe(admis_df)

# Spacing for neatness
st.write("\n\n")

# User input section with two columns
st.subheader("### Enter Your Information")

# Two columns layout for inputs
col1, col2 = st.columns(2)

# GRE input
GRE = col1.number_input('📊 GRE Score (out of 340)', min_value=1, max_value=340, help="Enter your GRE score.")

# TOEFL input
TOEFL = col2.number_input('📚 TOEFL Score (out of 120)', min_value=1, max_value=120, help="Enter your TOEFL score.")

# SOP input
SOP = col1.number_input('✍️ SOP (Statement of Purpose) Strength', min_value=1.0, max_value=5.0, step=0.1, help="Rate your SOP from 1.0 to 5.0")

# LOR input
LOR = col2.number_input('🔗 LOR (Letter of Recommendation) Strength', min_value=1.0, max_value=5.0, step=0.1, help="Rate your LOR from 1.0 to 5.0")

# Undergraduate GPA input
Undergrad_GPA = col1.number_input('📐 Undergraduate GPA (out of 10)', min_value=1.0, max_value=10.0, step=0.1, help="Enter your GPA on a scale of 1.0 to 10.0")

# Displaying the user's input values for confirmation
st.write("### Your Inputs:")
st.write(f"📊 **GRE Score**: {GRE}")
st.write(f"📚 **TOEFL Score**: {TOEFL}")
st.write(f"✍️ **SOP**: {SOP}")
st.write(f"🔗 **LOR**: {LOR}")
st.write(f"📐 **Undergraduate GPA**: {Undergrad_GPA}")

# Function for prediction
def fees_pred(GRE, TOEFL, SOP, LOR, Undergrad_GPA):
    # Load the trained model and scaler
    with open('ols_model.pkl', 'rb') as f:
        sm_model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Prepare the input data (including user inputs)
    input_data = np.array([GRE, TOEFL, 4, SOP, LOR, Undergrad_GPA, 1])
    input_data = input_data.reshape(1, -1)

    # Scale the input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Add constant for OLS model (intercept term)
    X_input_sm = sm.add_constant(input_data_scaled, has_constant='add')

    # Predict the chance of admission using the model
    prediction = sm_model.predict(X_input_sm)

    return prediction

# Button for triggering the prediction
if st.button("🔮 **Predict Your Chance of Admission**"):
    # Initialize progress bar
    progress_bar = st.progress(0)

    # Simulate progress and delay to show the progress bar gradually
    st.spinner("Calculating...")

    # Start with 0% and simulate steps
    progress_bar.progress(25)  # Step 1
    # Call the prediction function
    get_chance = fees_pred(GRE, TOEFL, SOP, LOR, Undergrad_GPA) * 100
    progress_bar.progress(50)  # Step 2

    # Simulate additional processing time (e.g., some post-prediction actions)
    progress_bar.progress(75)  # Step 3
    # Assuming there's some additional processing, for now we wait
    import time

    time.sleep(3)  # Just to make the spinner and progress feel more realistic

    # Finally, complete the progress at 100%
    progress_bar.progress(100)  # Step 4

    # Display the predicted chance of admission with appropriate formatting
    st.write(f"### 📈 **Predicted Chance of Admission**: {get_chance[0]:.2f}%")

    st.progress(100)


