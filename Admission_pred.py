import pickle
import streamlit as st
import numpy as np
import statsmodels.api as sm  # Import statsmodels for OLS

# Set Page Configuration
st.set_page_config(page_title="Admission Predictor", page_icon="🎓", layout="centered")


# Load model and scaler only once using caching
@st.cache_resource
def load_model():
    with open('ols_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


sm_model, scaler = load_model()

# Sidebar with Banner and Problem Statement
st.sidebar.image("https://img.studydekho.com/uploads/c/2017/12/c-jamboree-education-pvt-ltd-jaipur-3512.jpg", width=300)

st.sidebar.markdown(
    """
    ## 🎓 About Jamboree Education
    Founded in 1993, Jamboree Education is Asia's leading and largest test prep institute with 35 centres in India, Singapore, UAE, and Nepal. With over 25 years of experience, we have trained more than 140,000 students for exams such as GMAT, GRE, SAT, TOEFL, and IELTS.

    ## 🎯 Problem Statement
    Predict the likelihood of a student getting admission based on academic scores and profile strength.

    **Key Factors Considered:**
    - GRE Score
    - TOEFL Score
    - SOP Strength
    - LOR Strength
    - Undergraduate GPA

    **Model Used:** Linear Regression (OLS)

    Enter your details and find out your chances of getting admitted!
    """
)

# Title and Header
st.title("🎓 Admission Prediction Model")
st.markdown("---")
st.markdown(
    """
    ## 📌 About This App
    This AI-powered admission predictor estimates your chances of getting into a university based on key academic factors using **Linear Regression**.

    **Enter your details below and find out your probability of admission!**
    """,
    unsafe_allow_html=True,
)

# User Input Section
st.subheader("📋 Enter Your Information")
col1, col2 = st.columns(2)

GRE = col1.number_input("📊 GRE Score (out of 340)", 1, 340, help="Enter your GRE score.")
TOEFL = col2.number_input("📚 TOEFL Score (out of 120)", 1, 120, help="Enter your TOEFL score.")
SOP = col1.number_input("✍️ SOP Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.1, help="Rate your SOP.")
LOR = col2.number_input("🔗 LOR Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.1, help="Rate your LOR.")
GPA = col1.number_input("📐 Undergraduate GPA (out of 10)", 1.0, 10.0, 7.0, step=0.1, help="Enter your GPA.")


# Prediction Function
def predict_admission(GRE, TOEFL, SOP, LOR, GPA):
    input_data = np.array([GRE, TOEFL, 4, SOP, LOR, GPA, 1]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    X_input_sm = sm.add_constant(input_data_scaled, has_constant='add')
    prediction = sm_model.predict(X_input_sm) * 100
    return prediction[0]


# Prediction Button
if st.button("🔮 Predict Admission Chance"):
    with st.spinner("🔍 Analyzing your profile..."):
        prediction = predict_admission(GRE, TOEFL, SOP, LOR, GPA)

    st.success(f"🎯 Your predicted admission chance is **{prediction:.2f}%**!")

    # Display additional insights
    if prediction > 80:
        st.balloons()
        st.write("🎉 **Excellent!** Your chances are very high!")
    elif prediction > 50:
        st.write("🙂 **Good!** You have a decent chance, but you can improve further.")
    else:
        st.write("📉 **Low probability.** Consider enhancing your profile!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 14px;">
        <p>🚀 Created by <b>Aman Shrivastava</b></p>
        <p>📧 Contact: <a href="mailto:amanshrivastava26266@gmail.com">amanshrivastava26266@gmail.com</a></p>
        <p>🔗 <a href="https://www.linkedin.com/in/aman0802/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/0825aman" target="_blank">GitHub</a> | 
        <a href="https://www.kaggle.com/the0aman0shrivastava" target="_blank">Kaggle</a></p>
    </div>
""", unsafe_allow_html=True)
