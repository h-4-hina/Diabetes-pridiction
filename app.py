import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
try:
    model_data = joblib.load("diabetes_model.pkl")
    if isinstance(model_data, dict):
        if "model" in model_data:
            model = model_data["model"]
        elif len(model_data) == 1:
            model = list(model_data.values())[0]
        else:
            st.error("❌ Model not found in file. Please check your .pkl contents.")
            st.stop()
    else:
        model = model_data
except FileNotFoundError:
    st.error("❌ diabetes_model.pkl not found. Place it in the same folder as this app.")
    st.stop()

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Gradient Background CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #74ABE2, #5563DE);
        color: white;
    }
    /* Streamlit main content */
    .stApp {
        background: linear-gradient(135deg, #74ABE2, #5563DE);
        padding: 20px;
        border-radius: 12px;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    /* Inputs */
    .stNumberInput label {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# App title & description
# -----------------------------
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether you have diabetes based on your health parameters.")

st.markdown("---")

# -----------------------------
# Input fields
# -----------------------------
st.subheader("Enter Your Health Details")

glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=80)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict"):
    try:
        # Prepare features in the same order as the model was trained
        features = np.array([[glucose, blood_pressure, insulin, bmi, age]])
        prediction = model.predict(features)[0]

        st.markdown("---")
        st.subheader("🔎 Prediction Result:")

        if prediction == 1:
            st.error("🚨 You have diabetes.")
            st.subheader("🛡 Precautions & Recommendations")
            st.write("""
            - Follow a **low-sugar, balanced diet**.
            - Engage in **regular exercise** (30 min/day).
            - Monitor **blood sugar levels** regularly.
            - Consult a **doctor/endocrinologist** for medication if needed.
            - Avoid smoking & limit alcohol consumption.
            """)
        else:
            st.success("✅ You do not have diabetes.")
            st.subheader("💡 Health Maintenance Tips")
            st.write("""
            - Maintain a **healthy diet** with vegetables and fruits.
            - Stay **physically active**.
            - Get **annual check-ups** for early detection.
            - Maintain a healthy weight & BMI.
            """)

        # Show BMI category
        st.markdown("---")
        st.subheader("📊 Your BMI Category:")
        if bmi < 18.5:
            st.info(f"Your BMI is {bmi} → Underweight")
        elif 18.5 <= bmi <= 24.9:
            st.success(f"Your BMI is {bmi} → Normal weight")
        elif 25 <= bmi <= 29.9:
            st.warning(f"Your BMI is {bmi} → Overweight")
        else:
            st.error(f"Your BMI is {bmi} → Obese")

    except Exception as e:
        st.error(f"⚠ Error making prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed by Hina sawaira — University of Narowal")
