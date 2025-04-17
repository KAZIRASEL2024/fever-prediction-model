import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("fever_model.pkl")

st.title("🤒 Fever Prediction App")
st.markdown("Enter patient details to check if fever is likely.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
blood_pressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)
temperature = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
cough = st.selectbox("Cough", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])
body_pain = st.selectbox("Body Pain", ["No", "Yes"])

# Convert categorical to numerical
cough_val = 1 if cough == "Yes" else 0
headache_val = 1 if headache == "Yes" else 0
body_pain_val = 1 if body_pain == "Yes" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, blood_pressure, heart_rate, temperature, cough_val, headache_val, body_pain_val]])
    prediction = model.predict(input_data)

    if prediction[0] == "Yes":
        st.success("🦠 High chance of fever!")
    else:
        st.info("✅ Fever not likely.")