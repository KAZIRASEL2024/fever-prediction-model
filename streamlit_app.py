
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fever_model.pkl")

# Streamlit page configuration
st.title("Fever Prediction App")
st.write("This is a machine learning app to predict whether you have fever or not.")

# Input fields for prediction
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
temperature = st.number_input("Enter Temperature", min_value=0.0, max_value=100.0, value=98.6)
body_ache = st.selectbox("Do you have body ache?", ["Yes", "No"])
cough = st.selectbox("Do you have a cough?", ["Yes", "No"])
fatigue = st.selectbox("Do you feel fatigued?", ["Yes", "No"])

# Convert categorical inputs to numerical values
body_ache = 1 if body_ache == "Yes" else 0
cough = 1 if cough == "Yes" else 0
fatigue = 1 if fatigue == "Yes" else 0

# Prepare the feature vector for prediction
features = np.array([age, temperature, body_ache, cough, fatigue]).reshape(1, -1)

# Prediction
if st.button("Predict Fever Condition"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.write("The model predicts: **You have fever.**")
    else:
        st.write("The model predicts: **You do not have fever.**")
    