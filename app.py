import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("wine_cv_model.pkl")

st.set_page_config(page_title="Wine Quality Prediction")

st.title("🍷 Wine Quality Prediction (Cross Validation)")

# Inputs
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 15.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.01, 0.5, 0.076)
free_sulfur = st.number_input("Free Sulfur Dioxide", 1, 100, 11)
total_sulfur = st.number_input("Total Sulfur Dioxide", 6, 300, 34)
density = st.number_input("Density", 0.9900, 1.0050, 0.9978)
pH = st.number_input("pH", 2.5, 4.0, 3.51)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

# Prediction
if st.button("Predict Quality"):
    data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                      residual_sugar, chlorides, free_sulfur,
                      total_sulfur, density, pH, sulphates, alcohol]])

    prediction = model.predict(data)
    st.success(f"Predicted Wine Quality: {round(prediction[0], 2)} 🍷")
