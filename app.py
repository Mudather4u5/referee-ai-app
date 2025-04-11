
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("var_decision_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("VAR Decision Predictor")
st.write("Enter the details of the incident to predict the appropriate decision or law.")

# Create input form
fouls = st.number_input("Fouls", min_value=0)
yellow_cards = st.number_input("Yellow Cards", min_value=0)
red_cards = st.number_input("Red Cards", min_value=0)
penalties_awarded = st.number_input("Penalties Awarded", min_value=0)
offsides = st.number_input("Offsides", min_value=0)
var_decisions = st.number_input("Previous VAR Decisions", min_value=0)

# Predict button
if st.button("Predict Decision"):
    input_data = pd.DataFrame([[fouls, yellow_cards, red_cards, penalties_awarded, offsides, var_decisions]],
                              columns=["Fouls", "Yellow Cards", "Red Cards", "Penalties Awarded", "Offsides", "VAR Decisions"])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    st.subheader("Prediction:")
    st.write(f"Predicted Decision Code: {prediction[0]}")
