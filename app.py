import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler once
model = joblib.load("var_decision_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("VAR Decision Predictor")

st.write("Enter the 8 features below:")

# Create input fields for 8 features
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)
feature_5 = st.number_input("Feature 5", value=0.0)
feature_6 = st.number_input("Feature 6", value=0.0)
feature_7 = st.number_input("Feature 7", value=0.0)
feature_8 = st.number_input("Feature 8", value=0.0)

# Button to predict
if st.button("Predict"):
    input_data = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4,
                                feature_5, feature_6, feature_7, feature_8]],
                              columns=["feature_1", "feature_2", "feature_3", "feature_4",
                                       "feature_5", "feature_6", "feature_7", "feature_8"])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    st.success(f"Predicted VAR Decision Code: {prediction[0]}")
