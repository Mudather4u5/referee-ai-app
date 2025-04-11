import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("var_decision_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("VAR Decision Predictor")
st.write("Enter the incident details to predict the appropriate decision.")

# Create input form for the 8 features
feature_1 = st.number_input("Feature 1", min_value=0.0)
feature_2 = st.number_input("Feature 2", min_value=0.0)
feature_3 = st.number_input("Feature 3", min_value=0.0)
feature_4 = st.number_input("Feature 4", min_value=0.0)
feature_5 = st.number_input("Feature 5", min_value=0.0)
feature_6 = st.number_input("Feature 6", min_value=0.0)
feature_7 = st.number_input("Feature 7", min_value=0.0)
feature_8 = st.number_input("Feature 8", min_value=0.0)

# Predict button
if st.button("Predict Decision"):
    input_data = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4,
                                feature_5, feature_6, feature_7, feature_8]],
                              columns=["feature_1", "feature_2", "feature_3", "feature_4",
                                       "feature_5", "feature_6", "feature_7", "feature_8"])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.subheader("Prediction:")
    st.write(f"Predicted Decision Code: {prediction[0]}")