
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
with open('var_decision_rf_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

st.title("Referee Decision Prediction App")

# User input form
st.header("Enter Match Incident Data")
input_data = {}

input_data['FoulType'] = st.selectbox("Foul Type", ["Handball", "Offside", "Tackle", "Push", "Diving", "Other"])
input_data['PlayerPosition'] = st.selectbox("Player Position", ["Defender", "Midfielder", "Forward", "Goalkeeper"])
input_data['MatchTime'] = st.slider("Match Time (minutes)", 0, 120, 45)
input_data['Severity'] = st.slider("Foul Severity (1-10)", 1, 10, 5)
input_data['IsLastDefender'] = st.selectbox("Is Last Defender?", ["Yes", "No"])
input_data['InPenaltyArea'] = st.selectbox("In Penalty Area?", ["Yes", "No"])
input_data['TeamFoulsCount'] = st.number_input("Total Team Fouls", min_value=0, value=3)
input_data['PlayerWarnings'] = st.number_input("Player Warnings (Yellow Cards)", min_value=0, value=0)

# Convert categorical to numerical (same encoding used during training)
mapping = {
    "FoulType": {"Handball": 0, "Offside": 1, "Tackle": 2, "Push": 3, "Diving": 4, "Other": 5},
    "PlayerPosition": {"Defender": 0, "Midfielder": 1, "Forward": 2, "Goalkeeper": 3},
    "IsLastDefender": {"Yes": 1, "No": 0},
    "InPenaltyArea": {"Yes": 1, "No": 0}
}

for col in mapping:
    input_data[col] = mapping[col][input_data[col]]

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Scale input
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)

st.subheader("Predicted Decision")
st.write(prediction[0])
