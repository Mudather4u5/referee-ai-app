import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load("var_decision_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load new data to predict (replace 'new_data.csv' with your file)
new_data = pd.read_csv("new_data.csv")

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Make prediction
predictions = model.predict(new_data_scaled)

# Show predictions
print("Predictions:",predictions)