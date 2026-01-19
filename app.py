import streamlit as st
import pandas as pd
import pickle
# Load model
data = pickle.load(open("food_delivery_model.pkl", "rb"))
model = data["model"]
model_columns = data["columns"]

# Create input DataFrame
input_df = pd.DataFrame([{
    "Distance_km": distance,
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_day,
    "Vehicle_Type": vehicle
}])

# Encode input
input_encoded = pd.get_dummies(input_df)

# Add missing columns
for col in model_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

# Keep correct order
input_encoded = input_encoded[model_columns]

# Predict
prediction = model.predict(input_encoded)

