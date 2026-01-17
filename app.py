import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("food_delivery_model.pkl", "rb"))

st.title("🍔 Food Delivery Time Prediction")

st.write("Enter order details to predict delivery time")

# User Inputs
distance = st.number_input("Distance (km)", min_value=1.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=1)
experience = st.number_input("Courier Experience (years)", min_value=0.0)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])

# Create input dataframe
input_data = pd.DataFrame({
    "Distance_km": [distance],
    "Preparation_Time_min": [prep_time],
    "Courier_Experience_yrs": [experience],
    "Weather_Clear": [1 if weather == "Clear" else 0],
    "Weather_Foggy": [1 if weather == "Foggy" else 0],
    "Weather_Rainy": [1 if weather == "Rainy" else 0],
    "Weather_Windy": [1 if weather == "Windy" else 0],
    "Traffic_Level_Low": [1 if traffic == "Low" else 0],
    "Traffic_Level_Medium": [1 if traffic == "Medium" else 0],
    "Time_of_Day_Morning": [1 if time_day == "Morning" else 0],
    "Time_of_Day_Night": [1 if time_day == "Night" else 0],
    "Vehicle_Type_Scooter": [1 if vehicle == "Scooter" else 0]
})

# Predict
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")
