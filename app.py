import streamlit as st
import pandas as pd
import pickle

# Load model
data = pickle.load(open("food_delivery_model.pkl", "rb"))
model = data["model"]
model_columns = data["columns"]

st.title("🍔 Food Delivery Time Prediction")

# -------------------------
# STEP 1: USER INPUTS
# -------------------------
distance = st.number_input("Distance (km)", min_value=1.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=1)
experience = st.number_input("Courier Experience (years)", min_value=0.0)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])

# -------------------------
# STEP 2: CREATE INPUT DATA
# -------------------------
input_df = pd.DataFrame([{
    "Distance_km": distance,
    "Preparation_Time_min": prep_time,
    "Courier_Experience_yrs": experience,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_day,
    "Vehicle_Type": vehicle
}])

# -------------------------
# STEP 3: ENCODE INPUT
# -------------------------
input_encoded = pd.get_dummies(input_df)

for col in model_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[model_columns]

# -------------------------
# STEP 4: PREDICTION
# -------------------------
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_encoded)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")
