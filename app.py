import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Delivery Predictor", layout="wide")

# 2. Load model and columns (Using your specific Pickle logic)
# Ensure your pickle file contains a dictionary with "model" and "columns"
data = pickle.load(open("food_delivery_model.pkl", "rb"))
model = data["model"]
model_columns = data["columns"]

# 3. Sidebar - Input Parameters (The Controls)
with st.sidebar:
    st.markdown("## 🛵 Courier & Vehicle")
    # Using sliders for a professional feel
    experience = st.slider("Courier Experience (years)", 0.0, 15.0, 3.0)
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])
    
    st.markdown("---")
    st.markdown("## 🌍 Environment")
    weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Windy"])
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# 4. Main Page Header
st.markdown("<h1 style='text-align: center;'>🍔 Food Delivery Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Machine Learning based arrival estimation</p>", unsafe_allow_html=True)
st.divider()

# 5. Dashboard Grid (Matching your Crop App layout)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Order Overview")
    # Main inputs in the center
    distance = st.number_input("Distance (km)", min_value=1.0, value=5.0)
    prep_time = st.number_input("Preparation Time (min)", min_value=1, value=15)
    
    # Display values as big numbers like the screenshot
    st.write("Current Traffic Status")
    st.title(f"{traffic}")

with col2:
    st.subheader("🧪 Personnel & Weather")
    # Using metrics to create the clean look from your image
    st.metric(label="Courier Experience", value=f"{experience} Years")
    st.metric(label="Selected Weather", value=weather)
    st.metric(label="Vehicle", value=vehicle)

st.divider()

# 6. Prediction Logic
if st.button("Predict Delivery Time", use_container_width=True):
    # Step A: Create Input DataFrame
    input_df = pd.DataFrame([{
        "Distance_km": distance,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_day,
        "Vehicle_Type": vehicle
    }])

    # Step B: Encode Input (Matches your original logic)
    input_encoded = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Step C: Predict
    prediction = model.predict(input_encoded)
    
    # Step D: Display Result
    st.balloons()
    
    # Large, professional result box using Markdown
    st.markdown(f"""
        <div style="background-color:#262730; padding:20px; border-radius:10px; border-left: 5px solid #ff4b4b; text-align:center;">
            <h2 style="color:white;">Estimated Arrival Time</h2>
            <h1 style="color:#ff4b4b; font-size:50px;">{prediction[0]:.2f} Minutes</h1>
        </div>
    """, unsafe_allow_html=True)

    # Simple Summary for the Download button
    report_text = f"Delivery Estimate: {prediction[0]:.2f} mins\nDistance: {distance}km\nWeather: {weather}"
    st.download_button("📥 Download Estimate", report_text, file_name="delivery_report.txt")
