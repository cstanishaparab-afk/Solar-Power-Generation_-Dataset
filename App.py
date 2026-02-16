import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model + label encoder
model = joblib.load("power_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Solar Power Prediction", layout="centered")

st.title("☀️ Solar Power Generation Prediction")
st.write("Enter the details below to predict the Power Generated.")

st.sidebar.header("Input Features")

day_of_year = st.sidebar.number_input("Day of Year", min_value=1, max_value=366, value=100)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2020)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
first_hour = st.sidebar.number_input("First Hour of Period", min_value=0, max_value=23, value=12)

is_daylight = st.sidebar.selectbox("Is Daylight", ["Yes", "No"])
is_daylight = 1 if is_daylight == "Yes" else 0

distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon", value=0.0)
avg_temp_day = st.sidebar.number_input("Average Temperature (Day)", value=25)
avg_wind_dir_day = st.sidebar.number_input("Average Wind Direction (Day)", value=180)
avg_wind_speed_day = st.sidebar.number_input("Average Wind Speed (Day)", value=5.0)
sky_cover = st.sidebar.number_input("Sky Cover", min_value=0, max_value=100, value=20)
visibility = st.sidebar.number_input("Visibility", value=10.0)
relative_humidity = st.sidebar.number_input("Relative Humidity", min_value=0, max_value=100, value=50)
avg_wind_speed_period = st.sidebar.number_input("Average Wind Speed (Period)", value=5.0)
avg_pressure_period = st.sidebar.number_input("Average Barometric Pressure (Period)", value=1013.0)

# Categorical feature (LabelEncoder classes)
st.sidebar.subheader("Encoded Feature Input")
categories = list(encoder.keys())
category_value = st.sidebar.selectbox("Select Category", categories)

encoded_value = encoder[category_value]


# Input DataFrame (order matters)
input_data = pd.DataFrame({
    "Day of Year": [day_of_year],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "First Hour of Period": [first_hour],
    "Is Daylight": [is_daylight],
    "Distance to Solar Noon": [distance_to_solar_noon],
    "Average Temperature (Day)": [avg_temp_day],
    "Average Wind Direction (Day)": [avg_wind_dir_day],
    "Average Wind Speed (Day)": [avg_wind_speed_day],
    "Sky Cover": [sky_cover],
    "Visibility": [visibility],
    "Relative Humidity": [relative_humidity],
    "Average Wind Speed (Period)": [avg_wind_speed_period],
    "Average Barometric Pressure (Period)": [avg_pressure_period],
    "Encoded Feature": [encoded_value]
})

st.write("### 🔍 Input Data")
st.dataframe(input_data)

if st.button("⚡ Predict Power Generated"):
prediction = model.predict(input_data.values)
st.success(f"✅ Predicted Power Generated: **{prediction[0]:.2f}**")

