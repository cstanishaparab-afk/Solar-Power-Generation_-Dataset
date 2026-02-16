import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("power_prediction_model.pkl")

st.set_page_config(page_title="Solar Power Prediction", layout="centered")

st.title("☀️ Solar Power Generation Prediction App")
st.write("Enter the values below to predict **Power Generated**.")

# Show model expected features
st.write("📌 Model expects features:", model.n_features_in_)

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

# Input DataFrame (15 features)
input_data = pd.DataFrame([[day_of_year, year, month, day, first_hour,
                           is_daylight, distance_to_solar_noon, avg_temp_day,
                           avg_wind_dir_day, avg_wind_speed_day, sky_cover,
                           visibility, relative_humidity, avg_wind_speed_period,
                           avg_pressure_period]])

st.write("### 🔍 Input Data Preview")
st.dataframe(input_data)

if st.button("⚡ Predict Power Generated"):
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Power Generated: **{prediction[0]:.2f}**")
