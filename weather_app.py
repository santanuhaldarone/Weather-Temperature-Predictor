import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('weather_model.pkl')

st.title("Weather Temperature Predictor")

st.write("Enter today's weather details to predict the temperature.")

humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)
mean_pressure = st.slider("Mean Pressure (mbar)", 950.0, 1050.0, 1013.0)

input_data = np.array([[humidity, wind_speed, mean_pressure]])

if st.button("Predict Temperature"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Temperature: {prediction:.2f}°C")

st.markdown("---")
st.caption("Made with ❤️ by Santanu")