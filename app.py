import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoders
model = joblib.load('model.pkl')
le_source = joblib.load('le_source.pkl')
le_day = joblib.load('le_day.pkl')
le_month = joblib.load('le_month.pkl')

st.title("⚡ SOLAR POWER Predictor")
st.write("Enter details to predict energy production (MW)")

# Sidebar inputs
source = st.selectbox("Energy Source", le_source.classes_)
day = st.selectbox("Day of Week", le_day.classes_)
month = st.selectbox("Month", le_month.classes_)
hour = st.slider("Hour of Day", 0, 23, 12)
day_of_year = st.number_input("Day of Year (1-365)", 1, 365, 150)

if st.button("Predict Production"):
    # Encode inputs using the saved encoders
    source_enc = le_source.transform([source])[0]
    day_enc = le_day.transform([day])[0]
    month_enc = le_month.transform([month])[0]
    
    # Prepare feature array
    features = np.array([[hour, source_enc, day_of_year, day_enc, month_enc]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    st.success(f"Estimated Production: {prediction:.2f} MW")
