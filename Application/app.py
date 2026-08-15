import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("P:/Dlithe/Project/ML_model/calorie_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="centered"
)

# Title
st.title("Calorie Burn Prediction")
st.write("Enter your workout details to predict the calories burned.")

st.divider()

# User inputs

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)



height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=159.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=250.0,
    value=70.0
)

duration = st.number_input(
    "Workout Duration (minutes)",
    min_value=1.0,
    max_value=300.0,
    value=50.0
)

heart_rate = st.number_input(
    "Heart Rate (BPM)",
    min_value=40.0,
    max_value=220.0,
    value=120.0
)

body_temp = st.number_input(
    "Body Temperature (°C)",
    min_value=30.0,
    max_value=45.0,
    value=37.0
)

st.divider()

# Predict button
if st.button("Predict Calories", use_container_width=True):

    # Convert gender to numerical value
    if gender == "Male":
        gender_value = 1
    else:
        gender_value = 0

    # Create input DataFrame
    input_data = pd.DataFrame([[
        gender_value,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]], columns=[
        "Gender",
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp"
    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"Estimated Calories Burned: **{prediction[0]:.2f} kcal**"
    )