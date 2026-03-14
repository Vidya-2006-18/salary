import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
with open('lr.pkl', 'rb') as file:
    lr_model_data = pickle.load(file)
    lr_model = lr_model_data['model']
with open('rf.pkl', 'rb') as file:
    rf_model_data = pickle.load(file)
    rf_model = rf_model_data['model']
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
education_mapping = {'Bachelor': 0, 'High School': 1, 'Master': 2, 'PhD': 3}
job_title_mapping = {'Analyst': 0, 'Director': 1, 'Engineer': 2, 'Manager': 3}
st.title("Employee Salary Prediction System")
st.divider()
st.write("With this system, you can get estimations for the salaries of the Employees")
st.subheader("Enter Employee Details")
education = st.selectbox( "Education Level",list(education_mapping.keys()))
experience = st.number_input( "Years of Experience",min_value=0, max_value=60, value=5, step=1)
job_title = st.selectbox("Job Title",list(job_title_mapping.keys()))
st.divider()
st.subheader("Select Prediction Model")
selected_model_name = st.selectbox( "Choose a model for prediction:", ["Linear Regression", "Random Forest"])

if selected_model_name == "Linear Regression":
    model = lr_model
elif selected_model_name == "Random Forest":
    model = rf_model
predict_button = st.button("Predict Salary")

st.divider()

if predict_button:
    education_encoded = education_mapping[education]
    job_title_encoded = job_title_mapping[job_title]
    input_data = np.array([education_encoded, experience, job_title_encoded]).reshape(1, -1)
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)[0]

    st.success(f"Predicted Salary using {selected_model_name}: ${prediction:,.2f}")
else:
    st.info("Please enter the employee details and click 'Predict Salary' to get an estimation.")
