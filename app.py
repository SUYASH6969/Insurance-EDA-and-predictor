import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the model
with open('model_pipeline.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Streamlit app
st.title("Insurance Charges Estimator")

# Collect user input
age = st.number_input("Enter your age:", min_value=18, max_value=100, value=25, step=1)
bmi = st.number_input("Enter your BMI:", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Enter the number of children:", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Are you a smoker?", options=['yes', 'no'])
region = st.selectbox("Select your region:", options=['northeast', 'northwest', 'southeast', 'southwest'])

#print(age,bmi,children,smoker,region)
#print(type(age),type(bmi),type(children),type(smoker),type(region))

# Create input for prediction
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Make prediction
predicted_log_charges = model_pipeline.predict(input_data)
print("predicted_log_charges-",predicted_log_charges)

# Reverse the log transformation (since the model was trained on log-transformed target)
predicted_charges = np.exp(predicted_log_charges)
print("predicted_charges",predicted_charges)
# Display the predicted charges
st.write(f"Estimated  Charges: ${predicted_charges[0]:,.2f}")
