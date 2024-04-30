import streamlit as st
import pandas as pd
import pickle

import os

# Debug: Print current directory and contents
st.write("Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# Load your trained model
try:
    with open('Streamlit/model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Create a simple Streamlit application
st.title('Credit Card Fraud Detection')

# Assuming you want to input features directly for prediction
input_string = st.text_input('Enter features separated by commas')

if st.button('Predict'):
    try:
        # Convert input string to list, assuming inputs are in the correct order
        feature_list = [float(x) for x in input_string.split(',')]
        
        # Define the columns based on your model's requirements
        columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                   'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                   'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        # Create a DataFrame with the correct columns
        features_df = pd.DataFrame([feature_list], columns=columns)
        
        # Use the model to make predictions
        prediction = model.predict(features_df)
        st.write('Prediction:', 'Fraud' if prediction[0] else 'Not Fraud')
    except Exception as e:
        st.write("Error processing input or predicting:", e)
