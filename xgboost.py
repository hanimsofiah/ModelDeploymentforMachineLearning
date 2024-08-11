# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
df = pd.read_csv('creditcard.csv')

# %%
df.head()

# %%
df['Time'].nunique()

# %%
df.columns

# %%
df.describe().round().T

# %%
df['class'].value_counts()

# %%
df.info()

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
X = df.drop(columns = ['class'])
y = df['class']

# %%
preprocessor = ColumnTransformer(transformers=[('num',StandardScaler(),X.columns)])

# %%
from xgboost import XGBClassifier

# %%
my_model = XGBClassifier(random_state = 12)

pipeline = Pipeline(steps=[('process',preprocessor),('model',my_model)])
pipeline.fit(X,y)

# %%
from sklearn.model_selection import cross_val_score

score = cross_val_score(pipeline, X,y,cv=5)
print(score)
print(score.mean())

# %% [markdown]
# Model Deployment

# %%
import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a simple Streamlit application
st.title('Credit Card Fraud Detection')

# Assuming you want to input features directly for prediction
input_string = st.text_input('Enter features separated by commas')

if st.button('Predict'):
    try:
        # Convert input string to list, assuming inputs are in the correct order
        feature_list = [float(x) for x in input_string.split(',')]
        
        # Define the columns based on your model's requirements, exclude 'Class'
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



