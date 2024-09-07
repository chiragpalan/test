# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.datasets import load_iris

# Title of the app
st.title("Iris Model Prediction Using Pickle File from GitHub")

# Load the Iris dataset for feature names
iris = load_iris()

# URL of the pickle file hosted on GitHub
url = 'https://github.com/chiragpalan/test/blob/main/iris_model.pkl'

# Function to load the model from GitHub
@st.cache_resource
def load_model_from_github(url):
    response = requests.get(url)
    model = pickle.loads(response.content)
    return model

# Load the model
model = load_model_from_github(url)

# User input for prediction
st.write("### Make a Prediction")
sepal_length = st.slider("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))

# Predict using user input
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)
predicted_class = iris.target_names[prediction[0]]

st.write(f"### Predicted Class: {predicted_class}")
