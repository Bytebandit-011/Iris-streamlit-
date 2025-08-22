import streamlit as st
import numpy as np
import joblib

with open("model.joblib", 'rb') as f:
    model = joblib.load(f)

# Sliders for input features
sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0)
sepal_width = st.slider('Sepal width (cm)', 2.0, 4.5)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0)
petal_width = st.slider('Petal width (cm)', 0.1, 2.5)

# Predict button
if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.write(f'Prediction: {prediction[0]}')
