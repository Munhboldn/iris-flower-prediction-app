import streamlit as st
from joblib import load
import numpy as np

# Load the trained model
model = load('data/iris_model.joblib')

# Load images for Iris flower types
iris_setosa_img = 'images/iris_setosa.png'
iris_versicolor_img = 'images/iris_versicolor.png'
iris_virginica_img = 'images/iris_virginica.png'

# Title of the app
st.title('Iris Flower Prediction')

# Use two columns to split the input fields for sepals and petals
col1, col2 = st.columns(2)

with col1:
    st.subheader('Sepal Dimensions')
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=15.0, step=0.1)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=15.0, step=0.1)

with col2:
    st.subheader('Petal Dimensions')
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=15.0, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=15.0, step=0.1)

# Display the predict button and output in the center
col_center = st.columns([1, 1, 1])
with col_center[1]:
    if st.button('Predict'):
        inputs = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(inputs)
        flower_type = prediction[0]

        # Display the corresponding flower image based on the prediction
        if flower_type == 0:
            st.image(iris_setosa_img, caption="Iris Setosa")
        elif flower_type == 1:
            st.image(iris_versicolor_img, caption="Iris Versicolor")
        else:
            st.image(iris_virginica_img, caption="Iris Virginica")
