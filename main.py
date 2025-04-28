import numpy as np
import pandas as pd
from os import path


import joblib
import streamlit as st

st.title("Flower Classification Application")

filename="Iris_model.pkl"
modelPath= joblib.load(path.join("Model",filename))

sl= st.number_input("Insert sepal length")
sw= st.number_input("Insert sepal width")
pl= st.number_input("Insert petal length")
pw= st.number_input("Insert petal width")

if st.button("Predict"):
    pred=modelPath.predict(np.array([[sl,sw,pl,pw]]))
    st.write("The flower is:",pred[0])