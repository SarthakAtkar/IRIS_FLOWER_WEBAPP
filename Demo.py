import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""

# Simple Iris Flower Prediction App
         
This app predits the Iris flower type!
""")

st.sidebar.header("USer Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width',2.0, 4.4, 3.4)
    petal_width = st.sidebar.slider('petal_width',1.0, 3.4, 1.4)
    petal_length = st.sidebar.slider('petal_length',2.0, 4.4, 3.4)
    data = {'sepal_length': sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}
    features = pd.DataFrame(data, index = [0])
    return features

df = user_input_features()

st.subheader('USer Input parameters')
st.write(df)

iris = datasets.load_iris()
x = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(x, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class Labels and Their Corresponding Index Number')
st.write(iris.target_names)

st.subheader("PRedictions")
st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)