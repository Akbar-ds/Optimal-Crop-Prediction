
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.cluster
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Streamlit app for crop prediction
st.title('Optimal Crop Recommendation System')
st.subheader('Adjust the values to get prediction')

# File uploader for dataset
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.warning('Please upload a CSV file to continue.')
    st.stop()

# Assuming data is loaded correctly from file or upload
data.hist(figsize=(12,12), layout=(4,4), bins=20) 

# Preprocessing and K-means clustering
x = data.drop(['label'], axis=1).values

# Determine Optimum number of clusters by elbow method
plt.rcParams['figure.figsize'] = (10,4)
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method', fontsize=15)
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.show()

# Selecting number of clusters based on elbow method
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)
w = pd.concat([pd.DataFrame(y_means, columns=['cluster']), data['label']], axis=1)

for i in range(0, 4):
    st.write(f'Crops in cluster {i}: {w[w["cluster"]==i]["label"].unique()}')
    st.write('---------------------------------------------------------------------------------------')

# Splitting dataset into features (x) and labels (y)
y = data['label']
x = data.drop(['label'], axis=1)
st.write(f"Shape of x: {x.shape}")
st.write(f"Shape of y: {y.shape}")

# Creating training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
st.write(f"Shape of x train: {x_train.shape}")
st.write(f"Shape of x test: {x_test.shape}")
st.write(f"Shape of y train: {y_train.shape}")
st.write(f"Shape of y test: {y_test.shape}")

# Creating and training the predictive model (Logistic Regression)
model = LogisticRegression()
model.fit(x_train, y_train)

# Input sliders for environmental factors
N = st.slider('Nitrogen content in soil', min_value=0, max_value=200, value=100)
P = st.slider('Phosphorus content in soil', min_value=0, max_value=150, value=50)
K = st.slider('Potassium content in soil', min_value=0, max_value=250, value=100)
temperature = st.slider('Temperature in Celsius', min_value=0, max_value=50, value=25)
humidity = st.slider('Relative humidity in %', min_value=0, max_value=100, value=50)
ph = st.slider('pH value of the soil', min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.slider('Rainfall in mm', min_value=0, max_value=300, value=100)

# Predicting the crop using the Logistic Regression model
prediction = model.predict(np.array([[N, P, K, temperature, humidity, ph, rainfall]]))

st.subheader('Predicted Crop:')
st.markdown(f'<p style="font-size:30px; font-weight:bold;">{prediction[0]}</p>', unsafe_allow_html=True)


