# 1 import library
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 2 menyiapkan data
X = [[1], [2], [3], [4]] #tipe data list 2 dimensi
y = [101, 102, 103, 104] #tipe data list 1 dimensi

# 3 instansiasi objek
model = LinearRegression()

st.title("Prediksi Gaji")

# 4 training
model.fit(X, y)
input_user = st.number_input("Masukan value :")

prediction = model.predict([[input_user]])

# 6 visualisasi
fig, ax = plt.subplots()

prediksi_y = model.predict(X)
ax.scatter(X, y)
ax.plot(X, prediksi_y)
ax.scatter([input_user], [prediction])

st.pyplot(fig)
st.metric(label="Gaji", value=prediction)
