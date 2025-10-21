# years of experience > independent variable > x
# salary > dependent > y

# 1 import libraries
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# 2 siapkan data
X = [[1], [2], [3], [4]] # independent variable perlu list atau array 2 dimensi krn independent variable bisa 2
Y = [101, 102, 103, 104] # dependent prlu list atau array 1 dimensi

# 3 intansiasi objek
model = LinearRegression()
st.title("Prediksi Gaji")

# 4 training model / linear reggression
model.fit(X, Y)
input_user = st.number_input("Masukkan value: ")
prediction = model.predict([[input_user]])

# 6 visualisasi
fig, ax = plt.subplots()
prediksi_y = model.predict(X)
ax.scatter(X, Y)
ax.plot(X, prediksi_y)
ax.scatter([input_user], [prediction])
st.pyplot(fig)
st.metric(label="Gaji", value=prediction)
