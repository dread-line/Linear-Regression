import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv('indonesian_salary_by_region.csv')

st.title("PREDIKSIII")

region = df['REGION'].unique()

input_region = st.selectbox("Pilih Provinsi", options=region)


df_selected = df[df['REGION'] == input_region].copy()

x = df_selected[['YEAR']]
y = df_selected['SALARY']

model = LinearRegression()
model.fit(x, y)

min_year = df_selected['YEAR'].min()
max_year = df_selected['YEAR'].max()
input_year = st.number_input("Masukkan tahun: ", min_value = min_year, value = max_year+1)
prediction = model.predict([[input_year]])

formated_salary = f"Rp {prediction[0]:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".")

st.metric(label=f'Prediksi gaji di provinsi {input_region} di tahun {input_year}', value=formated_salary)
st.subheader("Grafik Prediksi")

fig, ax = plt.subplots()
ax.scatter(x, y, label=f'data historis ump/umk {input_region} 1997-2025', color='blue')
ax.plot(x, model.predict(x), color='yellow', label='Gars regresi linear')
ax.scatter([input_year], [prediction], color='red', label=f'prediksi tahun {input_year}')

ax.set_xlabel("Tahun")
ax.set_ylabel("Gaji")
ax.set_title(f'Tren kenaikan gaji di {input_region}')
ax.legend()
st.pyplot(fig)
# print(model.predict([[2030]]))
