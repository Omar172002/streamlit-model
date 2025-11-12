import streamlit as st
import requests
import numpy as np

st.title('Linear model client')
st.write('**Omar Arias Zepeda - A00830966**')
st.write('### Fórmula del modelo: Y = 2X + 1')

url = 'https://tensorflow-serving-iko2.onrender.com/v1/models/linear-model:predict'

def predict(values):
  x = np.array(values, dtype=np.float32).reshape(-1, 1)
  
  data = {'instances': x.tolist()}
  response = requests.post(url, json=data)
  
  print(response.text)
  return response

st.write('### Introduce los datos manualmente')
data_input = st.text_input('Ingresa valores separados por comas (ej: 0, 1, 2, 3)', '1.0, 2.0, 3.0, 4.0')
btnPredict = st.button('Predict')

if (btnPredict):
   try:
     # Convertir el texto a lista de números
     values = [float(x.strip()) for x in data_input.split(',')]
     prediction = predict(values)
     
     if prediction.status_code == 200:
       results = prediction.json()
       st.success('Predicciones exitosas!')
       
       # Mostrar resultados en una tabla
       st.write('### Resultados:')
       data_table = []
       for i, val in enumerate(values):
         predicted = results['predictions'][i][0]
         expected = 2 * val + 1
         data_table.append({
           'Entrada (X)': val,
           'Predicción (Y)': f'{predicted:.2f}',
           'Esperado (2X + 1)': f'{expected:.2f}'
         })
       st.table(data_table)
     else:
       st.error(f'Error: {prediction.status_code}')
   except Exception as e:
     st.error(f'Error: {str(e)}')