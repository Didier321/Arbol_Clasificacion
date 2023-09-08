from flask import Flask, render_template, request
import joblib
import os
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_regresion_arbol2.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    N= int(request.form['N'])
    P= int(request.form['P'])
    K= int(request.form['K'])
    temperatura= int(request.form['temperatura'])
    humedad	= int(request.form['humedad'])
    ph	= int(request.form['ph'])
    precipitacion	= int(request.form['precipitacion'])
    
    
    pred_probabilities = np.array([[N,P, K, temperatura, humedad,ph, precipitacion]])
    
    prediccion = model.predict(pred_probabilities)
    
    mensaje = "La clasificacion es : "
    mensaje+= prediccion[0]
    
    return render_template('result.html', pred=mensaje)
    

if __name__ == '__main__':
    app.run()