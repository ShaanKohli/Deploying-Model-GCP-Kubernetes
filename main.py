from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('pycaret_classification')
cols = ['Gender', 'Occupation', 'History_Hypertension', 'Med_BP', 'Is_Consult',
       'Is_Treatment', 'Year_Treatment', 'Is_HealthyDiet', 'Is_FastFood',
       'Is_Salt', 'Num_Salt', 'Is_Smoke', 'Is_Caffeine', 'Is_Alcohol',
       'Is_Hospital', 'Physical_Activity', 'Sex', 'Age_y', 'DIAGNOSIS',
       'HEIGHT', 'WEIGHT', 'BMI', 'BMI_VALUE', 'PULSE', 'MEDICINE1',
       'MEDICINE2']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Hypertension Class will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
