import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

#import pickle file
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes',
       'region']
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        region = float(request.form.get('region'))

        new_data_scaled = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,\
                                   Classes,region]])
        
        result = model.predict(new_data_scaled)
        print(result[0])

        return render_template('home.html', results = result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")