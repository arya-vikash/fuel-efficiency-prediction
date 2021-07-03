from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle
from model import predict_mpg

model=pickle.load(open('auto_mpg.pkl','rb'))

app=Flask(__name__)


# index page (default route)
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


# prediction route(result is rendered in index page)
@app.route('/predict', methods=['POST'])
def predict():
    
    vehicle= {
    'Cylinders': [int(request.form['Cylinders'])],
    'Displacement': [int(request.form['Displacement'])],
    'Horsepower': [int(request.form['Horsepower'])],
    'Weight': [int(request.form['Weight'])],
    'Acceleration': [int(request.form['Acceleration'])],
    'Model Year': [int(request.form['Model Year'])],
    'Origin': [int(request.form['Origin'])]
    }

    predictions = predict_mpg(vehicle, model)
    return render_template('index.html',data=predictions)

if __name__=='__main__':
    app.run()
