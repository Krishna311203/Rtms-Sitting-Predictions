from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

loaded_model1 = load_model('Depression_1.h5')
loaded_model2 = load_model('Depression_2.h5')
loaded_model3 = load_model('Depression_3.h5')

def depression1(age, rmt, dep):
    x_in = np.array([[rmt, age, dep]])
    pred = loaded_model1.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred

def depression2(age, rmt, dep, dep1):
    x_in = np.array([[rmt, age, dep, dep1]])
    pred = loaded_model2.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred

def depression3(age, rmt, dep, dep1, dep2):
    x_in = np.array([[rmt, age, dep, dep1, dep2]])
    pred = loaded_model3.predict(x_in)
    processed_pred = int(pred[0][0])
    return processed_pred

def pred(age, rmt, dep1):
    predicted_depression_1 = depression1(age, rmt, dep1)
    final = predicted_depression_1
    no_of_sessions = 10
    
    if predicted_depression_1 > 20:
        predicted_depression_2 = depression2(age, rmt, dep1, predicted_depression_1)
        final = predicted_depression_2
        no_of_sessions = 20
        
        if predicted_depression_2 > 20:
            predicted_depression_3 = depression3(age, rmt, dep1, predicted_depression_1, predicted_depression_2)
            final = predicted_depression_3
            no_of_sessions = 30
            
    ans = [final, no_of_sessions]
    return ans

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    age = data['age']
    rmt = data['rmt']
    dep1 = data['dep1']
    
    result = pred(age, rmt, dep1)
    
    response = {
        "predicted_depression": result[0],
        "no_of_sessions": result[1]
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')
