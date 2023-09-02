from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from flask_cors import CORS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
CORS(app)
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
    final = [predicted_depression_1]
    no_of_sessions = 10
    ans = {'RX': final, 'final_depression' : final, 'no_of_session': no_of_sessions}
    
    if predicted_depression_1 > 20:
        predicted_depression_2 = depression2(age, rmt, dep1, predicted_depression_1)
        final.append(predicted_depression_2)
        no_of_sessions = 20
        ans['RX'] = final
        ans['final_depression'] = predicted_depression_2
        
        if predicted_depression_2 > 20:
            predicted_depression_3 = depression3(age, rmt, dep1, predicted_depression_1, predicted_depression_2)
            final.append(predicted_depression_3)
            no_of_sessions = 30
            ans['RX'] = final
            ans['final_depression'] = predicted_depression_3
    return ans


@app.route('/')
def hello_world():
    return 'Hello World' 

# Create a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    rmt = data['rmt']
    dep1 = data['dep1']
    
    result = pred(age, rmt, dep1)
    
    return jsonify(result)

@app.route('/god', methods=['GET'])
def predictgod():
    age = int(request.args.get('age'))
    rmt = float(request.args.get('rmt'))
    dep1 = int(request.args.get('dep'))
    result = pred(age, rmt, dep1)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run()
