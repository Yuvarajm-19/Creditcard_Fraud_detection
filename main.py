from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the model
model = joblib.load('credit_card_fraud_model.joblib')  # Assuming the model file is in the same directory

app = Flask(__name__)
# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['Time'], data['V1'], data['V2'], data['V3'], data['V4'], data['V5'],
        data['V6'], data['V7'], data['V8'], data['V9'], data['V10'], data['V11'],
        data['V12'], data['V13'], data['V14'], data['V15'], data['V16'], data['V17'],
        data['V18'], data['V19'], data['V20'], data['V21'], data['V22'], data['V23'],
        data['V24'], data['V25'], data['V26'], data['V27'], data['V28'], data['Amount']
    ])
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Specify host as '0.0.0.0' and port as 10000 for Render
    port = int(os.getenv("PORT", 5000
                         ))  # Get the port from Render environment variable, default to 10000
    app.run(host='0.0.0.0', port=port, debug=True)
