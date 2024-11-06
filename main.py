from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the model (relative path, assuming it's in the same directory as main.py)
model = joblib.load('credit_card_fraud_model.joblib')

app = Flask(__name__)


# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming JSON data
    features = np.array([
        data['Time'], data['V1'], data['V2'], data['V3'], data['V4'], data['V5'],
        data['V6'], data['V7'], data['V8'], data['V9'], data['V10'], data['V11'],
        data['V12'], data['V13'], data['V14'], data['V15'], data['V16'], data['V17'],
        data['V18'], data['V19'], data['V20'], data['V21'], data['V22'], data['V23'],
        data['V24'], data['V25'], data['V26'], data['V27'], data['V28'], data['Amount']
    ])

    # Make prediction
    prediction = model.predict([features])

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
