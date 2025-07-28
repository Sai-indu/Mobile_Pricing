from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('svm_mobile_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array([data['features']])
    
    # Scale features
    input_scaled = scaler.transform(input_features)

    # Predict
    prediction = model.predict(input_scaled)
    return jsonify({'price_range': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
    
