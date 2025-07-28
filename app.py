from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('svm_mobile_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input features (same order as training)
feature_names = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',
    'pc', 'px_height', 'px_width', 'ram', 'sc_h',
    'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read and convert inputs from form
        input_values = [float(request.form[feature]) for feature in feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform([input_values])
        prediction = model.predict(input_scaled)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Mobile Price Range: {int(prediction[0])}'
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f' Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)
