from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

xgb_classifier = joblib.load('xgb_classifier_model_syn.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        input_df = input_df[['EPDS_Q1', 'EPDS_Q2', 'EPDS_Q3', 'EPDS_Q4', 'EPDS_Q5', 
                             'EPDS_Q6', 'EPDS_Q7', 'EPDS_Q8', 'EPDS_Q9', 'EPDS_Q10']]

        # Make a prediction using the loaded model
        prediction = xgb_classifier.predict(input_df)

        predicted_label = int(prediction[0]) 
        risk_status = "at risk" if predicted_label == 1 else "not at risk"

        response = {'prediction': risk_status}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
