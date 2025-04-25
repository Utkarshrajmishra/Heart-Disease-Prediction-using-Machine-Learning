from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, origins="http://localhost:5173") 


# Load the trained model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Updated Features List (Including all features)
        features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                    "exang", "oldpeak", "slope", "ca", "thal"]

        # Extract user input
        input_data = [data[feature] for feature in features]

        # Convert to NumPy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Make Prediction
        prediction = model.predict(input_array)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
