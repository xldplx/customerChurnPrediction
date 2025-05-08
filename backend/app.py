from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from pathlib import Path
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow all origins (for development)

# Load artifacts
try:
    preprocessor = joblib.load("dataset/processed/preprocessor.joblib")
    model = joblib.load("model/model.joblib")
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for churn prediction."""
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")

        # Convert to DataFrame and validate columns
        required_columns = ["tenure", "Contract", "MonthlyCharges", "InternetService", "PaymentMethod", "TotalCharges"]
        for col in required_columns:
            if col not in data:
                logger.error(f"Missing required column: {col}")
                return jsonify({"error": f"Missing required field: {col}", "status": "failed"}), 400

        df = pd.DataFrame([data])
        df_processed = preprocessor.transform(df)
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0, 1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
