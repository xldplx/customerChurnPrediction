from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from pathlib import Path
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow all origins (for development)

# Application paths
MODEL_DIR = "model"
DATA_DIR = "dataset/processed"
VISUALIZATIONS_DIR = f"{DATA_DIR}/visualizations"

# Ensure directories exist
for dir_path in [MODEL_DIR, DATA_DIR, VISUALIZATIONS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Load artifacts
try:
    # Load main model and preprocessor
    preprocessor = joblib.load(f"{DATA_DIR}/preprocessor.joblib")
    model = joblib.load(f"{MODEL_DIR}/model.joblib")
    
    # Try to load feature names (for feature importance)
    try:
        feature_names = joblib.load(f"{DATA_DIR}/feature_names.joblib")
    except:
        feature_names = None
        logger.warning("Feature names not found. Feature importance will not be available.")
    
    # Try to load model summary
    try:
        with open(f"{MODEL_DIR}/model_summary.json", 'r') as f:
            model_summary = json.load(f)
    except:
        model_summary = None
        logger.warning("Model summary not found.")
    
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for churn prediction."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with {len(data)} features: {data}")

        # Convert to DataFrame and validate
        df = pd.DataFrame([data])
        
        # Check for required columns
        required_columns = [
            "tenure", "Contract", "MonthlyCharges", "InternetService", "PaymentMethod",
            "TotalCharges", "gender", "Partner", "Dependents", "PhoneService", 
            "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_columns)}", 
                "status": "failed"
            }), 400
        
        # Log column data types
        logger.info(f"Column data types: {df.dtypes.to_dict()}")
        
        # Ensure numeric fields are numeric
        numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as e:
                    logger.error(f"Error converting {col} to numeric: {e}. Value: {df[col].values[0]}")
                    return jsonify({
                        "error": f"Invalid value for {col}. Must be a number.",
                        "status": "failed"
                    }), 400
        
        # Preprocess data
        try:
            logger.info("Starting preprocessing")
            df_processed = preprocessor.transform(df)
            logger.info(f"Preprocessing successful. Processed shape: {df_processed.shape}")
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            
            # Check categorical fields for valid values
            categorical_fields = [
                "Contract", "InternetService", "PaymentMethod", "gender", "Partner", 
                "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", 
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                "StreamingMovies", "PaperlessBilling"
            ]
            
            for field in categorical_fields:
                if field in df.columns:
                    value = df[field].values[0]
                    logger.info(f"Field {field} has value: '{value}'")
            
            return jsonify({
                "error": "Failed to preprocess input data. Ensure values are in the correct format.", 
                "status": "failed"
            }), 400
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0, 1]
        
        # Return prediction results
        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return information about the trained model."""
    if model_summary:
        return jsonify({
            "model_info": model_summary,
            "status": "success"
        })
    else:
        return jsonify({
            "error": "Model summary not available",
            "status": "failed"
        }), 404

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance data."""
    try:
        # Check if feature importance file exists
        importance_path = f"{MODEL_DIR}/feature_importance.csv"
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            return jsonify({
                "feature_importance": importance_df.to_dict(orient="records"),
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Feature importance data not available",
                "status": "failed"
            }), 404
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/model-comparison", methods=["GET"])
def model_comparison():
    """Return model comparison data."""
    try:
        comparison_path = f"{MODEL_DIR}/model_comparison/model_comparison.csv"
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            return jsonify({
                "model_comparison": comparison_df.to_dict(orient="records"),
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Model comparison data not available",
                "status": "failed"
            }), 404
    except Exception as e:
        logger.error(f"Error retrieving model comparison: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/visualizations/<path:filename>", methods=["GET"])
def get_visualization(filename):
    """Serve visualization images."""
    try:
        # Look in different directories for the visualization
        possible_paths = [
            f"{VISUALIZATIONS_DIR}/{filename}",
            f"{MODEL_DIR}/visualizations/{filename}",
            f"{MODEL_DIR}/{filename}",
            f"{MODEL_DIR}/model_comparison/{filename}",
            f"{MODEL_DIR}/shap/{filename}"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path, mimetype='image/png')
        
        return jsonify({
            "error": f"Visualization {filename} not found",
            "status": "failed"
        }), 404
    except Exception as e:
        logger.error(f"Error retrieving visualization: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/available-visualizations", methods=["GET"])
def list_visualizations():
    """List all available visualizations with categories."""
    try:
        # Directories to check for visualizations
        vis_dirs = [
            f"{VISUALIZATIONS_DIR}",
            f"{MODEL_DIR}/model_comparison",
            f"{MODEL_DIR}/shap"
        ]
        
        # Find all image files
        visualization_files = []
        for directory in vis_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        # Add to list with full path
                        rel_path = os.path.join(directory, file).replace("\\", "/")
                        if directory.startswith(VISUALIZATIONS_DIR):
                            category = get_category(file)
                            visualization_files.append({
                                "filename": file,
                                "path": f"/visualizations/{file}",
                                "category": category
                            })
                        elif "model_comparison" in directory:
                            visualization_files.append({
                                "filename": file,
                                "path": f"/visualizations/{file}",
                                "category": "Model Comparison"
                            })
                        elif "shap" in directory:
                            visualization_files.append({
                                "filename": file,
                                "path": f"/visualizations/{file}",
                                "category": "Model Interpretation"
                            })
        
        # Group by category
        categories = {}
        for vis in visualization_files:
            cat = vis["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(vis)
        
        # Return the list of visualizations grouped by category
        return jsonify({
            "categories": categories,
            "all_visualizations": visualization_files,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error listing visualizations: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

def get_category(filename):
    # Direct mapping
    if "churn_distribution" in filename:
        return "Data Analysis"
    elif "correlation_matrix" in filename:
        return "Data Analysis"
    elif "churn_correlation" in filename:
        return "Data Analysis"
    elif "churn_by_" in filename:
        return "Churn Patterns"
    elif any(x in filename for x in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]):
        return "Churn Patterns"
    elif "train_test" in filename or "smote" in filename:
        return "Data Processing"
    elif "xgboost_vs_randomforest" in filename:
        return "Model Comparison"
    elif any(x in filename for x in ["cm_", "confusion"]):
        return "Model Evaluation"
    elif any(x in filename for x in ["feature_importance", "feature_coefficient"]):
        return "Feature Importance"
    else:
        return "Other Visualizations"

@app.route("/data-summary", methods=["GET"])
def data_summary():
    """Return data summary statistics."""
    try:
        summary_path = f"{DATA_DIR}/analysis_summary.txt"
        overview_path = f"{DATA_DIR}/data_overview.csv"
        
        result = {}
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary_text = f.read()
                # Parse the summary text into a structured format
                summary_lines = summary_text.strip().split('\n')
                for line in summary_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result[key.strip()] = value.strip()
        
        if os.path.exists(overview_path):
            overview_df = pd.read_csv(overview_path)
            result["data_overview"] = overview_df.to_dict(orient="index")
        
        if result:
            return jsonify({
                "data_summary": result,
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Data summary not available",
                "status": "failed"
            }), 404
    except Exception as e:
        logger.error(f"Error retrieving data summary: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
