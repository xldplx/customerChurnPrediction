#!/usr/bin/env python3
"""
Simple API for Telco Customer Churn Prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model artifacts
model = None
encoder = None
scaler = None
feature_names = None

def load_model_artifacts():
    """Load all model artifacts"""
    global model, encoder, scaler, feature_names
    
    try:
        # Load the best model
        model_path = "models/best_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ Model loaded successfully")
        else:
            print("⚠ Model file not found!")
            return False
        
        # Load preprocessing artifacts
        encoder_path = "processed_data/encoder.joblib"
        scaler_path = "processed_data/scaler.joblib"
        feature_names_path = "processed_data/feature_names.joblib"
        
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            print("✓ Encoder loaded successfully")
        else:
            print("⚠ Encoder file not found!")
            return False
            
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠ Scaler file not found!")
            return False
            
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            print("✓ Feature names loaded successfully")
        else:
            print("⚠ Feature names file not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Define expected columns
        categorical_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Ensure numerical columns are numeric
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing TotalCharges
        if 'TotalCharges' in df.columns and df['TotalCharges'].isna().any():
            df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
        
        # Get existing categorical columns
        existing_cat_cols = [col for col in categorical_columns if col in df.columns]
        
        # Encode categorical features
        if encoder and existing_cat_cols:
            cat_encoded = encoder.transform(df[existing_cat_cols])
            cat_df = pd.DataFrame(
                cat_encoded,
                columns=encoder.get_feature_names_out(existing_cat_cols),
                index=df.index
            )
            
            # Remove original categorical columns and add encoded ones
            df_processed = df.drop(columns=existing_cat_cols)
            df_processed = pd.concat([df_processed, cat_df], axis=1)
        else:
            df_processed = df
        
        # Scale features
        if scaler:
            df_scaled = pd.DataFrame(
                scaler.transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
        else:
            df_scaled = df_processed
        
        # Ensure we have all required features
        if feature_names:
            # Add missing columns with zeros
            for col in feature_names:
                if col not in df_scaled.columns:
                    df_scaled[col] = 0
            
            # Reorder columns to match training data
            df_scaled = df_scaled[feature_names]
        
        return df_scaled.values
        
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': encoder is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'failed'
            }), 500
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No input data provided',
                'status': 'failed'
            }), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Return results
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0]),
            'status': 'success'
        })
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        # Load model summary if available
        summary_path = "models/model_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return jsonify({
                'model_info': summary,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Model summary not available',
                'status': 'failed'
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': f'Error retrieving model info: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance"""
    try:
        importance_path = "models/feature_importance.csv"
        if os.path.exists(importance_path):
            df = pd.read_csv(importance_path)
            return jsonify({
                'feature_importance': df.to_dict(orient='records'),
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Feature importance data not available',
                'status': 'failed'
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': f'Error retrieving feature importance: {str(e)}',
            'status': 'failed'
        }), 500

if __name__ == '__main__':
    print("Loading model artifacts...")
    if load_model_artifacts():
        print("✓ All artifacts loaded successfully!")
        print("Starting Flask API server...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("✗ Failed to load model artifacts. Please train the model first.")
        print("Run: python data_preprocessing.py && python model_training.py") 