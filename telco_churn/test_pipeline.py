#!/usr/bin/env python3
"""
Test script for the Telco Churn Pipeline
"""

import requests
import json
import time
import os

def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("="*50)
    print("TESTING DATA PREPROCESSING")
    print("="*50)
    
    try:
        from data_preprocessing import ChurnDataPreprocessor
        
        preprocessor = ChurnDataPreprocessor()
        results = preprocessor.run_full_pipeline()
        
        # Check if files were created
        required_files = [
            "processed_data/X_resampled.npy",
            "processed_data/y_resampled.npy",
            "processed_data/encoder.joblib",
            "processed_data/scaler.joblib",
            "processed_data/feature_names.joblib"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print("✅ Data preprocessing completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_model_training():
    """Test model training pipeline"""
    print("\n" + "="*50)
    print("TESTING MODEL TRAINING")
    print("="*50)
    
    try:
        from model_training import ChurnModelTrainer
        
        trainer = ChurnModelTrainer()
        results = trainer.run_full_training_pipeline()
        
        # Check if model files were created
        required_files = [
            "models/best_model.joblib",
            "models/model_summary.json",
            "models/training_results.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print("✅ Model training completed successfully!")
            print(f"Best model: {results['best_model_name']}")
            return True
            
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_api():
    """Test API functionality"""
    print("\n" + "="*50)
    print("TESTING API")
    print("="*50)
    
    # Sample input data
    test_data = {
        "tenure": 12,
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": 0
    }
    
    api_url = "http://127.0.0.1:5001"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        print("Testing prediction endpoint...")
        response = requests.post(
            f"{api_url}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"✅ Prediction successful: {prediction_data}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start it with: python api.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TELCO CHURN PIPELINE TESTING")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists("telco_customer_churn.csv"):
        print("❌ Dataset file 'telco_customer_churn.csv' not found!")
        print("Please ensure the dataset is in the telco_churn directory.")
        return
    
    # Test preprocessing
    preprocessing_success = test_data_preprocessing()
    
    # Test model training (only if preprocessing succeeded)
    if preprocessing_success:
        training_success = test_model_training()
    else:
        print("⏩ Skipping model training due to preprocessing failure")
        training_success = False
    
    # Test API (only if training succeeded)
    if training_success:
        print("\n📡 To test the API, please:")
        print("1. Run 'python api.py' in a separate terminal")
        print("2. Then run this test again or use the test_api() function")
        
        # Check if API is already running
        try:
            response = requests.get("http://127.0.0.1:5001/health", timeout=2)
            if response.status_code == 200:
                api_success = test_api()
            else:
                print("🔧 API server detected but not responding correctly")
                api_success = False
        except:
            print("🔧 API server not detected. Please start it manually.")
            api_success = False
    else:
        print("⏩ Skipping API test due to training failure")
        api_success = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Data Preprocessing: {'✅ PASS' if preprocessing_success else '❌ FAIL'}")
    print(f"Model Training: {'✅ PASS' if training_success else '❌ FAIL'}")
    print(f"API Testing: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if preprocessing_success and training_success:
        print("\n🎉 Pipeline is ready for use!")
        print("🚀 Start the API with: python api.py")
        print("🌐 Frontend can connect to: http://127.0.0.1:5001")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main() 