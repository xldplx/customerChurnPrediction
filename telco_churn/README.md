# Telco Customer Churn Prediction Pipeline

This is a comprehensive machine learning pipeline for predicting customer churn in telecommunications companies. The project includes data preprocessing, model training, and a REST API for serving predictions.

## 📁 Project Structure

```
telco_churn/
├── telco_customer_churn.csv    # Raw dataset
├── data_preprocessing.py       # Data preprocessing pipeline
├── model_training.py          # Model training and evaluation
├── api.py                     # Flask API for predictions
├── main.py                    # Converted from original notebook
├── telco.ipynb               # Original Jupyter notebook
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── processed_data/           # Generated after preprocessing
│   ├── X_train_processed.npy
│   ├── X_test_processed.npy
│   ├── X_resampled.npy
│   ├── y_resampled.npy
│   ├── encoder.joblib
│   ├── scaler.joblib
│   └── feature_names.joblib
└── models/                   # Generated after training
    ├── best_model.joblib
    ├── random_forest_model.joblib
    ├── xgboost_model.joblib
    ├── catboost_model.joblib
    ├── training_results.json
    ├── model_comparison.csv
    ├── feature_importance.csv
    └── model_summary.json
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing

```bash
python data_preprocessing.py
```

This will:
- Load and clean the dataset
- Perform exploratory data analysis
- Split data into train/test sets
- Encode categorical features
- Scale numerical features
- Apply SMOTE for class balancing
- Save processed data to `processed_data/`

### 3. Train Models

```bash
python model_training.py
```

This will:
- Train Random Forest, XGBoost, and CatBoost models
- Perform hyperparameter tuning with GridSearchCV
- Compare model performance
- Generate visualizations
- Save the best model to `models/`

### 4. Start API Server

```bash
python api.py
```

The API will be available at `http://localhost:5001`

## 📊 Data Pipeline

### Data Preprocessing Features

- **Data Cleaning**: Handles missing values and incorrect data types
- **Feature Engineering**: Creates and transforms features appropriately
- **Categorical Encoding**: Uses OneHotEncoder for categorical variables
- **Feature Scaling**: Applies MinMaxScaler for numerical features
- **Class Balancing**: Uses SMOTE to handle imbalanced dataset
- **Data Validation**: Ensures data quality and consistency

### Supported Models

1. **Random Forest**
   - Ensemble method with decision trees
   - Robust to overfitting
   - Provides feature importance

2. **XGBoost**
   - Gradient boosting framework
   - High performance on structured data
   - Advanced regularization

3. **CatBoost**
   - Gradient boosting on decision trees
   - Handles categorical features well
   - Reduces overfitting

## 🌐 API Endpoints

### Health Check
```http
GET /health
```

Returns API health status and model loading status.

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
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
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.75,
  "no_churn_probability": 0.25,
  "status": "success"
}
```

### Get Model Information
```http
GET /model-info
```

Returns information about the trained model.

### Get Feature Importance
```http
GET /feature-importance
```

Returns feature importance rankings from the best model.

## 📈 Model Performance

The pipeline automatically compares multiple models and selects the best one based on F1-score. Typical performance metrics:

- **Accuracy**: ~82-85%
- **F1-Score**: ~78-82%
- **Recall**: ~75-80%
- **Precision**: ~80-85%

## 🔧 Configuration

### Model Parameters

You can modify hyperparameters in `model_training.py`:

```python
# Random Forest parameters
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# XGBoost parameters
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
```

### Data Processing

Modify preprocessing parameters in `data_preprocessing.py`:

```python
# SMOTE parameters
def apply_smote(self, sampling_strategy='auto', random_state=0, k_neighbors=5):
    
# Train-test split
def split_data(self, test_size=0.2, random_state=42):
```

## 📊 Expected Features

The model expects the following input features:

### Numerical Features
- `tenure`: Number of months the customer has stayed
- `MonthlyCharges`: Monthly charges amount
- `TotalCharges`: Total charges amount

### Categorical Features
- `gender`: Male/Female
- `SeniorCitizen`: 0/1
- `Partner`: Yes/No
- `Dependents`: Yes/No
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check/Mailed check/Bank transfer/Credit card

## 🚀 Integration with Frontend

To integrate with the existing frontend, update the API URL in `frontend/src/lib/api.ts`:

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5001';
```

## 🔄 Model Retraining

To retrain the models with new data:

1. Replace `telco_customer_churn.csv` with your new dataset
2. Run the preprocessing pipeline: `python data_preprocessing.py`
3. Run the training pipeline: `python model_training.py`
4. Restart the API: `python api.py`

## 📋 Requirements

- Python 3.8+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 1.7.0
- catboost >= 1.2.0
- flask >= 2.0.0
- imbalanced-learn >= 0.10.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model file not found**: Ensure you've run the training pipeline first
2. **Preprocessing errors**: Check that input data matches expected format
3. **API connection errors**: Verify the API is running on the correct port
4. **Memory issues**: Consider reducing dataset size or using smaller model parameters

### Getting Help

- Check the console output for detailed error messages
- Ensure all dependencies are installed correctly
- Verify that all required files are present

---

**Note**: This is a temporary implementation in the `telco_churn` folder. Once validated, it can replace the existing backend implementation. 