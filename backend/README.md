# Customer Churn Prediction Backend

This backend implements a comprehensive machine learning workflow for customer churn prediction. The system performs data analysis, preprocessing, model training, and comparison, and exposes the final model through a REST API.

## Workflow Overview

1. **Data Analysis and Preprocessing**
   - Exploratory data analysis with visualizations
   - Missing value detection and imputation
   - Feature engineering and selection
   - Class imbalance handling with SMOTE

2. **Model Training and Comparison**
   - Compares multiple algorithms:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - CatBoost
   - Hyperparameter optimization with grid search
   - Model evaluation with various metrics

3. **Model Explainability**
   - Feature importance analysis
   - SHAP value explanations
   - Visualization of model decisions

4. **API Service**
   - RESTful endpoints for predictions
   - Access to model insights and visualizations
   - Model performance metrics

## Directory Structure

```
backend/
├── dataset/
│   ├── telco_customer_churn.csv  # Raw data
│   └── processed/                # Processed data and visualizations
├── model/
│   ├── model.joblib              # Trained model
│   ├── model_comparison/         # Model comparison results
│   └── shap/                     # SHAP explanations
├── app.py                        # Flask API
├── data_preprocessing.py         # Data preprocessing module
├── model_training.py             # Model training module
├── main.py                       # Main workflow orchestrator
├── run.sh                        # Shell script for Unix/Mac
├── run.bat                       # Batch script for Windows
└── requirements.txt              # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Easy Runner Scripts

For convenience, we've added runner scripts to make it easy to run different parts of the workflow:

#### On Linux/Mac:

```bash
# Make the script executable first
chmod +x run.sh

# Run the complete workflow
./run.sh all

# Run only data preprocessing
./run.sh preprocess

# Run only model training (assumes preprocessing is done)
./run.sh train

# Start only the API (assumes preprocessing and training are done)
./run.sh api

# Clean all generated data and models
./run.sh clean

# Add --debug flag for more verbose logging
./run.sh all --debug
```

#### On Windows:

```cmd
# Run the complete workflow
run.bat all

# Run only data preprocessing
run.bat preprocess

# Run only model training (assumes preprocessing is done)
run.bat train

# Start only the API (assumes preprocessing and training are done)
run.bat api

# Clean all generated data and models
run.bat clean

# Add --debug flag for more verbose logging
run.bat all --debug
```

### Running the ML Workflow Manually

To run the complete workflow manually:

```bash
python main.py --input-data dataset/telco_customer_churn.csv
```

Options:
- `--input-data`: Path to input dataset CSV (default: "dataset/telco_customer_churn.csv")
- `--processed-data-dir`: Directory for processed data (default: "dataset/processed")
- `--model-dir`: Directory for saving models (default: "model")
- `--skip-preprocessing`: Skip preprocessing step
- `--skip-training`: Skip model training step
- `--run-api`: Start the API after training
- `--debug`: Enable debug mode with more verbose logging

### Running Only the API

If you've already run the workflow and just want to start the API:

```bash
python main.py --skip-preprocessing --skip-training --run-api
```

## API Endpoints

- **POST /predict**: Get churn prediction for a customer
- **GET /model-info**: Get information about the trained model
- **GET /feature-importance**: Get feature importance data
- **GET /model-comparison**: Get model comparison results
- **GET /visualizations/<filename>**: Get a specific visualization
- **GET /available-visualizations**: List all available visualizations
- **GET /data-summary**: Get dataset summary statistics
- **GET /health**: API health check

## Example API Usage

### Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "Contract": "Month-to-month",
    "MonthlyCharges": 39.90,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "TotalCharges": 542.4
  }'
```

## Troubleshooting

If you encounter any issues:

1. Use the `--debug` flag to get more detailed logging
2. Check the `ml_workflow.log` file for error messages
3. Make sure the dataset exists at the specified location
4. Ensure all dependencies are installed correctly
5. Try running the steps separately (preprocess, then train, then API) 