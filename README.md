# Customer Churn Prediction System

Machine learning system to predict telecom customer churn using Python (Flask) and Next.js.

## Features

- **Data Pipeline**  
  - Handles missing values (median imputation)  
  - Balances data using SMOTE (85:15 → 50:50)  
  - Encodes categories with OneHotEncoder  

- **ML Model**  
  - XGBoost (optimized via GridSearchCV)  
  - Accuracy: 82.3% | Recall (Churn): 78.5%  
  - Key features: `tenure`, `MonthlyCharges`, `Contract`  

- **Web Interface**  
  - Next.js form with real-time validation  
  - Dark theme UI  


## Setup

1. **Backend**  
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend**  
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Why We Used

- **XGBoost**: Best performance for imbalanced data  
- **SMOTE**: Preserves original data distribution  
- **Flask+Next.js**: Lightweight and scalable  

## Future Work

- Add SHAP explainability  
- Docker deployment  
