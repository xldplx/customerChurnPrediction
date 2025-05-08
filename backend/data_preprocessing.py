import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(input_path, output_path):
    """Preprocess Telco Churn data with robust error handling."""
    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Clean TotalCharges (handle missing/coercion errors)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Target encoding
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop(['customerID', 'Churn'], axis=1)
        
        # Define features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply preprocessing
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        # Handle imbalance (skip if SMOTE fails)
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info("SMOTE applied successfully.")
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
        
        # Save artifacts
        Path(output_path).mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, f"{output_path}/preprocessor.joblib")
        pd.DataFrame(X_train).to_csv(f"{output_path}/X_train.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{output_path}/y_train.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{output_path}/X_test.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{output_path}/y_test.csv", index=False)
        logger.info(f"Artifacts saved to {output_path}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data(
        input_path="dataset/telco_customer_churn.csv",
        output_path="dataset/processed"
    )
