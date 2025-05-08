import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train_path, y_train_path, X_test_path, y_test_path, output_path):
    """Train and evaluate XGBoost with hyperparameter tuning."""
    try:
        # Load data
        logger.info("Loading preprocessed data...")
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Hyperparameter grid (optimized for churn prediction)
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'scale_pos_weight': [1, (len(y_train) - sum(y_train)) / sum(y_train)]
        }

        # Train model
        logger.info("Starting model training...")
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Evaluate
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Save model
        Path(output_path).mkdir(exist_ok=True)
        joblib.dump(best_model, f"{output_path}/model.joblib")
        logger.info(f"Model saved to {output_path}/model.joblib")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model(
        X_train_path="dataset/processed/X_train.csv",
        y_train_path="dataset/processed/y_train.csv",
        X_test_path="dataset/processed/X_test.csv",
        y_test_path="dataset/processed/y_test.csv",
        output_path="model"
    )
