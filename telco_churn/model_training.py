#!/usr/bin/env python3
"""
Model Training Pipeline for Telco Customer Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self, data_dir="processed_data"):
        """
        Initialize the model trainer
        """
        self.data_dir = data_dir
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Load training data (SMOTE resampled)
        self.X_train = np.load(f"{self.data_dir}/X_resampled.npy")
        self.y_train = np.load(f"{self.data_dir}/y_resampled.npy")
        
        # Load test data
        self.X_test = np.load(f"{self.data_dir}/X_test_processed.npy")
        self.y_test = np.load(f"{self.data_dir}/y_test_processed.npy")
        
        # Load feature names
        self.feature_names = joblib.load(f"{self.data_dir}/feature_names.joblib")
        
        print(f"Training data shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test data shape: X={self.X_test.shape}, y={self.y_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Check class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Training class distribution: {dict(zip(unique, counts))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self) -> Dict[str, Any]:
        """Train Random Forest with hyperparameter tuning"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)
        
        # Define parameters for tuning
        rf_params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        
        # GridSearchCV
        print("Performing hyperparameter tuning...")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        # Best model
        rf_model = rf_grid.best_estimator_
        
        # Predictions
        train_pred = rf_model.predict(self.X_train)
        test_pred = rf_model.predict(self.X_test)
        test_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        results = self._calculate_metrics("Random Forest", rf_model, train_pred, test_pred, test_proba)
        results['best_params'] = rf_grid.best_params_
        results['cv_score'] = rf_grid.best_score_
        
        # Store model and results
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = results
        
        print(f"Best Parameters: {rf_grid.best_params_}")
        print(f"CV Score: {rf_grid.best_score_:.4f}")
        
        return results
    
    def train_xgboost(self) -> Dict[str, Any]:
        """Train XGBoost with hyperparameter tuning"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST")
        print("="*50)
        
        # Parameters grid
        xgb_params = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
        
        # GridSearchCV
        print("Performing hyperparameter tuning...")
        xgb_grid = GridSearchCV(
            XGBClassifier(eval_metric='logloss', random_state=42),
            xgb_params,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        xgb_grid.fit(self.X_train, self.y_train)
        
        # Best model
        xgb_model = xgb_grid.best_estimator_
        
        # Predictions
        train_pred = xgb_model.predict(self.X_train)
        test_pred = xgb_model.predict(self.X_test)
        test_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        results = self._calculate_metrics("XGBoost", xgb_model, train_pred, test_pred, test_proba)
        results['best_params'] = xgb_grid.best_params_
        results['cv_score'] = xgb_grid.best_score_
        
        # Store model and results
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = results
        
        print(f"Best Parameters: {xgb_grid.best_params_}")
        print(f"CV Score: {xgb_grid.best_score_:.4f}")
        
        return results
    
    def train_catboost(self) -> Dict[str, Any]:
        """Train CatBoost with hyperparameter tuning"""
        print("\n" + "="*50)
        print("TRAINING CATBOOST")
        print("="*50)
        
        # CatBoost parameters grid
        cat_params = {
            "iterations": [100, 200],
            "depth": [3, 5, 7],
            "learning_rate": [0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5]
        }
        
        # GridSearchCV
        print("Performing hyperparameter tuning...")
        cat_grid = GridSearchCV(
            CatBoostClassifier(random_state=42, verbose=0),
            cat_params,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        cat_grid.fit(self.X_train, self.y_train)
        
        # Best model
        cat_model = cat_grid.best_estimator_
        
        # Predictions
        train_pred = cat_model.predict(self.X_train)
        test_pred = cat_model.predict(self.X_test)
        test_proba = cat_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        results = self._calculate_metrics("CatBoost", cat_model, train_pred, test_pred, test_proba)
        results['best_params'] = cat_grid.best_params_
        results['cv_score'] = cat_grid.best_score_
        
        # Store model and results
        self.models['CatBoost'] = cat_model
        self.results['CatBoost'] = results
        
        print(f"Best Parameters: {cat_grid.best_params_}")
        print(f"CV Score: {cat_grid.best_score_:.4f}")
        
        return results
    
    def _calculate_metrics(self, model_name: str, model, train_pred, test_pred, test_proba) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a model"""
        
        # Training metrics
        train_accuracy = accuracy_score(self.y_train, train_pred)
        
        # Test metrics
        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred)
        test_recall = recall_score(self.y_test, test_pred)
        test_precision = precision_score(self.y_test, test_pred)
        
        # Classification report
        class_report = classification_report(
            self.y_test, test_pred, 
            target_names=['No Churn', 'Churn'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'test_predictions': test_pred.tolist(),
            'test_probabilities': test_proba.tolist()
        }
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        
        return results
    
    def compare_models(self):
        """Compare all trained models and select the best one"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if not self.results:
            print("No models trained yet!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_Accuracy': results['train_accuracy'],
                'Test_Accuracy': results['test_accuracy'],
                'F1_Score': results['test_f1'],
                'Recall': results['test_recall'],
                'Precision': results['test_precision'],
                'CV_Score': results.get('cv_score', 0)
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print(self.comparison_df.round(4))
        
        # Find best model based on F1 score
        best_idx = self.comparison_df['F1_Score'].idxmax()
        self.best_model_name = self.comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        self.best_score = self.comparison_df.loc[best_idx, 'F1_Score']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best F1 Score: {self.best_score:.4f}")
        
        return self.comparison_df
    
    def plot_model_comparison(self, save_plot=True):
        """Create visualization comparing model performance"""
        if not hasattr(self, 'comparison_df'):
            self.compare_models()
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(self.comparison_df['Model'], self.comparison_df['Test_Accuracy'], 
                       color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[0, 1].bar(self.comparison_df['Model'], self.comparison_df['F1_Score'], 
                       color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[1, 0].bar(self.comparison_df['Model'], self.comparison_df['Recall'], 
                       color='orange', alpha=0.7)
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[1, 1].bar(self.comparison_df['Model'], self.comparison_df['Precision'], 
                       color='salmon', alpha=0.7)
        axes[1, 1].set_title('Precision Comparison')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plot:
            plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Combined metrics plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.comparison_df))
        width = 0.2
        
        plt.bar(x - width, self.comparison_df['Test_Accuracy'], width, 
                label='Accuracy', alpha=0.7)
        plt.bar(x, self.comparison_df['F1_Score'], width, 
                label='F1 Score', alpha=0.7)
        plt.bar(x + width, self.comparison_df['Recall'], width, 
                label='Recall', alpha=0.7)
        plt.bar(x + 2*width, self.comparison_df['Precision'], width, 
                label='Precision', alpha=0.7)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, self.comparison_df['Model'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        if save_plot:
            plt.savefig("model_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n=20, save_plot=True):
        """Plot feature importance for the best model"""
        if self.best_model is None:
            print("No best model selected yet!")
            return
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        else:
            print(f"Feature importance not available for {self.best_model_name}")
            return
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], 
                 color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        
        if save_plot:
            plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def save_models_and_results(self, output_dir="models"):
        """Save trained models and results"""
        print(f"\nSaving models and results to {output_dir}/...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all models
        for model_name, model in self.models.items():
            model_filename = f"{output_dir}/{model_name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, model_filename)
            print(f"Saved {model_name} model to {model_filename}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_filename = f"{output_dir}/best_model.joblib"
            joblib.dump(self.best_model, best_model_filename)
            print(f"Saved best model ({self.best_model_name}) to {best_model_filename}")
        
        # Save results
        results_filename = f"{output_dir}/training_results.json"
        with open(results_filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, results in self.results.items():
                serializable_results[model_name] = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results.items()
                }
            json.dump(serializable_results, f, indent=2)
        print(f"Saved training results to {results_filename}")
        
        # Save comparison DataFrame
        if hasattr(self, 'comparison_df'):
            comparison_filename = f"{output_dir}/model_comparison.csv"
            self.comparison_df.to_csv(comparison_filename, index=False)
            print(f"Saved model comparison to {comparison_filename}")
        
        # Save feature importance
        if self.best_model is not None:
            feature_importance_df = self.plot_feature_importance(save_plot=False)
            if feature_importance_df is not None:
                feature_importance_filename = f"{output_dir}/feature_importance.csv"
                feature_importance_df.to_csv(feature_importance_filename, index=False)
                print(f"Saved feature importance to {feature_importance_filename}")
        
        # Save model summary
        model_summary = {
            'best_model': self.best_model_name,
            'best_f1_score': float(self.best_score),
            'total_models_trained': len(self.models),
            'feature_count': len(self.feature_names),
            'training_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0]
        }
        
        summary_filename = f"{output_dir}/model_summary.json"
        with open(summary_filename, 'w') as f:
            json.dump(model_summary, f, indent=2)
        print(f"Saved model summary to {summary_filename}")
        
        return {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'results': self.results,
            'comparison_df': getattr(self, 'comparison_df', None)
        }
    
    def run_full_training_pipeline(self, output_dir="models"):
        """Run the complete model training pipeline"""
        print("="*60)
        print("TELCO CHURN MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_processed_data()
        
        # Step 2: Train models
        self.train_random_forest()
        self.train_xgboost()
        self.train_catboost()
        
        # Step 3: Compare models
        self.compare_models()
        
        # Step 4: Create visualizations
        self.plot_model_comparison()
        self.plot_feature_importance()
        
        # Step 5: Save everything
        results = self.save_models_and_results(output_dir=output_dir)
        
        print("="*60)
        print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Best Model: {self.best_model_name} (F1: {self.best_score:.4f})")
        print("="*60)
        
        return results

def main():
    """Main function to run model training"""
    trainer = ChurnModelTrainer()
    results = trainer.run_full_training_pipeline()
    return results

if __name__ == "__main__":
    main() 