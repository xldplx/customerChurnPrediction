import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import logging
import os
import json
from pathlib import Path
import shap
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load processed data for model training."""
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Load numpy arrays directly
        X_train = np.load(f"{data_path}/X_train.npy", allow_pickle=True)
        X_test = np.load(f"{data_path}/X_test.npy", allow_pickle=True)
        y_train = np.load(f"{data_path}/y_train.npy", allow_pickle=True)
        y_test = np.load(f"{data_path}/y_test.npy", allow_pickle=True)
        
        # Load feature names
        feature_names = joblib.load(f"{data_path}/feature_names.joblib")
        
        # Load original data for reference (optional)
        try:
            X_train_original = pd.read_csv(f"{data_path}/X_train_original.csv")
            X_test_original = pd.read_csv(f"{data_path}/X_test_original.csv")
        except Exception as e:
            logger.warning(f"Could not load original data: {e}")
            X_train_original = None
            X_test_original = None
        
        logger.info(f"Data loaded successfully. X_train shape: {X_train.shape}")
        return X_train, y_train, X_test, y_test, X_train_original, X_test_original, feature_names
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def compare_models(X_train, y_train, X_test, y_test, feature_names, output_path):
    """Compare multiple machine learning models for churn prediction with detailed visualizations."""
    logger.info("Starting model comparison...")
    
    # Ensure output directory exists
    Path(f"{output_path}/model_comparison").mkdir(parents=True, exist_ok=True)
    
    # Define models to compare with lightweight configurations - only include XGBoost and Random Forest
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    # Results storage
    results = {
        'model': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'training_time': [],
        'inference_time': []
    }
    
    # Compare each model
    import time
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        try:
            # Train model with timing
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions with timing
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            inference_time = time.time() - start_time
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results['model'].append(name)
            results['accuracy'].append(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['auc'].append(auc)
            results['training_time'].append(training_time)
            results['inference_time'].append(inference_time)
            
            logger.info(f"{name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
            
            # Generate detailed report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(class_report).transpose()
            report_df.to_csv(f"{output_path}/model_comparison/classification_report_{name.replace(' ', '_').lower()}.csv")
            
            # Confusion Matrix with nice visualization
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            
            # Calculate metrics for confusion matrix annotation
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Plot confusion matrix with percentages and counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {name}\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')
            
            # Add text annotations for metrics
            plt.figtext(0.15, 0.10, f'Specificity: {specificity:.3f}', fontsize=9)
            plt.figtext(0.15, 0.05, f'F1-Score: {f1:.3f}', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/model_comparison/cm_{name.replace(' ', '_').lower()}.png")
            plt.close()
            
            # Feature importance (if supported)
            try:
                if hasattr(model, 'feature_importances_'):
                    # Create feature importance dataframe
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Save feature importance data
                    importance_df.to_csv(f"{output_path}/model_comparison/feature_importance_{name.replace(' ', '_').lower()}.csv", index=False)
                    
                    # Plot feature importance - only top 10 features
                    plt.figure(figsize=(10, 8))
                    top_n = min(10, len(importance_df))  # Top 10 features or all if fewer
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
                    plt.title(f'Top {top_n} Feature Importances - {name}')
                    plt.tight_layout()
                    plt.savefig(f"{output_path}/model_comparison/feature_importance_{name.replace(' ', '_').lower()}.png")
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not generate feature importance for {name}: {e}")
            
            # Save model for reference
            joblib.dump(model, f"{output_path}/model_comparison/{name.replace(' ', '_').lower()}_base.joblib")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    results_df.to_csv(f"{output_path}/model_comparison/model_comparison.csv", index=False)
    
    # Create a comprehensive model comparison visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Combine all metrics in one plot
    plt.figure(figsize=(12, 8))
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create a DataFrame with model as index and metrics as columns for plotting
    metrics_df = results_df.set_index('model')[metrics]
    
    # Plot as bar chart with grouped bars
    ax = metrics_df.plot(kind='bar', figsize=(12, 8), width=0.8)
    
    # Customize the plot
    plt.title('Model Performance Comparison (XGBoost vs Random Forest)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/model_comparison/xgboost_vs_randomforest.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV for easy reference
    metrics_df.to_csv(f"{output_path}/model_comparison/xgboost_vs_randomforest.csv")
    
    # Find best model based on AUC (or fallback to other metrics if AUC fails)
    if results['auc']:
        best_model_idx = results['auc'].index(max(results['auc']))
        best_model_name = results['model'][best_model_idx]
        logger.info(f"Best model based on AUC: {best_model_name}")
    elif results['model']:
        best_model_name = results['model'][0]  # Default to first model if AUC fails
        logger.warning(f"Couldn't determine best model based on AUC, using {best_model_name}")
    else:
        logger.error("No models trained successfully!")
        return None, None
    
    return models, best_model_name

def train_optimized_model(X_train, y_train, X_test, y_test, feature_names, best_model_name, output_path):
    """Train and optimize the best model with hyperparameter tuning."""
    logger.info(f"Optimizing {best_model_name} model...")
    
    # Smaller parameter grids for quicker training
    param_grids = {
        "Random Forest": {
            'n_estimators': [100],
            'max_depth': [None, 10]
        },
        "XGBoost": {
            'n_estimators': [100],
            'max_depth': [3, 5]
        }
    }
    
    # Select base model and parameter grid
    if best_model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif best_model_name == "XGBoost":
        model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    else:
        logger.error(f"Unknown model name: {best_model_name}")
        return None
    
    param_grid = param_grids[best_model_name]
    
    try:
        # Grid search with cross-validation
        logger.info(f"Starting grid search for {best_model_name}...")
        grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Final evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        logger.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"Test Recall: {recall_score(y_test, y_pred):.4f}")
        
        # Detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        with open(f"{output_path}/best_model_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Optimized {best_model_name}')
        plt.tight_layout()
        plt.savefig(f"{output_path}/best_model_confusion_matrix.png")
        plt.close()
        
        # Feature importance
        plot_feature_importance(best_model, feature_names, best_model_name, output_path)
        
        # Save model
        joblib.dump(best_model, f"{output_path}/model.joblib")
        
        # Save best parameters
        with open(f"{output_path}/best_params.json", 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4)
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        # Fall back to the base model if optimization fails
        try:
            if best_model_name in models:
                base_model = models[best_model_name]
                logger.warning(f"Using base model without optimization")
                joblib.dump(base_model, f"{output_path}/model.joblib")
                return base_model
        except:
            logger.error(f"Couldn't fall back to base model")
            return None

def plot_feature_importance(model, feature_names, model_name, output_path):
    """Plot feature importance for the trained model."""
    logger.info("Generating feature importance plot...")
    
    # Set style for consistent visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        if model_name in ["Random Forest", "XGBoost"]:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Get top 20 features or all if less than 20
                top_n = min(20, len(indices))
                top_indices = indices[:top_n]
                
                # Create feature importance DataFrame for better handling
                importance_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices],
                    'Importance': importances[indices]
                })
                
                # Save importance scores
                importance_df.to_csv(f"{output_path}/feature_importance.csv", index=False)
                
                # Plot feature importance with a horizontal bar chart for better readability
                plt.figure(figsize=(14, 10))
                
                # Create color gradient based on importance
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, top_n))
                
                # Plot horizontal bars
                ax = plt.barh(
                    range(top_n),
                    importances[top_indices],
                    color=colors,
                    align='center',
                    alpha=0.8
                )
                
                # Add importance values at the end of each bar
                for i, v in enumerate(importances[top_indices]):
                    plt.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=12)
                
                # Customize the plot
                plt.yticks(range(top_n), [feature_names[i] for i in top_indices], fontsize=12)
                plt.xlabel('Importance', fontsize=14, labelpad=10)
                plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=18, pad=20)
                plt.grid(axis='x', alpha=0.3)
                
                # Add model description
                model_description = ""
                if model_name == "Random Forest":
                    model_description = (
                        "Random Forest importance is based on how much each feature\n"
                        "decreases impurity (Gini or entropy) across all trees."
                    )
                elif model_name == "XGBoost":
                    model_description = (
                        "XGBoost importance is based on how much each feature\n"
                        "contributes to improving the model's predictions."
                    )
                
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                plt.annotate(model_description, xy=(0.5, 0.01), xycoords='figure fraction', 
                            fontsize=12, ha='center', va='bottom', bbox=props)
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        elif model_name == "Logistic Regression":
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                indices = np.argsort(np.abs(coefficients))[::-1]
                
                # Get top 20 features or all if less than 20
                top_n = min(20, len(indices))
                top_indices = indices[:top_n]
                
                # Create coefficient DataFrame
                coef_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices],
                    'Coefficient': coefficients[indices],
                    'Abs_Coefficient': np.abs(coefficients[indices])
                })
                
                # Save coefficients
                coef_df.to_csv(f"{output_path}/feature_coefficients.csv", index=False)
                
                # Plot coefficients with positive/negative color coding
                plt.figure(figsize=(14, 10))
                
                # Create color map based on coefficient sign
                colors = ['red' if c < 0 else 'blue' for c in coefficients[top_indices]]
                
                # Plot horizontal bars
                bars = plt.barh(
                    range(top_n),
                    coefficients[top_indices],
                    color=colors,
                    align='center',
                    alpha=0.7
                )
                
                # Add vertical line at x=0
                plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                
                # Add coefficient values
                for i, v in enumerate(coefficients[top_indices]):
                    text_color = 'black'
                    if abs(v) > 0.1:
                        text_color = 'white' if v < 0 else 'black'
                    plt.text(v + (0.01 if v >= 0 else -0.05), i, f'{v:.4f}', va='center', fontsize=12, color=text_color)
                
                # Customize the plot
                plt.yticks(range(top_n), [feature_names[i] for i in top_indices], fontsize=12)
                plt.xlabel('Coefficient Value', fontsize=14, labelpad=10)
                plt.title(f'Top {top_n} Feature Coefficients - Logistic Regression', fontsize=18, pad=20)
                plt.grid(axis='x', alpha=0.3)
                
                # Add coefficient explanation
                explanation = (
                    "Coefficient Interpretation:\n"
                    "• Positive (blue): Higher values increase churn probability\n"
                    "• Negative (red): Higher values decrease churn probability\n"
                    "• Magnitude: Larger absolute values have stronger effects"
                )
                
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                plt.annotate(explanation, xy=(0.5, 0.01), xycoords='figure fraction', 
                            fontsize=12, ha='center', va='bottom', bbox=props)
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/feature_coefficients.png", dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        logger.error(f"Error generating feature importance: {e}")

def generate_shap_explanations(model, X_test, feature_names, model_name, output_path, max_display=10):
    """Generate enhanced SHAP explanations for model predictions."""
    logger.info("Generating SHAP explanations...")
    
    # Create directory for SHAP plots
    Path(f"{output_path}/shap").mkdir(parents=True, exist_ok=True)
    
    # Sample a small subset of test data for SHAP analysis (for efficiency)
    n_samples = min(100, X_test.shape[0])
    X_sample = X_test[:n_samples]
    
    try:
        # Only generate summary plots which are most useful
        if model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Set style for consistent visualization
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 1. Bar plot - most important for insights
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                             plot_type="bar", max_display=max_display, show=False,
                             color=plt.cm.viridis(0.6))
            
            # Customize the plot
            plt.gcf().axes[-1].set_aspect(100)  # Adjust colorbar aspect
            plt.gcf().axes[-1].set_box_aspect(100)  # More adjustment for colorbar
            
            plt.title(f'SHAP Feature Importance ({model_name})', fontsize=18, pad=20)
            plt.xlabel('mean(|SHAP value|)', fontsize=14, labelpad=10)
            plt.ylabel('')  # Features are already labeled
            plt.tight_layout()
            plt.savefig(f"{output_path}/shap/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Beeswarm plot - shows feature impact direction
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                             max_display=max_display, show=False)
            
            # Customize the plot
            plt.gcf().axes[-1].set_aspect(100)  # Adjust colorbar aspect
            plt.gcf().axes[-1].set_box_aspect(100)  # More adjustment for colorbar
            
            plt.title(f'SHAP Summary Plot ({model_name})', fontsize=18, pad=20)
            plt.xlabel('SHAP value (impact on model output)', fontsize=14, labelpad=10)
            
            # Add an explanation text box
            textstr = '\n'.join([
                'Interpretation:',
                '• Features are ordered by importance (top = most important)',
                '• Red points = high feature value, Blue points = low feature value',
                '• Points to right = increase churn prediction',
                '• Points to left = decrease churn prediction'
            ])
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            plt.annotate(textstr, xy=(0.5, 0.01), xycoords='figure fraction', 
                        fontsize=12, ha='center', va='bottom', bbox=props)
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/shap/summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP explanations generated successfully")
    except Exception as e:
        logger.warning(f"Could not generate SHAP explanations: {e}")

def train_model(data_path, output_path):
    """Train and evaluate machine learning models for churn prediction."""
    try:
        # Load preprocessed data
        X_train, y_train, X_test, y_test, X_train_original, X_test_original, feature_names = load_data(data_path)
        
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save feature names for later use
        joblib.dump(feature_names, f"{output_path}/feature_names.joblib")
        
        # Compare models
        models, best_model_name = compare_models(X_train, y_train, X_test, y_test, feature_names, output_path)
        
        if best_model_name:
            # Train optimized version of best model
            optimized_model = train_optimized_model(X_train, y_train, X_test, y_test, feature_names, best_model_name, output_path)
            
            # Save model
            if optimized_model:
                joblib.dump(optimized_model, f"{output_path}/model.joblib")
                logger.info(f"Final model saved to {output_path}/model.joblib")
                
                # Plot feature importance
                plot_feature_importance(optimized_model, feature_names, best_model_name, output_path)
                
                # Generate SHAP explanations if possible
                try:
                    generate_shap_explanations(optimized_model, X_test, feature_names, best_model_name, output_path)
                except Exception as e:
                    logger.warning(f"Could not generate SHAP explanations: {e}")
                
                # Create model summary
                model_summary = {
                    "model_type": best_model_name,
                    "feature_count": len(feature_names),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names
                }
                
                # Add metrics to summary
                y_pred = optimized_model.predict(X_test)
                y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]
                
                model_summary["metrics"] = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred)),
                    "recall": float(recall_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred)),
                    "auc": float(roc_auc_score(y_test, y_pred_proba))
                }
                
                # Save model summary
                with open(f"{output_path}/model_summary.json", 'w') as f:
                    json.dump(model_summary, f, indent=4)
                
                logger.info(f"Model summary saved to {output_path}/model_summary.json")
                
                return optimized_model
            else:
                logger.error("Failed to train optimized model.")
                
                # Fallback: Use the best base model
                best_model = models.get(best_model_name)
                if best_model:
                    joblib.dump(best_model, f"{output_path}/model.joblib")
                    logger.info(f"Fallback to base model. Saved to {output_path}/model.joblib")
                    return best_model
        else:
            logger.error("No suitable model found.")
            return None
            
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    train_model(
        data_path="dataset/processed",
        output_path="model"
    )
