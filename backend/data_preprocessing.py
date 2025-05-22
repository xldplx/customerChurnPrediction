import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset(df, output_path):
    """Perform exploratory data analysis and create visualizations for important features only."""
    logger.info("Performing exploratory data analysis...")
    
    # Set styling for all visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'primary': '#4361ee',
        'secondary': '#3a0ca3',
        'accent': '#f72585',
        'positive': '#4cc9f0',
        'negative': '#f94144',
        'neutral': '#adb5bd'
    }
    
    # Create output directory for visualizations
    viz_path = f"{output_path}/visualizations"
    Path(viz_path).mkdir(parents=True, exist_ok=True)
    
    # 1. Data overview with skewness and kurtosis - saved as CSV only, no visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    info = pd.DataFrame({
        'dtypes': df.dtypes,
        'nunique': df.nunique(),
        'missing_values': df.isnull().sum(),
        'missing_percentage': df.isnull().sum() / len(df) * 100
    })
    
    # Add skewness and kurtosis for numeric columns
    skew_kurt = pd.DataFrame({
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurt()
    })
    
    # Merge information
    info = pd.concat([info, skew_kurt], axis=1)
    info.to_csv(f"{output_path}/data_overview.csv")
    logger.info(f"Data overview saved to {output_path}/data_overview.csv")
    
    # Ensure Churn is properly encoded for plotting
    churn_for_plot = df['Churn']
    if not pd.api.types.is_numeric_dtype(churn_for_plot):
        # If still a string, convert to binary
        churn_for_plot = churn_for_plot.map({'Yes': 1, 'No': 0})
    
    # 2. Churn distribution - ONE OF THE MOST IMPORTANT VISUALIZATIONS
    try:
        plt.figure(figsize=(10, 6))
        
        # Count the occurrences
        churn_counts = churn_for_plot.value_counts()
        total = len(churn_for_plot)
        
        # Create a custom bar plot
        ax = sns.barplot(x=churn_counts.index, y=churn_counts.values, 
                         palette=[colors['positive'], colors['negative']])
        
        # Add count and percentage labels
        for i, (count, pct) in enumerate(zip(churn_counts.values, churn_counts.values/total*100)):
            ax.text(i, count/2, f'{count}\n({pct:.1f}%)', 
                   ha='center', va='center', fontsize=12, color='white', 
                   fontweight='bold')
        
        # Customize the plot
        plt.title('Customer Churn Distribution', fontsize=16)
        plt.xlabel('Churn Status', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.xticks([0, 1], ['No Churn (0)', 'Churn (1)'])
        
        plt.tight_layout()
        plt.savefig(f"{viz_path}/churn_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Churn distribution visualization saved")
    except Exception as e:
        logger.warning(f"Error creating churn distribution plot: {e}")
    
    # 3. Correlation matrix for numerical features - IMPORTANT VISUALIZATION
    try:
        # Only include actual numeric columns in correlation
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_features) > 1:  # Need at least 2 features for correlation
            correlation = df[numerical_features].corr()
            
            # Create correlation matrix
            plt.figure(figsize=(12, 10))
            
            # Generate a mask for the upper triangle
            mask = np.triu(correlation)
            
            # Custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                       annot=True, fmt=".2f", square=True, linewidths=0.5)
            
            # Customize the plot
            plt.title('Correlation Matrix of Numerical Features', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f"{viz_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Correlation matrix saved")
            
            # Top correlated features with Churn - IMPORTANT VISUALIZATION
            if 'Churn' in correlation.columns:
                plt.figure(figsize=(10, 8))
                churn_corr = correlation['Churn'].drop('Churn').sort_values(ascending=False)
                
                # Plot top 10 correlations (or all if less than 10)
                top_n = min(10, len(churn_corr))
                top_corr = churn_corr.iloc[:top_n]
                
                # Create a horizontal bar chart
                ax = sns.barplot(x=top_corr.values, y=top_corr.index, 
                                palette=sns.color_palette("RdBu_r", n_colors=len(top_corr)))
                
                # Add values to bars
                for i, v in enumerate(top_corr.values):
                    ax.text(v + (0.01 if v >= 0 else -0.05), i, f'{v:.3f}', 
                           va='center', fontsize=10)
                
                # Add vertical line at x=0
                plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                
                # Customize the plot
                plt.title('Features Most Correlated with Churn', fontsize=16)
                plt.xlabel('Correlation Coefficient', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{viz_path}/churn_correlation.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Churn correlation visualization saved")
    except Exception as e:
        logger.warning(f"Error creating correlation matrix: {e}")
    
    # 4. Top 3 Important Categorical Features vs Churn
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'Churn']
    
    # Add binary features that might be categorical in nature
    binary_features = [col for col in df.columns if df[col].nunique() <= 2 and col != 'Churn']
    all_cat_features = list(set(categorical_features + binary_features))
    
    # Sort by association with churn
    feature_importance = {}
    for feature in all_cat_features:
        try:
            if feature in df.columns:
                # Calculate association using crosstab
                crosstab = pd.crosstab(df[feature], churn_for_plot)
                feature_importance[feature] = crosstab[1].sum() / crosstab.values.sum()
        except Exception as e:
            logger.warning(f"Error calculating importance for {feature}: {e}")
    
    # Sort features by importance and get top 3 only
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    important_categorical = [feature for feature, _ in sorted_features[:3]]
    
    # Plot important categorical features vs churn rate
    for feature in important_categorical:
        try:
            if feature in df.columns and df[feature].nunique() > 1:
                # Churn rate plot
                plt.figure(figsize=(12, 6))
                churn_by_category = pd.crosstab(df[feature], churn_for_plot, normalize='index') * 100
                
                # Create bar chart
                bars = plt.bar(churn_by_category.index, churn_by_category[1], 
                              color=sns.color_palette("YlOrRd", n_colors=len(churn_by_category)))
                
                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
                
                # Customize the plot
                plt.title(f'Churn Rate by {feature}', fontsize=16)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('Churn Rate (%)', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(f"{viz_path}/churn_by_{feature}.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Error creating plot for {feature}: {e}")
    
    logger.info(f"Categorical feature analysis saved")
    
    # 5. Top 3 Important Numerical Features vs Churn
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = [col for col in numerical_features if col != 'Churn']
    
    # Calculate correlation with churn to find most important numerical features
    corr_with_churn = {}
    for col in numerical_features:
        try:
            corr_with_churn[col] = abs(df[col].corr(churn_for_plot))
        except:
            corr_with_churn[col] = 0
    
    # Get top 3 important numerical features only
    sorted_numericals = sorted(corr_with_churn.items(), key=lambda x: x[1], reverse=True)
    important_numerical = [col for col, _ in sorted_numericals[:3]]
    
    # Create focused visualizations for important numerical features
    for feature in important_numerical:
        try:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                # Boxplot by churn - most informative plot
                plt.figure(figsize=(12, 6))
                plot_data = pd.DataFrame({
                    'value': df[feature].astype(float),
                    'churn': churn_for_plot
                }).dropna()
                
                sns.boxplot(x='churn', y='value', data=plot_data, 
                           palette=[colors['positive'], colors['negative']])
                
                # Customize the plot
                plt.title(f'{feature} by Churn Status', fontsize=16)
                plt.xlabel('Churn Status', fontsize=12)
                plt.ylabel(feature, fontsize=12)
                plt.xticks([0, 1], ['No Churn (0)', 'Churn (1)'])
                plt.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{viz_path}/{feature}_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Error creating plot for {feature}: {e}")
    
    logger.info(f"Numerical feature analysis saved")
    
    # Summarize dataset
    summary = {
        'total_records': len(df),
        'churn_rate': df['Churn'].value_counts(normalize=True).to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    # Only add correlation if it exists
    try:
        if 'correlation' in locals() and 'Churn' in correlation:
            summary['top_correlations'] = correlation['Churn'].sort_values(ascending=False).to_dict()
    except:
        pass
    
    return summary

def identify_and_drop_unnecessary_features(df):
    """Identify and drop unnecessary features based on analysis."""
    logger.info("Identifying unnecessary features...")
    
    # 1. Drop features with no variation (all values are the same)
    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_features:
        df = df.drop(columns=constant_features)
        logger.info(f"Dropped constant features: {constant_features}")
    
    # 2. Drop customerID - not relevant for prediction
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        logger.info("Dropped customerID column")
    
    # 3. Identify highly correlated features (optional)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_features) > 1:
        correlation = df[numerical_features].corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            logger.info(f"Highly correlated features that could be dropped: {to_drop}")
            # Not automatically dropping to allow manual review
    
    return df

def preprocess_data(input_path, output_path):
    """Preprocess Telco Churn data with robust error handling and EDA."""
    try:
        # Create output directories
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Save raw data overview
        df.describe().to_csv(f"{output_path}/raw_data_statistics.csv")
        df.head(10).to_csv(f"{output_path}/raw_data_sample.csv")
        logger.info(f"Raw data statistics saved to {output_path}")
        
        # Perform exploratory data analysis
        analysis_summary = analyze_dataset(df, output_path)
        with open(f"{output_path}/analysis_summary.txt", 'w') as f:
            for key, value in analysis_summary.items():
                f.write(f"{key}: {value}\n")
        
        # Clean TotalCharges (handle missing/coercion errors)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_count = df['TotalCharges'].isna().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in TotalCharges, filling with median")
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Identify and drop unnecessary features
        df = identify_and_drop_unnecessary_features(df)
        
        # Target encoding - ensure Churn is numeric
        if not pd.api.types.is_numeric_dtype(df['Churn']):
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            logger.info("Encoded Churn column to binary values (1=Yes, 0=No)")
        
        # Save clean data before splitting
        df.to_csv(f"{output_path}/clean_data.csv", index=False)
        logger.info(f"Clean data saved to {output_path}/clean_data.csv")
        
        # Define features
        y = df['Churn']
        X = df.drop(['Churn'], axis=1)
        
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
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
        
        logger.info(f"Data split: Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Save train/test split distributions
        try:
            train_distribution = pd.Series(y_train).value_counts(normalize=True)
            test_distribution = pd.Series(y_test).value_counts(normalize=True)
            
            plt.figure(figsize=(10, 6))
            pd.DataFrame({
                'Train': train_distribution,
                'Test': test_distribution
            }).plot(kind='bar')
            plt.title('Churn Distribution in Train and Test Sets')
            plt.ylabel('Proportion')
            plt.tight_layout()
            plt.savefig(f"{output_path}/visualizations/train_test_distribution.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Error creating train/test distribution plot: {e}")
        
        # Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        logger.info(f"Processed features shape: Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        # Handle imbalance (skip if SMOTE fails)
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            logger.info("SMOTE applied successfully.")
            logger.info(f"Resampled train set shape: {X_train_resampled.shape}")
            
            # Save class distribution after SMOTE
            try:
                resampled_distribution = pd.Series(y_train_resampled).value_counts(normalize=True)
                plt.figure(figsize=(10, 6))
                pd.DataFrame({
                    'Original': train_distribution,
                    'After SMOTE': resampled_distribution
                }).plot(kind='bar')
                plt.title('Class Distribution Before and After SMOTE')
                plt.ylabel('Proportion')
                plt.tight_layout()
                plt.savefig(f"{output_path}/visualizations/smote_effect.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating SMOTE effect plot: {e}")
            
            # Update the training data
            X_train_processed = X_train_resampled
            y_train = y_train_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
        
        # Save artifacts
        feature_names = []
        feature_names.extend(numerical_features)
        
        # Add one-hot encoded feature names
        if categorical_features:
            for col in categorical_features:
                unique_values = X[col].unique()
                if len(unique_values) > 1:
                    # Skip the first value due to drop='first' in OneHotEncoder
                    for val in unique_values[1:]:
                        feature_names.append(f"{col}_{val}")
        
        joblib.dump(preprocessor, f"{output_path}/preprocessor.joblib")
        joblib.dump(feature_names, f"{output_path}/feature_names.joblib")
        
        # Save the processed data as numpy arrays for consistency
        np.save(f"{output_path}/X_train.npy", X_train_processed)
        np.save(f"{output_path}/X_test.npy", X_test_processed)
        np.save(f"{output_path}/y_train.npy", y_train)
        np.save(f"{output_path}/y_test.npy", y_test)
            
        # Also save the original data splits for reference
        X_train.to_csv(f"{output_path}/X_train_original.csv", index=False)
        X_test.to_csv(f"{output_path}/X_test_original.csv", index=False)
        pd.Series(y_train, name='Churn').to_csv(f"{output_path}/y_train_original.csv", index=False)
        pd.Series(y_test, name='Churn').to_csv(f"{output_path}/y_test_original.csv", index=False)
        
        logger.info(f"Artifacts saved to {output_path}")
        
        return {
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features
        }

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data(
        input_path="dataset/telco_customer_churn.csv",
        output_path="dataset/processed"
    )
