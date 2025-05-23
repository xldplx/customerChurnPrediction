#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Telco Customer Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

class ChurnDataPreprocessor:
    def __init__(self, data_path="telco_customer_churn.csv"):
        """
        Initialize the preprocessor with data path
        """
        self.data_path = data_path
        self.encoder = None
        self.scaler = None
        self.target_column = "Churn"
        self.categorical_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
    def load_data(self):
        """Load and perform initial data inspection"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        
        # Display basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\nExploratory Data Analysis...")
        
        # Check unique values for each column
        for column in self.df.columns:
            unique_vals = np.unique(self.df[column].fillna('0'))
            nr_values = len(unique_vals)
            if nr_values <= 12:
                print(f'Feature {column}: {nr_values} unique values -- {unique_vals}')
            else:
                print(f'Feature {column}: {nr_values} unique values')
        
        # Basic statistics
        print("\nNumerical columns statistics:")
        print(self.df.describe())
        
        print("\nCategorical columns statistics:")
        print(self.df.describe(include=object))
    
    def clean_data(self):
        """Clean the dataset"""
        print("\nCleaning data...")
        
        # Remove unnecessary columns
        if 'customerID' in self.df.columns:
            self.df = self.df.drop(columns=['customerID'])
            print("Removed customerID column")
        
        # Fix data types
        self.df['tenure'] = self.df['tenure'].astype(float)
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].astype(float)
        
        # Fix TotalCharges (seems to be incorrectly set to tenure in original)
        if 'TotalCharges' in self.df.columns:
            # Handle TotalCharges as string with spaces
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            # Fill missing TotalCharges with tenure * MonthlyCharges
            mask = self.df['TotalCharges'].isna()
            self.df.loc[mask, 'TotalCharges'] = (
                self.df.loc[mask, 'tenure'] * self.df.loc[mask, 'MonthlyCharges']
            )
        
        return self.df
    
    def visualize_data(self, save_plots=True):
        """Create visualizations for data exploration"""
        print("\nCreating visualizations...")
        
        # Churn distribution
        plt.figure(figsize=(8, 6))
        sb.countplot(data=self.df, x=self.target_column)
        plt.title("Count Plot of Churn", fontsize=16)
        plt.xlabel("Churn", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        if save_plots:
            plt.savefig("churn_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Categorical variables distribution
        plt.figure(figsize=(20, 15))
        for i, col in enumerate(self.categorical_columns):
            if col in self.df.columns:
                plt.subplot(6, 3, i+1)
                if len(self.df[col].unique()) <= 5:
                    sb.countplot(data=self.df, x=col, alpha=0.7)
                else:
                    sb.countplot(data=self.df, y=col, 
                               order=self.df[col].value_counts().index, alpha=0.7)
                plt.title(f"Distribution of {col}", fontsize=12)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig("categorical_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Numerical variables boxplots
        for col in self.numerical_columns:
            if col in self.df.columns:
                plt.figure(figsize=(8, 5))
                sb.boxplot(data=self.df, y=col)
                plt.title(f"Boxplot of {col}")
                print(f'The median of {col} is: {self.df[col].median()}')
                if save_plots:
                    plt.savefig(f"{col}_boxplot.png", dpi=300, bbox_inches='tight')
                plt.show()
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data with test_size={test_size}...")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test set shape: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # Check class distribution
        print("\nClass distribution in train set:")
        print(self.y_train.value_counts(normalize=True))
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def encode_categorical_features(self):
        """Encode categorical features using OneHotEncoder"""
        print("\nEncoding categorical features...")
        
        # Initialize and fit encoder on training data
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        
        # Get categorical columns that exist in the data
        existing_cat_cols = [col for col in self.categorical_columns if col in self.X_train.columns]
        
        # Fit and transform training data
        cat_train_encoded = self.encoder.fit_transform(self.X_train[existing_cat_cols])
        cat_train_df = pd.DataFrame(
            cat_train_encoded,
            columns=self.encoder.get_feature_names_out(existing_cat_cols),
            index=self.X_train.index
        )
        
        # Remove original categorical columns and add encoded ones
        self.X_train_processed = self.X_train.drop(columns=existing_cat_cols)
        self.X_train_processed = pd.concat([self.X_train_processed, cat_train_df], axis=1)
        
        # Transform test data
        cat_test_encoded = self.encoder.transform(self.X_test[existing_cat_cols])
        cat_test_df = pd.DataFrame(
            cat_test_encoded,
            columns=self.encoder.get_feature_names_out(existing_cat_cols),
            index=self.X_test.index
        )
        
        self.X_test_processed = self.X_test.drop(columns=existing_cat_cols)
        self.X_test_processed = pd.concat([self.X_test_processed, cat_test_df], axis=1)
        
        print(f"After encoding - Train shape: {self.X_train_processed.shape}")
        print(f"After encoding - Test shape: {self.X_test_processed.shape}")
        
        return self.X_train_processed, self.X_test_processed
    
    def scale_features(self):
        """Scale numerical features using MinMaxScaler"""
        print("\nScaling features...")
        
        self.scaler = MinMaxScaler()
        
        # Fit and transform training data
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train_processed),
            columns=self.X_train_processed.columns,
            index=self.X_train_processed.index
        )
        
        # Transform test data
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test_processed),
            columns=self.X_test_processed.columns,
            index=self.X_test_processed.index
        )
        
        print(f"After scaling - Train shape: {self.X_train_scaled.shape}")
        print(f"After scaling - Test shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def apply_smote(self, sampling_strategy='auto', random_state=0, k_neighbors=5):
        """Apply SMOTE for handling class imbalance"""
        print("\nApplying SMOTE for class balancing...")
        
        print("Before SMOTE:")
        print(self.y_train.value_counts())
        
        # Convert target to binary numeric if needed
        y_train_numeric = self.y_train.map({'No': 0, 'Yes': 1}) if self.y_train.dtype == object else self.y_train
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors
        )
        
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X_train_scaled, y_train_numeric)
        
        print("After SMOTE:")
        print(pd.Series(self.y_resampled).value_counts())
        
        return self.X_resampled, self.y_resampled
    
    def save_processed_data(self, output_dir="processed_data"):
        """Save processed data and preprocessing artifacts"""
        print(f"\nSaving processed data to {output_dir}/...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed datasets
        np.save(f"{output_dir}/X_train_processed.npy", self.X_train_scaled.values)
        np.save(f"{output_dir}/X_test_processed.npy", self.X_test_scaled.values)
        np.save(f"{output_dir}/y_train_processed.npy", self.y_train.map({'No': 0, 'Yes': 1}).values)
        np.save(f"{output_dir}/y_test_processed.npy", self.y_test.map({'No': 0, 'Yes': 1}).values)
        
        # Save SMOTE resampled data
        np.save(f"{output_dir}/X_resampled.npy", self.X_resampled)
        np.save(f"{output_dir}/y_resampled.npy", self.y_resampled)
        
        # Save preprocessing artifacts
        joblib.dump(self.encoder, f"{output_dir}/encoder.joblib")
        joblib.dump(self.scaler, f"{output_dir}/scaler.joblib")
        
        # Save feature names
        feature_names = list(self.X_train_scaled.columns)
        joblib.dump(feature_names, f"{output_dir}/feature_names.joblib")
        
        # Save column information
        column_info = {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_names': feature_names
        }
        joblib.dump(column_info, f"{output_dir}/column_info.joblib")
        
        print("Data preprocessing completed and saved!")
        
        return {
            'X_resampled': self.X_resampled,
            'y_resampled': self.y_resampled,
            'X_test_scaled': self.X_test_scaled,
            'y_test_numeric': self.y_test.map({'No': 0, 'Yes': 1}),
            'encoder': self.encoder,
            'scaler': self.scaler,
            'feature_names': feature_names
        }
    
    def run_full_pipeline(self, save_plots=True, output_dir="processed_data"):
        """Run the complete preprocessing pipeline"""
        print("="*50)
        print("TELCO CHURN DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Clean data
        self.clean_data()
        
        # Step 4: Visualize data
        self.visualize_data(save_plots=save_plots)
        
        # Step 5: Split data
        self.split_data()
        
        # Step 6: Encode categorical features
        self.encode_categorical_features()
        
        # Step 7: Scale features
        self.scale_features()
        
        # Step 8: Apply SMOTE
        self.apply_smote()
        
        # Step 9: Save processed data
        results = self.save_processed_data(output_dir=output_dir)
        
        print("="*50)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return results

def main():
    """Main function to run preprocessing"""
    preprocessor = ChurnDataPreprocessor()
    results = preprocessor.run_full_pipeline()
    return results

if __name__ == "__main__":
    main() 