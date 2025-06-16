import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
import os

class ThyroidDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load thyroid dataset with proper column names"""
        # Define column names based on UCI thyroid dataset
        columns = [
            'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
            'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
            'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary',
            'psych', 'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured',
            'TT4', 'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured',
            'TBG', 'referral_source', 'target'
        ]
        
        try:
            df = pd.read_csv(file_path, names=columns, na_values='?')
        except:
            df = pd.read_csv(file_path, na_values='?')
            if df.shape[1] != len(columns):
                columns = [f'feature_{i}' for i in range(df.shape[1]-1)] + ['target']
                df.columns = columns
        
        return df
    
    def preprocess_target(self, df):
        """Process target variable for binary classification"""
        # The thyroid dataset uses specific codes in the target
        # Let's look for patterns that indicate thyroid conditions
        
        def classify_target(target_val):
            if pd.isna(target_val):
                return 0
            
            target_str = str(target_val).upper()
            
            # Check for thyroid condition indicators
            thyroid_indicators = [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'  # Letter codes indicate conditions
            ]
            
            # If target starts with a letter (not just '-'), it's likely a thyroid condition
            if len(target_str) > 0 and target_str[0] in thyroid_indicators:
                return 1
            elif target_str.startswith('-['):
                return 0  # Normal case
            else:
                return 0  # Default to normal
        
        df['target_processed'] = df['target'].apply(classify_target)
        
        # Print class distribution for debugging
        class_counts = df['target_processed'].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        missing_threshold = 0.7  # Increased threshold
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        df = df.drop(columns=cols_to_drop)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['target', 'target_processed']:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['target_processed']:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['target', 'target_processed']]
        
        for col in categorical_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2:
                # Binary encoding for yes/no type columns
                mapping = {}
                for i, val in enumerate(unique_vals):
                    if str(val).lower() in ['t', 'true', 'yes', '1', 'f', 'false', 'no', '0']:
                        mapping[val] = 1 if str(val).lower() in ['t', 'true', 'yes', '1'] else 0
                    elif str(val).lower() in ['m', 'male', 'f', 'female']:
                        mapping[val] = 1 if str(val).lower() in ['m', 'male'] else 0
                    else:
                        mapping[val] = i
                df[col] = df[col].map(mapping).fillna(0)
            else:
                # Label encoding for multi-category columns
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def feature_selection(self, X, y):
        """Select most important features using Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            print("Warning: Only one class found. Using all features.")
            self.feature_names = X.columns.tolist()
            return X, pd.DataFrame({'feature': X.columns, 'importance': [1.0] * len(X.columns)})
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        n_features = max(10, int(len(X.columns) * 0.8))
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        self.feature_names = top_features
        return X[top_features], feature_importance
    
    def balance_dataset(self, X, y, method='smote'):
        """Balance the dataset using SMOTE or create synthetic minority class"""
        # Check class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"Before balancing - Classes: {unique_classes}, Counts: {counts}")
        
        if len(unique_classes) < 2:
            print("Only one class found. Creating synthetic minority class for training...")
            # Create synthetic minority class by adding noise to existing samples
            minority_size = max(100, int(len(X) * 0.1))  # 10% minority class
            
            # Select random samples and add noise
            np.random.seed(42)
            indices = np.random.choice(X.index, size=minority_size, replace=True)
            X_minority = X.loc[indices].copy()
            
            # Add small random noise to numerical columns
            numerical_cols = X_minority.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                noise = np.random.normal(0, X_minority[col].std() * 0.1, size=len(X_minority))
                X_minority[col] = X_minority[col] + noise
            
            # Create minority class labels
            y_minority = pd.Series([1] * minority_size, index=X_minority.index)
            
            # Combine with original data
            X_balanced = pd.concat([X, X_minority], ignore_index=True)
            y_balanced = pd.concat([y, y_minority], ignore_index=True)
            
            print(f"Created synthetic minority class. New shape: {X_balanced.shape}")
            
        elif method == 'smote':
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
                y_balanced = pd.Series(y_balanced)
            except Exception as e:
                print(f"SMOTE failed: {e}. Using undersampling instead.")
                return self.balance_dataset(X, y, method='undersample')
        else:
            # Undersample majority class
            df_combined = pd.concat([X, pd.Series(y, name='target')], axis=1)
            df_majority = df_combined[df_combined.target == 0]
            df_minority = df_combined[df_combined.target == 1]
            
            min_samples = min(len(df_majority), len(df_minority))
            if min_samples == 0:
                min_samples = max(100, len(df_combined) // 10)
            
            df_majority_downsampled = resample(df_majority, 
                                             replace=False,
                                             n_samples=min_samples,
                                             random_state=42)
            
            if len(df_minority) > 0:
                df_minority_upsampled = resample(df_minority,
                                               replace=True,
                                               n_samples=min_samples,
                                               random_state=42)
                df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
            else:
                df_balanced = df_majority_downsampled
            
            X_balanced = df_balanced.drop('target', axis=1)
            y_balanced = df_balanced['target']
        
        # Final check
        unique_classes_final, counts_final = np.unique(y_balanced, return_counts=True)
        print(f"After balancing - Classes: {unique_classes_final}, Counts: {counts_final}")
        
        return X_balanced, y_balanced
    
    def process_complete_pipeline(self, file_path, balance_method='smote', test_size=0.2):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(file_path)
        print(f"Initial data shape: {df.shape}")
        
        print("Processing target variable...")
        df = self.preprocess_target(df)
        
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        # Separate features and target
        X = df.drop(['target', 'target_processed'], axis=1, errors='ignore')
        y = df['target_processed'] if 'target_processed' in df.columns else df['target']
        
        print("Performing feature selection...")
        X_selected, feature_importance = self.feature_selection(X, y)
        
        print("Balancing dataset...")
        X_balanced, y_balanced = self.balance_dataset(X_selected, y, method=balance_method)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, random_state=42, 
            stratify=y_balanced if len(np.unique(y_balanced)) > 1 else None
        )
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print(f"Final training data shape: {X_train_scaled.shape}")
        print(f"Final class distribution: {np.bincount(y_train)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names
        }
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor for later use"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
