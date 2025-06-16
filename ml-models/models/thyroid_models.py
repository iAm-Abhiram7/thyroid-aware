import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import matplotlib.pyplot as plt

class ThyroidModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize base models"""
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'knn': KNeighborsClassifier()
        }
    
    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'knn': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
        return param_grids
    
    def _prepare_data_for_model(self, X, model_name):
        """Prepare data for specific models, especially KNN"""
        if model_name == 'knn':
            # Ensure C-contiguous memory layout for KNN
            if isinstance(X, pd.DataFrame):
                return np.ascontiguousarray(X.values)
            else:
                return np.ascontiguousarray(X)
        else:
            # Other models can handle DataFrames directly
            return X
    
    def train_base_models(self, X_train, y_train, X_test, y_test):
        """Train base models without hyperparameter tuning"""
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Prepare data for the specific model
            X_train_prepared = self._prepare_data_for_model(X_train, name)
            X_test_prepared = self._prepare_data_for_model(X_test, name)
            
            model.fit(X_train_prepared, y_train)
            
            y_pred = model.predict(X_test_prepared)
            y_pred_proba = model.predict_proba(X_test_prepared)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Fixed print statement - handle None AUC properly
            if auc is not None:
                print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            else:
                print(f"{name} - Accuracy: {accuracy:.4f}, AUC: N/A")

            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
    
    def hyperparameter_tuning(self, X_train, y_train, cv=3):
        """Perform hyperparameter tuning using GridSearchCV"""
        self.initialize_models()
        param_grids = self.get_hyperparameter_grids()
        
        for name, model in self.models.items():
            print(f"Tuning hyperparameters for {name}...")
            
            # Prepare data for the specific model
            X_train_prepared = self._prepare_data_for_model(X_train, name)
            
            grid_search = GridSearchCV(
                model, 
                param_grids[name], 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_prepared, y_train)
            self.best_models[name] = grid_search.best_estimator_
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def evaluate_tuned_models(self, X_train, y_train, X_test, y_test):
        """Evaluate hyperparameter-tuned models"""
        tuned_results = {}
        
        for name, model in self.best_models.items():
            print(f"Evaluating tuned {name}...")
            
            # Prepare data for the specific model
            X_train_prepared = self._prepare_data_for_model(X_train, name)
            X_test_prepared = self._prepare_data_for_model(X_test, name)
            
            cv_scores = cross_val_score(model, X_train_prepared, y_train, cv=3, scoring='accuracy')
            
            y_pred = model.predict(X_test_prepared)
            y_pred_proba = model.predict_proba(X_test_prepared)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            tuned_results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"{name} - CV: {cv_scores.mean():.4f}(±{cv_scores.std():.4f}), Test Accuracy: {accuracy:.4f}")
        
        self.results.update(tuned_results)
        return tuned_results
    
    def compare_models(self):
        """Compare all trained models"""
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'AUC': result.get('auc', 'N/A'),
                'CV_Mean': result.get('cv_mean', 'N/A'),
                'CV_Std': result.get('cv_std', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_best_model(self, filepath):
        """Save the best performing model"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'accuracy': self.results[best_model_name]['accuracy']
        }
        
        joblib.dump(model_data, filepath)
        print(f"Best model ({best_model_name}) saved to {filepath}")
        
        return best_model_name, best_model

    def plot_results(self, save_path=None):
        """Plot model comparison results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better plots
        plt.style.use('default')
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        aucs = [self.results[model].get('auc', 0) for model in models if self.results[model].get('auc')]
        
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Accuracy comparison
        plt.subplot(1, 3, 1)
        bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0.8, 1.0)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: AUC comparison
        plt.subplot(1, 3, 2)
        if aucs and len(aucs) == len(models):
            bars = plt.bar(models, aucs, color=['#9b59b6', '#f39c12', '#1abc9c'])
            plt.title('Model AUC Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('AUC Score', fontsize=12)
            plt.ylim(0.9, 1.0)
            plt.xticks(rotation=45)
            
            for bar, auc in zip(bars, aucs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 3: Combined metrics
        plt.subplot(1, 3, 3)
        x = range(len(models))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', 
                color='#3498db', alpha=0.8)
        if aucs and len(aucs) == len(models):
            plt.bar([i + width/2 for i in x], aucs, width, label='AUC', 
                    color='#e74c3c', alpha=0.8)
        
        plt.title('Accuracy vs AUC', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0.8, 1.0)
        plt.xticks(x, models, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return plt
