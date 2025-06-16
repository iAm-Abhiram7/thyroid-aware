import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def comprehensive_evaluation(self, model, X_test, y_test, model_name):
        """Perform comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
        else:
            fpr, tpr, roc_auc = None, None, None
            precision, recall, avg_precision = None, None, None
        
        self.evaluation_results[model_name] = {
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curve': (fpr, tpr, roc_auc),
            'pr_curve': (precision, recall, avg_precision),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.evaluation_results[model_name]
    
    def plot_evaluation_metrics(self, model_names=None, save_path=None):
        """Plot comprehensive evaluation metrics"""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC Curves
        ax = axes[0, 0]
        for name in model_names:
            if self.evaluation_results[name]['roc_curve'][0] is not None:
                fpr, tpr, roc_auc = self.evaluation_results[name]['roc_curve']
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True)
        
        # Precision-Recall Curves
        ax = axes[0, 1]
        for name in model_names:
            if self.evaluation_results[name]['pr_curve'][0] is not None:
                precision, recall, avg_precision = self.evaluation_results[name]['pr_curve']
                ax.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True)
        
        # Accuracy Comparison
        ax = axes[0, 2]
        accuracies = [self.evaluation_results[name]['classification_report']['accuracy'] 
                     for name in model_names]
        bars = ax.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # Confusion Matrices
        for i, name in enumerate(model_names[:3]):  # Show first 3 models
            ax = axes[1, i]
            cm = self.evaluation_results[name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, save_path=None):
        """Generate comprehensive evaluation report"""
        report_data = []
        
        for model_name, results in self.evaluation_results.items():
            report = results['classification_report']
            roc_auc = results['roc_curve'][2] if results['roc_curve'][2] is not None else 'N/A'
            avg_precision = results['pr_curve'][2] if results['pr_curve'][2] is not None else 'N/A'
            
            report_data.append({
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Precision': report['macro avg']['precision'],
                'Recall': report['macro avg']['recall'],
                'F1-Score': report['macro avg']['f1-score'],
                'ROC-AUC': roc_auc,
                'Average Precision': avg_precision
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Accuracy', ascending=False)
        
        if save_path:
            report_df.to_csv(save_path, index=False)
        
        return report_df
