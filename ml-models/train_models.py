#!/usr/bin/env python3
"""
ThyroidAware: ML Model Training Pipeline
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import pandas as pd
import numpy as np
from preprocessing.data_loader import ThyroidDataProcessor
from models.thyroid_models import ThyroidModelTrainer
from evaluation.model_evaluator import ModelEvaluator

def main():
    print("=" * 60)
    print("ThyroidAware: ML Model Training Pipeline")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = "../data/raw/thyroid0387.data"
    MODELS_DIR = "./trained_models"
    RESULTS_DIR = "./evaluation"
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n1. Data Preprocessing")
    print("-" * 30)
    
    processor = ThyroidDataProcessor()
    
    try:
        data = processor.process_complete_pipeline(
            DATA_PATH, 
            balance_method='smote',
            test_size=0.2
        )
    except FileNotFoundError:
        print(f"Dataset not found at {DATA_PATH}")
        print("Creating sample data for testing...")
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        sample_data = {
            'age': np.random.randint(18, 80, n_samples),
            'TSH': np.random.normal(2.0, 1.0, n_samples),
            'T3': np.random.normal(1.5, 0.5, n_samples),
            'TT4': np.random.normal(100, 20, n_samples),
            'FTI': np.random.normal(100, 15, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(sample_data)
        X = df.drop('target', axis=1)
        y = df['target']
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Step 2: Base Model Training
    print("\n2. Base Model Training")
    print("-" * 30)
    
    trainer = ThyroidModelTrainer()
    trainer.train_base_models(X_train, y_train, X_test, y_test)
    
    # Store base model accuracies before they get overwritten by tuned models
    base_model_accuracies = {name: result['accuracy'] for name, result in trainer.results.items()}
    
    # Step 3: Hyperparameter Tuning
    print("\n3. Hyperparameter Tuning")
    print("-" * 30)
    
    trainer.hyperparameter_tuning(X_train, y_train, cv=3)
    
    # Step 4: Evaluate Tuned Models
    print("\n4. Evaluating Tuned Models")
    print("-" * 30)
    
    tuned_results = trainer.evaluate_tuned_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Model Comparison
    print("\n5. Model Comparison")
    print("-" * 30)
    
    comparison_df = trainer.compare_models()
    
    # Step 6: Comprehensive Evaluation
    print("\n6. Comprehensive Evaluation")
    print("-" * 30)
    
    evaluator = ModelEvaluator()
    
    for model_name, result in trainer.results.items():
        if 'model' in result:
            evaluator.comprehensive_evaluation(
                result['model'], X_test, y_test, model_name
            )
    
    # Generate evaluation report
    eval_report = evaluator.generate_evaluation_report(
        f"{RESULTS_DIR}/evaluation_report.csv"
    )
    print("\nEvaluation Report:")
    print(eval_report.to_string(index=False))
    
    # Step 7: Visualizations
    print("\n7. Generating Visualizations")
    print("-" * 30)
    
    trainer.plot_results(f"{RESULTS_DIR}/model_comparison.png")
    evaluator.plot_evaluation_metrics(save_path=f"{RESULTS_DIR}/evaluation_metrics.png")
    
    # Step 8: Save Best Model
    print("\n8. Saving Best Model")
    print("-" * 30)
    
    best_model_name, best_model = trainer.save_best_model(f"{MODELS_DIR}/best_model.pkl")
    
    # Calculate improvement using stored base model accuracies
    if base_model_accuracies:
        base_accuracy = base_model_accuracies[best_model_name]
        tuned_accuracy = trainer.results[best_model_name]['accuracy']
        improvement = ((tuned_accuracy - base_accuracy) / base_accuracy) * 100
        
        print(f"\nTraining Complete!")
        print(f"Best Model: {best_model_name}")
        print(f"Base Model Accuracy: {base_accuracy:.4f}")
        print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Additional performance summary
        print(f"\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"🏆 Best Model: {best_model_name.upper()}")
        print(f"📊 Final Accuracy: {tuned_accuracy:.1%}")
        print(f"📈 Improvement: {improvement:.1f}%")
        print(f"🎯 AUC Score: {trainer.results[best_model_name].get('auc', 'N/A'):.1%}")
        print(f"✅ Target (20% improvement): {'ACHIEVED' if improvement >= 20 else 'PARTIAL'}")
    else:
        print(f"\nTraining Complete!")
        print(f"Best Model: {best_model_name}")
        print(f"Final Accuracy: {trainer.results[best_model_name]['accuracy']:.4f}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'trainer_results': trainer.results,
        'base_accuracies': base_model_accuracies
    }

if __name__ == "__main__":
    results = main()
