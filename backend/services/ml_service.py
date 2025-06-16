import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
from pathlib import Path

class ThyroidMLService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.model_name = ""
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model and preprocessor"""
        try:
            # Path to the trained model
            model_path = Path(__file__).parent.parent.parent / "ml-models" / "trained_models" / "best_model.pkl"
            preprocessor_path = Path(__file__).parent.parent.parent / "ml-models" / "trained_models" / "preprocessor.pkl"
            
            # Load the best model
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.model_name = model_data['model_name']
                print(f"Loaded {self.model_name} model successfully")
            else:
                print(f"Model file not found at {model_path}")
                return False
            
            # Load the preprocessor
            if preprocessor_path.exists():
                preprocessor_data = joblib.load(preprocessor_path)
                self.preprocessor = preprocessor_data['scaler']
                self.feature_names = preprocessor_data['feature_names']
                print(f"Loaded preprocessor with {len(self.feature_names)} features")
            else:
                print(f"Preprocessor file not found at {preprocessor_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        """Preprocess input data for model prediction"""
        try:
            # Create a DataFrame with the expected features
            # Map input data to model features
            feature_mapping = {
                'age': input_data.get('age', 30),
                'tsh': input_data.get('tsh', 2.0),
                't3': input_data.get('t3', 1.5),
                't4': input_data.get('t4', 100),
                'fti': input_data.get('fti', 100),
                'on_thyroxine': int(input_data.get('on_thyroxine', False)),
                'query_hypothyroid': int(input_data.get('query_hypothyroid', False)),
                'query_hyperthyroid': int(input_data.get('query_hyperthyroid', False)),
                'pregnant': int(input_data.get('pregnant', False)),
                'thyroid_surgery': int(input_data.get('thyroid_surgery', False))
            }
            
            # Create DataFrame with all possible features (fill missing with defaults)
            df = pd.DataFrame([feature_mapping])
            
            # If we have specific feature names from training, use only those
            if self.feature_names:
                # Ensure we have all required features
                for feature in self.feature_names:
                    if feature not in df.columns:
                        df[feature] = 0  # Default value for missing features
                
                # Select only the features used during training
                df = df[self.feature_names]
            
            # Scale the features
            if self.preprocessor:
                scaled_data = self.preprocessor.transform(df)
                return scaled_data
            else:
                return df.values
                
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            raise
    
    def predict(self, input_data: Dict) -> Tuple[int, float, str]:
        """Make prediction using the loaded model"""
        try:
            if not self.model:
                raise ValueError("Model not loaded")
            
            # Preprocess the input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.8  # Default confidence for models without probability
            
            # Determine risk level based on prediction and confidence
            if prediction == 0:
                risk_level = "low" if confidence > 0.8 else "medium"
            else:
                risk_level = "high" if confidence > 0.8 else "medium"
            
            return int(prediction), confidence, risk_level
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def interpret_results(self, prediction: int, confidence: float, input_data: Dict) -> Dict:
        """Interpret the prediction results and provide recommendations"""
        
        interpretation = {
            'prediction': prediction,
            'confidence': confidence,
            'diagnosis': 'Normal thyroid function' if prediction == 0 else 'Possible thyroid condition detected',
            'recommendations': [],
            'risk_factors': [],
            'next_steps': []
        }
        
        # Add specific recommendations based on prediction
        if prediction == 1:  # Thyroid condition detected
            interpretation['recommendations'].extend([
                'Consult with an endocrinologist for further evaluation',
                'Consider additional thyroid function tests',
                'Monitor symptoms closely',
                'Maintain a thyroid-healthy diet rich in iodine and selenium'
            ])
            
            # Check for specific risk factors
            if input_data.get('tsh', 0) > 4.0:
                interpretation['risk_factors'].append('Elevated TSH levels')
            if input_data.get('t3', 0) < 0.8:
                interpretation['risk_factors'].append('Low T3 levels')
            if input_data.get('t4', 0) < 60:
                interpretation['risk_factors'].append('Low T4 levels')
                
        else:  # Normal function
            interpretation['recommendations'].extend([
                'Continue regular health monitoring',
                'Maintain a balanced diet',
                'Regular exercise and stress management',
                'Annual thyroid function screening'
            ])
        
        # Add general next steps
        if confidence < 0.7:
            interpretation['next_steps'].append('Consider retesting for more accurate results')
        
        interpretation['next_steps'].extend([
            'Keep track of any symptoms',
            'Schedule regular follow-up appointments',
            'Maintain healthy lifestyle habits'
        ])
        
        return interpretation

# Create a global instance
ml_service = ThyroidMLService()
