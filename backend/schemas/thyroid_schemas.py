from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Thyroid Test Schemas
class ThyroidTestBase(BaseModel):
    tsh: Optional[float] = None
    t3: Optional[float] = None
    t4: Optional[float] = None
    fti: Optional[float] = None
    age: Optional[int] = None
    on_thyroxine: bool = False
    query_hypothyroid: bool = False
    query_hyperthyroid: bool = False
    pregnant: bool = False
    thyroid_surgery: bool = False
    test_date: Optional[datetime] = None
    notes: Optional[str] = None

class ThyroidTestCreate(ThyroidTestBase):
    input_method: str = "manual"

class ThyroidTestResponse(ThyroidTestBase):
    id: int
    user_id: int
    prediction: Optional[int] = None
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# ML Prediction Schemas
class PredictionRequest(BaseModel):
    tsh: float
    t3: float
    t4: float
    fti: float
    age: int
    on_thyroxine: bool = False
    query_hypothyroid: bool = False
    query_hyperthyroid: bool = False
    pregnant: bool = False
    thyroid_surgery: bool = False

class PredictionResponse(BaseModel):
    prediction: int  # 0 = normal, 1 = thyroid condition
    confidence: float
    risk_level: str  # 'low', 'medium', 'high'
    interpretation: str
    recommendations: List[str]

# Symptom Schemas
class SymptomBase(BaseModel):
    fatigue: bool = False
    weight_changes: Optional[str] = None
    mood_changes: bool = False
    sleep_issues: bool = False
    temperature_sensitivity: Optional[str] = None
    heart_rate_changes: bool = False
    hair_changes: bool = False
    skin_changes: bool = False
    other_symptoms: Optional[Dict[str, Any]] = None
    severity_score: Optional[int] = None
    duration_weeks: Optional[int] = None

class SymptomCreate(SymptomBase):
    pass

class SymptomResponse(SymptomBase):
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Conversation Schemas
class ConversationCreate(BaseModel):
    content: str
    session_id: Optional[str] = None
    intent: Optional[str] = None

class ConversationResponse(BaseModel):
    id: int
    user_id: int
    session_id: Optional[str]
    message_type: str
    content: str
    context_data: Optional[Dict[str, Any]]
    intent: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True

# Image Upload Schema
class ImageUploadResponse(BaseModel):
    filename: str
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    success: bool
    message: str

# Authentication Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Dashboard Schemas
class DashboardData(BaseModel):
    recent_tests: List[ThyroidTestResponse]
    symptom_trends: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    conversation_summary: Dict[str, Any]
