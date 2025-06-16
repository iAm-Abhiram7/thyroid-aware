from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    thyroid_tests = relationship("ThyroidTest", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    symptoms = relationship("UserSymptom", back_populates="user")

class ThyroidTest(Base):
    __tablename__ = "thyroid_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Thyroid test parameters
    tsh = Column(Float)
    t3 = Column(Float)
    t4 = Column(Float)
    fti = Column(Float)
    age = Column(Integer)
    
    # Additional parameters that might be extracted
    on_thyroxine = Column(Boolean, default=False)
    query_hypothyroid = Column(Boolean, default=False)
    query_hyperthyroid = Column(Boolean, default=False)
    pregnant = Column(Boolean, default=False)
    thyroid_surgery = Column(Boolean, default=False)
    
    # ML Model results
    prediction = Column(Integer)  # 0 = normal, 1 = thyroid condition
    confidence_score = Column(Float)
    model_used = Column(String, default="xgboost")
    
    # Metadata
    test_date = Column(DateTime(timezone=True))
    input_method = Column(String)  # 'manual', 'image_upload', 'csv'
    original_image_path = Column(String)  # if uploaded via image
    notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="thyroid_tests")

class UserSymptom(Base):
    __tablename__ = "user_symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Common thyroid symptoms
    fatigue = Column(Boolean, default=False)
    weight_changes = Column(String)  # 'gain', 'loss', 'none'
    mood_changes = Column(Boolean, default=False)
    sleep_issues = Column(Boolean, default=False)
    temperature_sensitivity = Column(String)  # 'cold', 'heat', 'none'
    heart_rate_changes = Column(Boolean, default=False)
    hair_changes = Column(Boolean, default=False)
    skin_changes = Column(Boolean, default=False)
    
    # Additional symptoms as JSON for flexibility
    other_symptoms = Column(JSON)
    
    severity_score = Column(Integer)  # 1-10 scale
    duration_weeks = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="symptoms")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    session_id = Column(String, index=True)  # For grouping related messages
    message_type = Column(String)  # 'user', 'bot'
    content = Column(Text, nullable=False)
    
    # Context information
    context_data = Column(JSON)  # Store relevant user data for this conversation
    intent = Column(String)  # 'lifestyle_advice', 'symptom_discussion', 'test_interpretation'
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")

class LifestyleRecommendation(Base):
    __tablename__ = "lifestyle_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    category = Column(String)  # 'diet', 'exercise', 'sleep', 'stress', 'medication'
    recommendation = Column(Text, nullable=False)
    priority = Column(String)  # 'high', 'medium', 'low'
    
    # Based on what data
    based_on_test_id = Column(Integer, ForeignKey("thyroid_tests.id"))
    based_on_symptoms = Column(Boolean, default=False)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    thyroid_test = relationship("ThyroidTest")
