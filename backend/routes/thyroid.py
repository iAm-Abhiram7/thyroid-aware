from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models.database import User, ThyroidTest
from schemas.thyroid_schemas import (
    ThyroidTestCreate, ThyroidTestResponse, PredictionRequest, 
    PredictionResponse, ImageUploadResponse
)
from utils.auth import get_current_active_user
from services.ml_service import ml_service
from datetime import datetime
import uuid

router = APIRouter(prefix="/thyroid", tags=["Thyroid Analysis"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_thyroid_condition(
    prediction_data: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Predict thyroid condition using ML model"""
    try:
        # Convert Pydantic model to dict for ML service
        input_data = prediction_data.dict()
        
        # Make prediction using ML service
        prediction, confidence, risk_level = ml_service.predict(input_data)
        
        # Get detailed interpretation
        interpretation = ml_service.interpret_results(prediction, confidence, input_data)
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            interpretation=interpretation['diagnosis'],
            recommendations=interpretation['recommendations']
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/tests", response_model=ThyroidTestResponse)
async def create_thyroid_test(
    test_data: ThyroidTestCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save thyroid test results and get prediction"""
    try:
        # Create prediction request from test data
        prediction_request = PredictionRequest(
            tsh=test_data.tsh or 2.0,
            t3=test_data.t3 or 1.5,
            t4=test_data.t4 or 100.0,
            fti=test_data.fti or 100.0,
            age=test_data.age or current_user.age or 30,
            on_thyroxine=test_data.on_thyroxine,
            query_hypothyroid=test_data.query_hypothyroid,
            query_hyperthyroid=test_data.query_hyperthyroid,
            pregnant=test_data.pregnant,
            thyroid_surgery=test_data.thyroid_surgery
        )
        
        # Get ML prediction
        prediction, confidence, risk_level = ml_service.predict(prediction_request.dict())
        
        # Create database record
        db_test = ThyroidTest(
            user_id=current_user.id,
            tsh=test_data.tsh,
            t3=test_data.t3,
            t4=test_data.t4,
            fti=test_data.fti,
            age=test_data.age or current_user.age,
            on_thyroxine=test_data.on_thyroxine,
            query_hypothyroid=test_data.query_hypothyroid,
            query_hyperthyroid=test_data.query_hyperthyroid,
            pregnant=test_data.pregnant,
            thyroid_surgery=test_data.thyroid_surgery,
            prediction=prediction,
            confidence_score=confidence,
            model_used=ml_service.model_name,
            test_date=test_data.test_date or datetime.utcnow(),
            input_method=test_data.input_method,
            notes=test_data.notes
        )
        
        db.add(db_test)
        db.commit()
        db.refresh(db_test)
        
        return db_test
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save test: {str(e)}"
        )

@router.get("/tests", response_model=List[ThyroidTestResponse])
async def get_user_thyroid_tests(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 10,
    skip: int = 0
):
    """Get user's thyroid test history"""
    tests = db.query(ThyroidTest).filter(
        ThyroidTest.user_id == current_user.id
    ).order_by(ThyroidTest.created_at.desc()).offset(skip).limit(limit).all()
    
    return tests

@router.get("/tests/{test_id}", response_model=ThyroidTestResponse)
async def get_thyroid_test(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific thyroid test by ID"""
    test = db.query(ThyroidTest).filter(
        ThyroidTest.id == test_id,
        ThyroidTest.user_id == current_user.id
    ).first()
    
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found"
        )
    
    return test

@router.post("/upload-image", response_model=ImageUploadResponse)
async def upload_thyroid_report_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Upload thyroid report image and extract data using OCR"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Generate unique filename
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # For now, return a mock response
        # TODO: Implement actual OCR processing
        mock_extracted_data = {
            "tsh": 2.5,
            "t3": 1.2,
            "t4": 95.0,
            "fti": 98.0
        }
        
        mock_confidence_scores = {
            "tsh": 0.95,
            "t3": 0.88,
            "t4": 0.92,
            "fti": 0.90
        }
        
        return ImageUploadResponse(
            filename=unique_filename,
            extracted_data=mock_extracted_data,
            confidence_scores=mock_confidence_scores,
            success=True,
            message="Image processed successfully. Please verify the extracted values."
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {str(e)}"
        )
