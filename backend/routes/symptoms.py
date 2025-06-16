from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models.database import User, UserSymptom
from schemas.thyroid_schemas import SymptomCreate, SymptomResponse
from utils.auth import get_current_active_user

router = APIRouter(prefix="/symptoms", tags=["Symptoms"])

@router.post("/", response_model=SymptomResponse)
async def create_symptom_record(
    symptom_data: SymptomCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Record user symptoms"""
    db_symptom = UserSymptom(
        user_id=current_user.id,
        **symptom_data.dict()
    )
    
    db.add(db_symptom)
    db.commit()
    db.refresh(db_symptom)
    
    return db_symptom

@router.get("/", response_model=List[SymptomResponse])
async def get_user_symptoms(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """Get user's symptom history"""
    symptoms = db.query(UserSymptom).filter(
        UserSymptom.user_id == current_user.id
    ).order_by(UserSymptom.created_at.desc()).limit(limit).all()
    
    return symptoms
