from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models.database import User, Conversation, ThyroidTest, UserSymptom
from schemas.thyroid_schemas import ConversationCreate, ConversationResponse
from utils.auth import get_current_active_user
import uuid
from datetime import datetime, timedelta

router = APIRouter(prefix="/chat", tags=["Chatbot"])

def get_user_context(user_id: int, db: Session) -> dict:
    """Get user context for chatbot conversations"""
    # Get recent thyroid tests
    recent_tests = db.query(ThyroidTest).filter(
        ThyroidTest.user_id == user_id
    ).order_by(ThyroidTest.created_at.desc()).limit(3).all()
    
    # Get recent symptoms
    recent_symptoms = db.query(UserSymptom).filter(
        UserSymptom.user_id == user_id
    ).order_by(UserSymptom.created_at.desc()).limit(2).all()
    
    context = {
        "recent_tests": [
            {
                "tsh": test.tsh,
                "t3": test.t3,
                "t4": test.t4,
                "prediction": test.prediction,
                "confidence": test.confidence_score,
                "date": test.created_at.isoformat()
            } for test in recent_tests
        ],
        "recent_symptoms": [
            {
                "fatigue": symptom.fatigue,
                "weight_changes": symptom.weight_changes,
                "mood_changes": symptom.mood_changes,
                "severity": symptom.severity_score,
                "date": symptom.created_at.isoformat()
            } for symptom in recent_symptoms
        ]
    }
    
    return context

def generate_bot_response(user_message: str, context: dict) -> str:
    """Generate chatbot response based on user message and context"""
    # Simple rule-based responses for now
    # TODO: Replace with actual AI model integration
    
    user_message_lower = user_message.lower()
    
    # Greeting responses
    if any(greeting in user_message_lower for greeting in ['hello', 'hi', 'hey']):
        return "Hello! I'm your ThyroidAware assistant. I can help you understand your thyroid test results and provide lifestyle recommendations. How can I assist you today?"
    
    # Test result inquiries
    if any(word in user_message_lower for word in ['test', 'result', 'tsh', 't3', 't4']):
        if context['recent_tests']:
            latest_test = context['recent_tests'][0]
            if latest_test['prediction'] == 1:
                return f"Based on your recent test results, there are some indicators that suggest thyroid dysfunction. Your TSH level was {latest_test['tsh']}, which should be discussed with your healthcare provider. Would you like some lifestyle recommendations?"
            else:
                return f"Your recent test results look normal! Your TSH level was {latest_test['tsh']}, which is within the healthy range. Keep maintaining your healthy lifestyle habits."
        else:
            return "I don't see any recent test results in your profile. Would you like to input your thyroid test values so I can help interpret them?"
    
    # Symptom inquiries
    if any(word in user_message_lower for word in ['symptom', 'tired', 'fatigue', 'weight']):
        return "I understand you're experiencing some symptoms. Thyroid conditions can cause fatigue, weight changes, mood swings, and temperature sensitivity. Have you had your thyroid levels checked recently? I'd recommend tracking your symptoms and discussing them with your healthcare provider."
    
    # Lifestyle advice
    if any(word in user_message_lower for word in ['diet', 'food', 'exercise', 'lifestyle']):
        return "Great question about lifestyle! For thyroid health, I recommend: 1) Eating iodine-rich foods like seafood and dairy, 2) Including selenium sources like Brazil nuts, 3) Regular moderate exercise, 4) Managing stress through meditation or yoga, 5) Getting adequate sleep (7-9 hours). Would you like more specific advice on any of these areas?"
    
    # Default response
    return "I'm here to help with your thyroid health questions. You can ask me about test results, symptoms, lifestyle recommendations, or general thyroid health information. What would you like to know more about?"

@router.post("/message", response_model=ConversationResponse)
async def send_message(
    message_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Send message to chatbot and get response"""
    try:
        # Generate session ID if not provided
        session_id = message_data.session_id or str(uuid.uuid4())
        
        # Get user context for personalized responses
        context = get_user_context(current_user.id, db)
        
        # Save user message
        user_message = Conversation(
            user_id=current_user.id,
            session_id=session_id,
            message_type="user",
            content=message_data.content,
            context_data=context,
            intent=message_data.intent
        )
        
        db.add(user_message)
        db.commit()
        
        # Generate bot response
        bot_response_text = generate_bot_response(message_data.content, context)
        
        # Save bot response
        bot_message = Conversation(
            user_id=current_user.id,
            session_id=session_id,
            message_type="bot",
            content=bot_response_text,
            context_data=context,
            intent="response"
        )
        
        db.add(bot_message)
        db.commit()
        db.refresh(bot_message)
        
        return bot_message
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

@router.get("/history/{session_id}", response_model=List[ConversationResponse])
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get chat history for a session"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.session_id == session_id
    ).order_by(Conversation.timestamp.asc()).all()
    
    return conversations

@router.get("/sessions", response_model=List[str])
async def get_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of user's chat sessions"""
    sessions = db.query(Conversation.session_id).filter(
        Conversation.user_id == current_user.id
    ).distinct().all()
    
    return [session[0] for session in sessions]
