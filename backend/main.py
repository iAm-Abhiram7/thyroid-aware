from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from database import create_tables
from routes import auth, thyroid, symptoms, chat

# Create FastAPI app
app = FastAPI(
    title="ThyroidAware API",
    description="AI-powered thyroid health monitoring and analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(thyroid.router)
app.include_router(symptoms.router)
app.include_router(chat.router)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    create_tables()
    print("ThyroidAware API started successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ThyroidAware API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-06-12T23:41:00Z"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
