from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.on_event("startup")
def startup_event():
    recommendation_service.initialize()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return recommendation_service.recommend(req.message, top_k=5)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Artifacts missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))