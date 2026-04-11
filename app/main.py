from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import router as chat_router
from app.services.recommendation_service import recommendation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup — prevents memory spike on first request
    recommendation_service.initialize()
    yield


app = FastAPI(title="AI Music Intelligence Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mip-frontend-fsdzbwfzc3frahg6.centralus-01.azurewebsites.net",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "Music AI backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}