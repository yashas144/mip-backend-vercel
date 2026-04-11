from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import router as chat_router

app = FastAPI(title="AI Music Intelligence Backend")

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