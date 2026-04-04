# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.auth.router import router as auth_router

app = FastAPI(
    title="T-GIB API",
    description="Temporal Graph Information Bottleneck — Paradigm Shift Ranker",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "T-GIB API is running. Visit /docs for API reference."}
