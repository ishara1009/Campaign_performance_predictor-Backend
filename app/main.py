"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict, ocr, history
from app.services.predictor import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model artifacts on startup so first request is fast
    logger.info("Application startup — loading model artifacts …")
    predictor.load()
    logger.info("Startup complete.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Campaign Performance Predictor API",
    description="Predicts social media engagement using a Transformer model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict.router)
app.include_router(ocr.router)
app.include_router(history.router)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor._loaded}
