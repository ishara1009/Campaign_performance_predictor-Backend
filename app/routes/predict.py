"""
/api/predict — run Transformer model and return predictions + explainability.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.predictor import predictor
from app.services.explainability import generate_explainability, generate_groq_insights
from app.services.database import save_prediction

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    try:
        predictions = predictor.predict(
            caption=req.caption,
            content=req.content,
            platform=req.platform,
            post_date=req.post_date,
            post_time=req.post_time,
            followers=req.followers,
            ad_boost=req.ad_boost,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Explainability via Groq
    tips = generate_explainability(
        predictions=predictions,
        caption=req.caption,
        content=req.content,
        platform=req.platform,
        post_date=req.post_date,
        post_time=req.post_time,
        followers=req.followers,
        ad_boost=req.ad_boost,
    )

    # Rich Groq Insights: hashtags, peak times, best dates, deep explanations
    insights = generate_groq_insights(
        predictions=predictions,
        caption=req.caption,
        content=req.content,
        platform=req.platform,
        post_date=req.post_date,
        post_time=req.post_time,
        followers=req.followers,
        ad_boost=req.ad_boost,
    )

    # Persist to Supabase (non-blocking; ignore errors)
    row = save_prediction(
        caption=req.caption,
        content=req.content,
        platform=req.platform,
        post_date=req.post_date,
        post_time=req.post_time,
        followers=req.followers,
        ad_boost=req.ad_boost,
        predictions=predictions,
    )

    return PredictionResponse(
        id=row["id"] if row else None,
        predictions=predictions,
        explainability=tips,
        groq_insights=insights,
    )
