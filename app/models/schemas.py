from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, time


class PredictionRequest(BaseModel):
    caption: str = Field(..., description="Post caption text")
    content: str = Field(..., description="Post content / body text")
    platform: str = Field(..., description="Social media platform name")
    post_date: str = Field(..., description="Scheduled post date (YYYY-MM-DD)")
    post_time: str = Field(..., description="Scheduled post time (HH:MM)")
    followers: int = Field(..., ge=0, description="Page / account follower count")
    ad_boost: int = Field(..., ge=0, le=1, description="1 if post will be boosted, 0 otherwise")


class PredictionResult(BaseModel):
    likes: float
    comments: float
    shares: float
    clicks: float
    timing_quality_score: float


class ExplainabilityTip(BaseModel):
    metric: str
    current_value: float
    suggestions: list[str]
    hashtags: list[str] = []


class GroqInsights(BaseModel):
    hashtags: list[str] = []
    peak_times: list[str] = []
    best_dates: list[str] = []
    likes_explanation: str = ""
    comments_explanation: str = ""
    shares_explanation: str = ""


class PredictionResponse(BaseModel):
    id: Optional[str] = None
    predictions: PredictionResult
    explainability: list[ExplainabilityTip]
    groq_insights: GroqInsights = GroqInsights()


class OCRResponse(BaseModel):
    text: str
    source: str  # "easyocr" | "vision_api"


class HistoryItem(BaseModel):
    id: str
    caption: str
    platform: str
    post_date: str
    post_time: str
    followers: int
    ad_boost: int
    likes: float
    comments: float
    shares: float
    clicks: float
    timing_quality_score: float
    created_at: str
