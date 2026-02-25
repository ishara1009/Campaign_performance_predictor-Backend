"""
Database service — thin wrapper around supabase-py for storing and
retrieving prediction history.
"""

import logging
from supabase import create_client, Client  # type: ignore
from app.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ---------------------------------------------------------------------------

def save_prediction(
    caption: str,
    content: str,
    platform: str,
    post_date: str,
    post_time: str,
    followers: int,
    ad_boost: int,
    predictions: dict,
) -> dict | None:
    """Insert a new prediction record and return the created row."""
    try:
        client = get_client()
        record = {
            "caption": caption,
            "content": content,
            "platform": platform,
            "post_date": post_date,
            "post_time": post_time,
            "followers": followers,
            "ad_boost": bool(ad_boost),
            "pred_likes": predictions.get("likes", 0),
            "pred_comments": predictions.get("comments", 0),
            "pred_shares": predictions.get("shares", 0),
            "pred_clicks": predictions.get("clicks", 0),
            "pred_timing_quality_score": predictions.get("timing_quality_score", 0),
        }
        response = client.table("predictions").insert(record).execute()
        return response.data[0] if response.data else None
    except Exception as exc:
        logger.error("Failed to save prediction: %s", exc)
        return None


def get_prediction_history(limit: int = 20) -> list[dict]:
    """Fetch the latest prediction records."""
    try:
        client = get_client()
        response = (
            client.table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        return []
