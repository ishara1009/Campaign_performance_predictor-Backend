"""
/api/history — retrieve past prediction records from Supabase.
"""

from fastapi import APIRouter, Query

from app.services.database import get_prediction_history

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
async def history(limit: int = Query(default=20, ge=1, le=100)):
    records = get_prediction_history(limit=limit)
    return {"data": records, "count": len(records)}
