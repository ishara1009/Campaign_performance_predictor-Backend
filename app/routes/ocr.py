"""
/api/ocr — extract text from an uploaded image.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import OCRResponse
from app.services.ocr_service import extract_text_from_image

router = APIRouter(prefix="/api", tags=["ocr"])

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_SIZE_MB = 10


@router.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WEBP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > _MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum allowed size is {_MAX_SIZE_MB} MB.",
        )

    result = extract_text_from_image(image_bytes)

    if not result["text"]:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the image.",
        )

    return OCRResponse(text=result["text"], source=result["source"])
