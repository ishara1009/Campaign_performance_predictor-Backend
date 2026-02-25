"""
OCR service — tries easyocr first, falls back to the vision API if text is
too short or confidence is low.  The vision API key is referenced only as
`token` to avoid hard-coded API names in the source.
"""

import logging
import os
import re
from io import BytesIO
from pathlib import Path

from PIL import Image  # type: ignore

from app.config import VISION_TOKEN

logger = logging.getLogger(__name__)

# Minimum character count to consider easyocr successful
_MIN_TEXT_LEN = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(raw_blocks: list) -> str:
    """Join OCR result blocks into a single clean paragraph.

    easyocr returns (bbox, text, confidence) tuples.
    We filter low-confidence tokens and collapse whitespace.
    """
    texts = []
    for item in raw_blocks:
        if len(item) == 3:
            _, text, conf = item
        else:
            text = item
            conf = 1.0
        text = text.strip()
        if conf >= 0.3 and text:
            texts.append(text)

    joined = " ".join(texts)
    # Collapse multiple spaces / newlines into single space
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def _extract_via_vision_api(image_bytes: bytes) -> str:
    """Use the external vision API to extract text from the image."""
    import base64
    import google.generativeai as genai  # type: ignore

    token = VISION_TOKEN
    genai.configure(api_key=token)

    model = genai.GenerativeModel("gemini-1.5-flash")

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    prompt = (
        "Extract ALL text visible in this image. "
        "The text may be in Sinhala, English, or a mix of both. "
        "Return a single clean paragraph — do NOT add bullet points, "
        "line breaks, or any extra formatting. "
        "Preserve the original wording and order."
    )

    response = model.generate_content([prompt, img])
    text = response.text.strip() if response.text else ""
    # Collapse newlines → space
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Returns {"text": str, "source": "easyocr" | "vision_api"}.

    Strategy:
      1. Try easyocr with English (fast, offline).
      2. If result is too short, use the vision API which handles
         Sinhala and mixed-language content reliably.
    """
    try:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = reader.readtext(img)
        text = _clean_text(results)

        if len(text) >= _MIN_TEXT_LEN:
            logger.info("OCR succeeded via easyocr (%d chars).", len(text))
            return {"text": text, "source": "easyocr"}

        logger.info("easyocr returned too little text (%d chars); trying vision API.", len(text))

    except Exception as exc:
        logger.warning("easyocr failed: %s — falling back to vision API.", exc)

    # Fallback
    try:
        text = _extract_via_vision_api(image_bytes)
        logger.info("OCR succeeded via vision API (%d chars).", len(text))
        return {"text": text, "source": "vision_api"}
    except Exception as exc:
        logger.error("Vision API also failed: %s", exc)
        return {"text": "", "source": "error"}
