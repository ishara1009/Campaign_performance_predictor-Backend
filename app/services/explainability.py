"""
Explainability service — uses Groq's LLM to generate actionable,
research-grade recommendations for improving every predicted metric.
"""

import json
import logging
from groq import Groq  # type: ignore

from app.config import GROQ_API_KEY

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama3-8b-8192"

_SYSTEM_PROMPT = """You are an expert social media marketing strategist and data scientist.
Your role is to analyse the predicted performance metrics of a social media post and provide
detailed, actionable recommendations to improve each metric.

Your suggestions must be specific, practical, and evidence-based.
Always recommend relevant hashtags where applicable.
Consider platform-specific best practices, peak engagement times, content quality, and audience psychology.
Focus on Sri Lankan SME context when relevant.

Respond ONLY with a valid JSON array — no prose outside the JSON."""

_USER_TEMPLATE = """A social media post has the following predicted performance:

Platform  : {platform}
Caption   : {caption}
Content   : {content}
Post Date : {post_date}
Post Time : {post_time}
Followers : {followers}
Ad Boost  : {ad_boost}

Predicted Metrics:
  Likes                : {likes:.1f}
  Comments             : {comments:.1f}
  Shares               : {shares:.1f}
  Clicks               : {clicks:.1f}
  Timing Quality Score : {timing_quality_score:.2f}

For EACH metric provide a JSON object with:
  - "metric"        : metric name
  - "current_value" : the predicted value
  - "suggestions"   : array of 4-6 specific, actionable tips to increase this metric
  - "hashtags"      : array of 5-8 relevant trending hashtags (only for Likes, Comments, Shares metrics)

Return a JSON array of 5 objects (one per metric).

Focus areas per metric:
  Likes   → caption attractiveness, visual quality, posting time, hashtag strategy, emotional triggers
  Comments → conversation-starting captions, questions, polls, community engagement
  Shares  → shareability, value-adding content, relatable themes, trending topics
  Clicks  → call-to-action strength, curiosity gap, link placement, ad copy effectiveness
  Timing Quality Score → optimal day of week, peak hours for the platform, audience activity patterns, event-based posting
"""


def generate_explainability(
    predictions: dict,
    caption: str,
    content: str,
    platform: str,
    post_date: str,
    post_time: str,
    followers: int,
    ad_boost: int,
) -> list[dict]:
    """
    Call Groq LLM and return a list of explainability tip objects.
    Each object: { metric, current_value, suggestions: [...], hashtags: [...] }
    """
    client = Groq(api_key=GROQ_API_KEY)

    user_message = _USER_TEMPLATE.format(
        platform=platform,
        caption=caption,
        content=content,
        post_date=post_date,
        post_time=post_time,
        followers=followers,
        ad_boost="Yes" if ad_boost else "No",
        **predictions,
    )

    try:
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fence if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        tips = json.loads(raw)
        return tips

    except Exception as exc:
        logger.error("Groq explainability failed: %s", exc)
        # Return a minimal fallback so the API never crashes
        return [
            {
                "metric": m,
                "current_value": predictions.get(m, 0),
                "suggestions": ["Could not generate suggestions at this time."],
                "hashtags": [],
            }
            for m in ["likes", "comments", "shares", "clicks", "timing_quality_score"]
        ]
