"""
Explainability service — uses Groq's LLM to generate actionable,
research-grade recommendations for improving every predicted metric,
plus hashtag suggestions, peak posting times, best dates and deep
per-metric explanations.
"""

import json
import logging
from groq import Groq  # type: ignore

from app.config import GROQ_API_KEY

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"

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


# ── NEW: Groq Insights (hashtags, peak times, best dates, deep explanations) ─

_INSIGHTS_SYSTEM = """You are an expert social media marketing strategist with deep knowledge of
platform algorithms, audience psychology, and content optimisation — especially for Sri Lankan SMEs.
Respond ONLY with a single valid JSON object. No prose, no markdown, no code fences outside the JSON."""

_INSIGHTS_USER_TEMPLATE = """A social media post has been analysed with the following details:

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

Please provide a JSON object with exactly these keys:

{{
  "hashtags": [
    // 10-15 highly relevant, trending hashtags for this post (include # prefix).
    // Mix broad, niche and platform-specific tags. Prioritise Sri Lankan relevance where applicable.
  ],
  "peak_times": [
    // 4-6 specific peak posting time windows for {platform}, e.g. "Tuesday 7 PM – 9 PM"
    // Base on {platform} platform data and this audience size ({followers} followers).
  ],
  "best_dates": [
    // 4-6 best days / dates to post for maximum reach on {platform},
    // e.g. "Wednesday", "Friday", "Saturday morning".
    // Consider {platform} platform behaviour and local Sri Lankan context.
  ],
  "likes_explanation": "A detailed 3-4 sentence paragraph explaining WHY this post is predicted to receive {likes:.0f} likes and providing concrete, specific advice (caption tone, visual style, emotional hook, posting frequency, hashtag mix) to significantly increase that number.",
  "comments_explanation": "A detailed 3-4 sentence paragraph explaining WHY this post is predicted to receive {comments:.0f} comments. Provide specific tactics such as asking open-ended questions, controversy/debate framing, CTA phrasing, community engagement, reply strategies.",
  "shares_explanation": "A detailed 3-4 sentence paragraph explaining WHY this post is predicted to receive {shares:.0f} shares. Cover shareability factors: value-adding content, relatable themes, emotional resonance, meme potential, save-worthy information, and how to rewrite the caption to maximise share intent."
}}"""


def generate_groq_insights(
    predictions: dict,
    caption: str,
    content: str,
    platform: str,
    post_date: str,
    post_time: str,
    followers: int,
    ad_boost: int,
) -> dict:
    """
    Call Groq LLM and return a rich insights dict with hashtags, peak times,
    best dates, and deep per-metric explanations for likes, comments, shares.
    """
    from app.models.schemas import GroqInsights

    client = Groq(api_key=GROQ_API_KEY)

    user_message = _INSIGHTS_USER_TEMPLATE.format(
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
                {"role": "system", "content": _INSIGHTS_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fence if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return GroqInsights(
            hashtags=data.get("hashtags", []),
            peak_times=data.get("peak_times", []),
            best_dates=data.get("best_dates", []),
            likes_explanation=data.get("likes_explanation", ""),
            comments_explanation=data.get("comments_explanation", ""),
            shares_explanation=data.get("shares_explanation", ""),
        )

    except Exception as exc:
        logger.error("Groq insights failed: %s", exc)
        return GroqInsights()
