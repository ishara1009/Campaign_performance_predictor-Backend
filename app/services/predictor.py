"""
Prediction service — loads Transformer.keras, tokenizer and y_scaler at startup
and exposes a single predict() function.

NOTE: We rebuild the Transformer architecture manually and load only the weights.
This avoids Keras version-mismatch issues with serialised layer configs
(e.g. `quantization_config` in Embedding introduced in Keras 3.x).
"""

import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.config import (
    TRANSFORMER_MODEL_PATH,
    TOKENIZER_PATH,
    Y_SCALER_PATH,
    MAX_LEN,
    MAX_VOCAB,
    TARGETS,
    PLATFORM_MAP,
)

logger = logging.getLogger(__name__)

# ── Architecture constants (must match training exactly) ────────────────────
_EMB = 128
_NUM_FEATURES = 6   # platform_id, post_hour, day_of_week, is_weekend, followers_log, ad_boost
_N_OUT = len(TARGETS)   # 5


def _build_transformer() -> tf.keras.Model:
    """Recreate the Transformer architecture from training."""
    seq_input = tf.keras.Input((MAX_LEN,), name="seq_input")
    num_input = tf.keras.Input((_NUM_FEATURES,), name="num_input")

    x = tf.keras.layers.Embedding(MAX_VOCAB, _EMB, mask_zero=True)(seq_input)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=_EMB)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, num_input])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(_N_OUT)(x)

    model = tf.keras.Model([seq_input, num_input], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.Huber(),
    )
    return model


class Predictor:
    """Singleton that holds the loaded model artifacts."""

    _instance: "Predictor | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        logger.info("Building Transformer architecture …")
        self.model = _build_transformer()

        # Build the model by running a dummy forward pass so all weight shapes
        # are fully materialised before we restore checkpointed values.
        dummy_seq = np.zeros((1, MAX_LEN), dtype="int32")
        dummy_num = np.zeros((1, _NUM_FEATURES), dtype="float32")
        self.model.predict([dummy_seq, dummy_num], verbose=0)

        logger.info("Loading Transformer weights from %s …", TRANSFORMER_MODEL_PATH)
        try:
            self.model.load_weights(TRANSFORMER_MODEL_PATH)
            logger.info("Weights loaded successfully via load_weights.")
        except Exception as weight_err:
            # Fallback: try full model load with a custom Embedding that strips
            # the unrecognised quantization_config kwarg.
            logger.warning(
                "load_weights failed (%s) — attempting full model load with compat patch …",
                weight_err,
            )

            class _CompatEmbedding(tf.keras.layers.Embedding):
                def __init__(self, *args, quantization_config=None, **kwargs):
                    super().__init__(*args, **kwargs)

            self.model = tf.keras.models.load_model(
                TRANSFORMER_MODEL_PATH,
                custom_objects={"Embedding": _CompatEmbedding},
            )

        logger.info("Loading tokenizer …")
        with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
            self.tokenizer = tokenizer_from_json(f.read())

        logger.info("Loading y_scaler …")
        with open(Y_SCALER_PATH, "rb") as f:
            self.y_scaler = pickle.load(f)

        self._loaded = True
        logger.info("All model artifacts loaded.")

    # ------------------------------------------------------------------

    def _encode_text(self, caption: str, content: str) -> np.ndarray:
        text = f"{caption} {content}".strip()
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        return padded.astype("int32")

    def _encode_numeric(
        self,
        platform: str,
        post_date: str,
        post_time: str,
        followers: int,
        ad_boost: int,
    ) -> np.ndarray:
        import pandas as pd

        # Platform ID (fallback to 0 if unknown)
        platform_id = PLATFORM_MAP.get(platform, 0)

        # Time features
        hour = pd.to_datetime(post_time, format="%H:%M", errors="coerce").hour
        if pd.isna(hour):
            hour = 12

        dt = pd.to_datetime(post_date, errors="coerce")
        day_of_week = dt.dayofweek if not pd.isna(dt) else 0
        is_weekend = int(day_of_week in [5, 6])

        followers_log = float(np.log1p(max(followers, 0)))

        features = [
            platform_id,
            hour,
            day_of_week,
            is_weekend,
            followers_log,
            int(ad_boost),
        ]
        return np.array([features], dtype="float32")

    # ------------------------------------------------------------------

    def predict(
        self,
        caption: str,
        content: str,
        platform: str,
        post_date: str,
        post_time: str,
        followers: int,
        ad_boost: int,
    ) -> dict[str, float]:
        self.load()

        x_text = self._encode_text(caption, content)
        x_num = self._encode_numeric(platform, post_date, post_time, followers, ad_boost)

        raw = self.model.predict([x_text, x_num], verbose=0)          # shape (1, 5)
        inverse = self.y_scaler.inverse_transform(raw)[0]              # back to scaled space

        result: dict[str, float] = {}
        for i, target in enumerate(TARGETS):
            val = float(inverse[i])
            if target in ("likes", "comments", "shares", "clicks"):
                val = float(np.expm1(max(val, 0)))                     # undo log1p
            val = round(max(val, 0), 2)
            result[target] = val

        return result


# Global singleton
predictor = Predictor()
