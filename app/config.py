import os
from dotenv import load_dotenv

load_dotenv()

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "SavedModels")
TRANSFORMER_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "Transformer.keras")
TOKENIZER_PATH = os.path.join(SAVED_MODELS_DIR, "tokenizer.json")
Y_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "y_scaler.pkl")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# External APIs
VISION_TOKEN = os.getenv("VISION_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model constants (must match training)
MAX_VOCAB = 30000
MAX_LEN = 80
TARGETS = ["likes", "comments", "shares", "clicks", "timing_quality_score"]

# Platform mapping — sorted alphabetical (matches pandas category codes from training)
PLATFORM_MAP: dict[str, int] = {
    "Facebook": 0,
    "Instagram": 1,
    "TikTok": 2,
    "Twitter": 3,
    "YouTube": 4,
}
