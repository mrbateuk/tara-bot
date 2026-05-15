import os
from functools import lru_cache


@lru_cache()
def get_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required env var: {key}")
    return val


class Config:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    serpapi_key: str = os.getenv("SERPAPI_KEY", "")
    allowed_user_id: str = os.getenv("ALLOWED_USER_ID", "")
    affiliate_template = os.getenv("AFFILIATE_TEMPLATE")
