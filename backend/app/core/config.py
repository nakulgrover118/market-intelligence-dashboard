from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, overridable via environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MARKET_")

    app_name: str = "Market Intelligence Dashboard"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
