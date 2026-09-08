from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, overridable via environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MARKET_")

    app_name: str = "Market Intelligence Dashboard"
    environment: str = "development"
    # Comma-separated in the env var (MARKET_CORS_ORIGINS=https://a.com,https://b.com);
    # defaults to the Vite dev server so local development works with no
    # config. A real deployment must set this to the deployed frontend's
    # actual origin — see docs/deployment.md.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
