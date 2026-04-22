from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(..., description="Claude API key")
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"

    db_url: str = "sqlite:///./pipeline.db"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    fixtures_dir: Path = Path("tests/fixtures")

    use_replay_fixtures: bool = False
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
