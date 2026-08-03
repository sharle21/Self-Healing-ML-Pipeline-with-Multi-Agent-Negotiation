import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", description="Claude API key (optional for heuristic-only mode)")
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"

    db_url: str = Field(
        default_factory=lambda: os.getenv(
            "DB_URL", "sqlite:///./pipeline.db"
        ),
        description="Database URL (PostgreSQL or SQLite)"
    )

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    model_path: Path = Path("models/lgbm_credit_default.joblib")
    fixtures_dir: Path = Path("tests/fixtures")
    traces_dir: Path = Path("traces")
    weight_versions_dir: Path = Path("weight_versions")

    use_replay_fixtures: bool = False
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_tenants_config() -> dict[str, Any]:
    text = files("self_healing_pipeline.config").joinpath("tenants.yaml").read_text()
    data = yaml.safe_load(text)
    return cast(dict[str, Any], data["tenants"])
