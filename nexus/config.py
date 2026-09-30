"""Global settings loaded from environment."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ai_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    memory_path: Path = Path(".nexus/memory.db")
    trace_path: Path = Path(".nexus/traces.db")
    max_retries: int = 3
    default_timeout: float = 120.0

    model_config = {
        "env_prefix": "NEXUS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
