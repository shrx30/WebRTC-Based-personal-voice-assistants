from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Distributed Agent Platform"
    APP_VERSION: str = "0.1.0"

    OPENAI_API_KEY: str = ""
    QDRANT_URL: str = "http://localhost:6333"

    DATABASE_URL: str = (
        "postgresql://postgres:postgres@localhost:5432/research"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()