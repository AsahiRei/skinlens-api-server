
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "SkinLens API Server"
    APP_VERSION: str = "1.0.0"
    APP_AUTHOR: str = "Arn Christian S. Rosales"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]
    MODEL_PATH: str = "models/detection/detection_model.keras"
    CLASS_PATH: str = "models/detection/class.json"
    IMG_SIZE: tuple[int, int] = (224, 224)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

