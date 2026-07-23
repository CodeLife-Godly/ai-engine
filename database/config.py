"""
Project configuration.

Loads environment variables from the .env file and exposes a single
configuration object for the entire application.
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration.
    """

    # ---------------- Database ----------------

    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "smartstock_ai")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # ---------------- API Keys ----------------

    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # ---------------- AI ----------------

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-large-en-v1.5",
    )

    DEVICE: str = os.getenv("DEVICE", "mps")

    # ---------------- Paths ----------------

    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    DATASET_DIR: Path = DATA_DIR / "datasets"

    MODEL_DIR: Path = PROJECT_ROOT / "models"
    TRAINED_MODEL_DIR: Path = MODEL_DIR / "trained"
    EMBEDDING_DIR: Path = MODEL_DIR / "embeddings"

    LOG_DIR: Path = PROJECT_ROOT / "logs"


config = Config()