"""Local paths shared by model bootstrap and scan startup."""

import os
from pathlib import Path

DEFAULT_MODEL_DIR = Path("artifacts") / "models"
URL_MODEL_FILENAME = "url_detector.joblib"
SMS_MODEL_FILENAME = "sms_detector.joblib"


def configured_model_dir() -> Path:
    """Return the directory of trusted local model artifacts."""
    return Path(os.getenv("SAVDHAAN_MODEL_DIR") or DEFAULT_MODEL_DIR)
