"""Local paths shared by model bootstrap and scan startup."""

from pathlib import Path

DEFAULT_MODEL_DIR = Path("artifacts") / "models"
URL_MODEL_FILENAME = "url_detector.joblib"
SMS_MODEL_FILENAME = "sms_detector.joblib"
