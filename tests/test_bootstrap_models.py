import json
import subprocess
import sys
from pathlib import Path

from core.models.artifacts import SMS_MODEL_FILENAME, URL_MODEL_FILENAME
from core.models.sms_detector import SMSDetector
from core.models.url_detector import URLDetector
from scripts.bootstrap_models import bootstrap_models


def test_bootstrap_cli_builds_labeled_baselines_that_load(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_models.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        capture_output=True,
        timeout=45,
        check=True,
    )
    assert b"baseline models" in completed.stdout
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_data"] == "synthetic_baseline"
    assert len(manifest["corpus_sha256"]) == 64
    assert manifest["url"] == {"file": URL_MODEL_FILENAME, "sample_count": 32}
    assert manifest["sms"] == {"file": SMS_MODEL_FILENAME, "sample_count": 32}

    url = URLDetector.load(tmp_path / URL_MODEL_FILENAME)
    sms = SMSDetector.load(tmp_path / SMS_MODEL_FILENAME)
    assert url.predict("http://bank-kyc-update-login.test/secure/verify-account").probability > url.predict(
        "https://library.test/Agenda"
    ).probability
    assert sms.predict("Scan this UPI QR to receive money. Approve the collect request now.").probability > sms.predict(
        "Your library book is due on Friday. Renew through the library catalogue."
    ).probability

    baseline = sms.predict("Review the repayment schedule.")
    assert bootstrap_models(tmp_path) == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert SMSDetector.load(tmp_path / SMS_MODEL_FILENAME).predict("Review the repayment schedule.") == baseline
