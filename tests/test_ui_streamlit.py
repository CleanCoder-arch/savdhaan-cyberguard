import sqlite3
from contextlib import closing
from pathlib import Path

import cv2
import pytest
from streamlit.testing.v1 import AppTest

from core.extractors import ocr_engine
from core.extractors.ocr_engine import OCRResult
from core.scanner import ScanEngine
from core.schemas import ScanType
from core.storage.repository import ScanRepository
from scripts.bootstrap_models import bootstrap_models

_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def ui_models(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("streamlit-models")
    bootstrap_models(path)
    return path


@pytest.fixture
def dashboard(tmp_path, monkeypatch, ui_models):
    database = tmp_path / "ui-history.sqlite3"
    monkeypatch.setenv("SAVDHAAN_DB_PATH", str(database))
    monkeypatch.setenv("SAVDHAAN_MODEL_DIR", str(ui_models))
    monkeypatch.setenv("SAVDHAAN_BENCHMARK_RESULTS", str(tmp_path / "not-created.json"))
    app = AppTest.from_file(str(_APP), default_timeout=45).run()
    assert not app.exception
    return app, ScanRepository(database), database


def test_sample_scan_shows_explanation_and_upserts_latest_feedback(dashboard) -> None:
    app, repository, _ = dashboard
    assert [tab.label for tab in app.tabs] == ["Scanner", "History", "Analytics"]
    assert app.toggle(key="live_intel").value is False
    app.radio(key="scan_type").set_value(ScanType.SMS).run()
    app.button(key="sample_kyc").click().run()
    assert "KYC" in app.text_area(key="sms_input").value
    assert repository.list_scans() == ()

    app.button(key="scan_input").click().run()
    assert not app.exception
    report = app.session_state["scan_report"]
    assert report.scan_type is ScanType.SMS
    assert report.score_attribution
    assert [scan.scan_id for scan in repository.list_scans()] == [report.scan_id]
    assert any(expander.label == "Why this result?" for expander in app.expander)
    assert any(metric.label == "Scans in 30 days" and metric.value == "1" for metric in app.metric)
    assert len(app.get("download_button")) == 2

    app.button(key=f"helpful_yes_{report.scan_id}").click().run()
    assert not app.exception
    assert repository.get_feedback(report.scan_id).helpful is True
    app.button(key=f"helpful_no_{report.scan_id}").click().run()
    assert not app.exception
    assert repository.get_feedback(report.scan_id).helpful is False
    assert len(repository.list_scans()) == 1


def test_url_validation_does_not_echo_the_input_in_an_error(dashboard) -> None:
    app, repository, _ = dashboard
    app.text_input(key="url_input").set_value("javascript:private-token").run()
    app.button(key="scan_input").click().run()
    assert not app.exception
    assert [item.value for item in app.error] == ["Enter a valid HTTP or HTTPS URL."]
    assert repository.list_scans() == ()


def test_qr_upload_reaches_local_decoder_without_persisting_payment_data(dashboard) -> None:
    app, repository, database = dashboard
    payee = "private-virtual@upi"
    encoded = cv2.QRCodeEncoder_create().encode(f"upi://pay?pa={payee}&pn=PrivateRecipient&cu=INR")
    enlarged = cv2.resize(encoded, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", enlarged)
    assert ok
    app.radio(key="scan_type").set_value(ScanType.QR).run()
    app.file_uploader[0].set_value(("payment.png", png.tobytes(), "image/png")).run()
    app.button(key="scan_input").click().run()

    assert not app.exception
    report = app.session_state["scan_report"]
    assert report.upi is not None and report.upi.pa == payee
    assert repository.list_scans()[0].scan_type is ScanType.QR
    with closing(sqlite3.connect(database)) as connection:
        dump = "\n".join(connection.iterdump())
    assert payee not in dump
    assert "PrivateRecipient" not in dump


def test_screenshot_requires_ocr_then_submits_the_edited_text(dashboard, monkeypatch) -> None:
    app, repository, _ = dashboard
    ocr_calls: list[tuple[bytes, str]] = []
    scans: list[str] = []
    original_scan = ScanEngine.scan

    def read_ocr(image: bytes, *, language: str) -> OCRResult:
        ocr_calls.append((image, language))
        return OCRResult("OCR draft with typ0", 91.5)

    async def scan(self, request, *, offline: bool = False):
        assert request.scan_type is ScanType.SCREENSHOT
        assert request.image_bytes is None
        assert offline is True
        scans.append(request.text)
        return await original_scan(self, request, offline=offline)

    monkeypatch.setattr(ocr_engine, "extract_ocr", read_ocr)
    monkeypatch.setattr(ScanEngine, "scan", scan)
    app.radio(key="scan_type").set_value(ScanType.SCREENSHOT).run()
    app.file_uploader[0].set_value(("sample.png", b"screenshot one", "image/png")).run()
    assert not any(button.key == "scan_screenshot" for button in app.button)

    app.button(key="extract_ocr").click().run()
    assert not app.exception
    assert ocr_calls == [(b"screenshot one", "eng")]
    assert app.text_area(key="ocr_text").value == "OCR draft with typ0"
    assert repository.list_scans() == ()
    assert scans == []

    edited = "Click the link to keep your KYC active and enter your OTP."
    app.text_area(key="ocr_text").set_value(edited).run()
    app.button(key="scan_screenshot").click().run()
    assert not app.exception
    assert scans == [edited]
    assert repository.list_scans()[0].scan_type is ScanType.SCREENSHOT

    app.file_uploader[0].set_value(("sample.png", b"screenshot two", "image/png")).run()
    assert not app.exception
    assert not any(field.key == "ocr_text" for field in app.text_area)
    assert not any(button.key == "scan_screenshot" for button in app.button)
