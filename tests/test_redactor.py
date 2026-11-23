import json
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from uuid import UUID

import pytest
from PIL import Image, ImageDraw

from core.guidance.redactor import build_redacted_report, generate_alert_card, redact_text
from core.models.indian_scam_tagger import ScamCategory
from core.schemas import EvidenceItem, LinkReport, RiskLevel, ScanIssue, ScanReport, ScanStatus, UPIPayload


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Pay sample.user-1@upi now", "Pay [REDACTED ADDRESS] now"),
        ("sample+tag@example.invalid", "[REDACTED ADDRESS]"),
        ("sample%40okbank", "[REDACTED ADDRESS]"),
        ("upi://pay?pa=sample%40upi&pn=Test%20Name", "[REDACTED URL]"),
        ("https://user:secret@example.invalid/Agenda?code=secret#page", "[REDACTED URL]"),
        ("example.invalid/reset?nonce=secret#other", "[REDACTED URL]"),
        ("www.example.invalid/reset#secret", "[REDACTED URL]"),
        ("192.0.2.10:8080/reset?code=private", "[REDACTED URL]"),
        ("//[2001:db8::1]/reset?code=private", "[REDACTED URL]"),
        ("https://example.invalid/secret-private-path", "[REDACTED URL]"),
        ("access_token=abc-def&session_id=cookie", "access_token=[REDACTED SECRET]&session_id=[REDACTED SECRET]"),
        ("api_key = secret", "api_key = [REDACTED SECRET]"),
        ("Mobile +91 98765-43210", "Mobile [REDACTED PHONE]"),
        ("Mobile +91 (98765) 43210", "Mobile [REDACTED PHONE]"),
        ("Contact: (+91) ९८७६५ ४३२१०", "Contact: [REDACTED PHONE]"),
        ("Dial +919876543210", "Dial [REDACTED PHONE]"),
        ("Call 09876543210", "Call [REDACTED PHONE]"),
        ("Dial 011-23456789", "Dial [REDACTED PHONE]"),
        ("Phone number: 1800 11 1109", "Phone number: [REDACTED PHONE]"),
        ("Aadhaar 234512345678", "Aadhaar [REDACTED AADHAAR]"),
        ("Aadhaar 2345-1234-5678", "Aadhaar [REDACTED AADHAAR]"),
        ("Aadhaar XXXX XXXX 5678", "Aadhaar [REDACTED AADHAAR]"),
        ("PAN abcpa1234f", "PAN [REDACTED PAN]"),
        ("Bank OTP is 123456.", "Bank OTP is [REDACTED OTP]."),
        ("Bank OTP:\n  123456.", "Bank OTP:\n  [REDACTED OTP]."),
        ("123456 is your OTP.", "[REDACTED OTP] is your OTP."),
        ("Verification code = 123-456", "Verification code = [REDACTED OTP]"),
        ("OTP: １２３４", "OTP: [REDACTED OTP]"),
        ("OTP: १२३४", "OTP: [REDACTED OTP]"),
        ("OTP: 12\u200b3456", "OTP: [REDACTED OTP]"),
        ("Order 41372 on 2026-11-24 at 1930", "Order 41372 on 2026-11-24 at 1930"),
        ("Hotpot tastes good. Bill 12345.", "Hotpot tastes good. Bill 12345."),
        ("", ""),
    ],
)
def test_text_scrubber_replaces_sensitive_patterns(text, expected):
    assert redact_text(text) == expected
    assert redact_text(expected) == expected


def _report(name: str, url: str) -> ScanReport:
    return ScanReport(
        scan_id=UUID("e871363b-4443-469d-84b5-f04526021146"),
        scan_type="qr",
        timestamp=datetime(2026, 7, 10, 12, tzinfo=timezone.utc),
        status=ScanStatus.PARTIAL,
        risk_level=RiskLevel.HIGH,
        risk_score=82,
        scam_categories=(ScamCategory.UPI_QR, ScamCategory.FAKE_KYC, ScamCategory.UPI_QR),
        evidence=(EvidenceItem(source="extractor", code="decoded_qr", summary=f"Private name {name}"),),
        links=(LinkReport(url=url),),
        upi=UPIPayload(pa="private@upi", pn=name, am=Decimal("19.99"), cu="INR", mc="1234"),
        issues=(ScanIssue(component="provider", code="private_issue_code"),),
        latency_ms=9.5,
    )


def test_report_exports_exact_metadata_allowlist():
    report = _report("Private Test Name", "https://example.invalid/?token=private-token")
    exported = build_redacted_report(report)

    assert exported == {
        "scan_type": "qr",
        "status": "partial",
        "risk_level": "HIGH",
        "risk_score": 82,
        "scam_categories": ["upi_qr", "fake_kyc"],
        "evidence_count": 1,
        "link_count": 1,
        "issue_count": 1,
        "upi_detected": True,
    }
    serialized = json.dumps(exported)
    for private_value in ("Private Test Name", "private@upi", "private-token", "private_issue_code", str(report.scan_id)):
        assert private_value not in serialized


def test_report_preserves_an_unavailable_risk_score():
    report = ScanReport(scan_type="sms")
    exported = build_redacted_report(report)
    assert exported["risk_level"] is None
    assert exported["risk_score"] is None
    assert exported["scam_categories"] == []
    assert exported["upi_detected"] is False


def test_png_output_depends_only_on_public_fields():
    first = _report("Private Test Name", "https://example.invalid/?token=private-token")
    second = _report("Different Private Name", "https://other.invalid/?token=other-secret")
    first_card = generate_alert_card(first)
    assert first_card == generate_alert_card(second)

    with Image.open(BytesIO(first_card)) as image:
        image.verify()
    with Image.open(BytesIO(first_card)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width == 900
        assert image.height >= 400
        assert image.text == {}


def test_card_identifies_unavailable_risk_and_official_reporting(monkeypatch):
    draw_text = ImageDraw.ImageDraw.text
    drawn_strings = []

    def record_text(draw, position, text, **kwargs):
        drawn_strings.append(text)
        return draw_text(draw, position, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    report = ScanReport(
        scan_type="screenshot",
        status=ScanStatus.PARTIAL,
        scam_categories=tuple(ScamCategory),
        issues=(ScanIssue(component="ocr", code="unavailable"),),
    )
    generate_alert_card(report)
    assert "Risk level unavailable" in drawn_strings
    assert "Partial scan" in drawn_strings
    assert "https://cybercrime.gov.in/" in drawn_strings
    assert "Lottery scam" in " ".join(drawn_strings)
    assert "call 1930" in " ".join(drawn_strings)
