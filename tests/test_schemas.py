import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.models.indian_scam_tagger import ScamCategory, tag_indian_scams
from core.schemas import (
    EvidenceItem,
    LinkReport,
    RiskLevel,
    ScanIssue,
    ScanReport,
    ScanRequest,
    ScanType,
    UPIPayload,
)


@pytest.mark.parametrize(
    ("scan_type", "field", "value"),
    [
        ("url", "url", "https://example.test/Agenda"),
        ("sms", "text", "कृपया अपना OTP साझा न करें"),
        ("qr", "image_bytes", b"bounded image bytes"),
        ("screenshot", "image_bytes", b"bounded image bytes"),
        ("screenshot", "text", "Text confirmed during OCR review"),
    ],
)
def test_scan_request_accepts_one_matching_input(scan_type: str, field: str, value: str | bytes) -> None:
    result = ScanRequest.model_validate({"scan_type": scan_type, field: value})
    assert result.scan_type is ScanType(scan_type)
    assert getattr(result, field) == value
    assert str(value) not in repr(result)


@pytest.mark.parametrize(
    "payload",
    [
        {"scan_type": "sms"},
        {"scan_type": "sms", "text": " "},
        {"scan_type": "sms", "text": 123},
        {"scan_type": "url", "text": "https://example.test"},
        {"scan_type": "url", "url": "https://example.test", "text": "extra"},
        {"scan_type": "qr", "image_bytes": b""},
        {"scan_type": "qr", "image_bytes": bytearray(b"image")},
        {"scan_type": "qr", "image_bytes": b"x" * (10 * 1024 * 1024 + 1)},
        {"scan_type": "screenshot", "image_bytes": b"image", "text": "reviewed"},
        {"scan_type": "email", "text": "message"},
        {"scan_type": b"sms", "text": "message"},
        {"scan_type": "url", "url": "javascript:alert(1)"},
        {"scan_type": "sms", "text": "message", "unexpected": "value"},
    ],
)
def test_scan_request_rejects_wrong_inputs(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ScanRequest.model_validate(payload)


def test_scan_request_normalizes_sender_and_supports_base64_image_json() -> None:
    sms = ScanRequest(scan_type="sms", text="sample", sender_phone="+91 98765-43210")
    qr = ScanRequest.model_validate_json('{"scan_type":"qr","image_bytes":"_wA="}')
    assert sms.sender_phone == "+919876543210"
    assert sms.sender_phone not in repr(sms)
    assert qr.image_bytes == b"\xff\x00"
    assert json.loads(qr.model_dump_json())["image_bytes"] == "_wA="


def test_validation_error_string_omits_raw_input() -> None:
    secret = "https://secret-value@example.test/"
    with pytest.raises(ValidationError) as error:
        ScanRequest(scan_type="url", url=secret)
    assert secret not in str(error.value)
    assert secret not in repr(error.value.errors(include_input=False, include_context=False))


def test_partial_report_retains_completed_evidence_and_case_sensitive_link() -> None:
    evidence = EvidenceItem(source="taxonomy", code="fake_kyc", summary="KYC renewal request")
    link = LinkReport(url="https://example.test/Agenda?Session=PrivateValue", evidence=(evidence,))
    report = ScanReport(
        scan_type="sms",
        status="partial",
        issues=(ScanIssue(component="urlhaus", code="timeout"),),
        risk_level="HIGH",
        risk_score=85,
        scam_categories=(ScamCategory.FAKE_KYC,),
        links=(link,),
        evidence=(evidence,),
        latency_ms=15.1,
    )

    assert report.risk_level is RiskLevel.HIGH
    assert report.timestamp.tzinfo is timezone.utc
    assert report.links[0].url.endswith("/Agenda?Session=PrivateValue")
    assert "PrivateValue" not in repr(report)
    assert "KYC renewal request" not in repr(report)
    assert ScanReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "partial"},
        {"status": "complete", "issues": (ScanIssue(component="urlhaus", code="timeout"),)},
        {"risk_score": 10},
        {"risk_level": "LOW"},
        {"risk_level": "LOW", "risk_score": False},
        {"risk_level": "LOW", "risk_score": -1},
        {"risk_level": "LOW", "risk_score": 101},
        {"risk_level": "LOW", "risk_score": "15"},
        {"risk_level": "LOW", "risk_score": 15.0},
        {"risk_level": b"LOW", "risk_score": 10},
        {"scam_categories": ("arbitrary",)},
        {"timestamp": datetime(2026, 11, 24)},
        {"feed_freshness": datetime(2026, 11, 24)},
        {"latency_ms": float("inf")},
        {"latency_ms": -1},
    ],
)
def test_scan_report_rejects_inconsistent_or_wrongly_typed_fields(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ScanReport(scan_type="sms", **fields)


def test_complete_reports_can_represent_no_assessment() -> None:
    report = ScanReport(scan_type="qr")
    assert report.risk_level is None
    assert report.risk_score is None
    assert report.issues == ()


@pytest.mark.parametrize("code", ["", "UPI", "raw value", "x" * 65])
def test_evidence_and_issue_require_machine_readable_codes(code: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(source="extractor", code=code, summary="A result")
    with pytest.raises(ValidationError):
        ScanIssue(component="ocr", code=code)


def test_taxonomy_rule_ids_fit_evidence_schema() -> None:
    tags = tag_indian_scams("Scan the QR code to receive money.")
    assert tags
    evidence = tuple(
        EvidenceItem(source="taxonomy", code=rule_id, summary="Scam pattern matched")
        for tag in tags
        for rule_id in tag.rule_ids
    )
    report = ScanReport(
        scan_type="sms", scam_categories=tuple(tag.category for tag in tags), evidence=evidence
    )
    assert report.evidence[0].code == "upi_qr.scan_to_receive"


def test_evidence_rejects_unrecognized_source_and_oversized_summary() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(source="unrecognized", code="ocr", summary="Result")
    with pytest.raises(ValidationError):
        EvidenceItem(source="extractor", code="ocr", summary="a" * 501)


def test_link_report_can_have_a_separate_risk_assessment() -> None:
    link = LinkReport(url="https://example.test", risk_level="CRITICAL", risk_score=100)
    assert link.risk_score == 100
    with pytest.raises(ValidationError):
        LinkReport(url="https://example.test", risk_score=100)


def test_upi_payload_accepts_optional_fields_without_echoing_values() -> None:
    payload = UPIPayload(pa="demo.recipient@bank", pn="Test recipient", am=Decimal("12.25"), cu="INR", mc="0000")
    assert payload.am == Decimal("12.25")
    assert payload.pa not in repr(payload)
    assert payload.pn not in repr(payload)
    assert UPIPayload(pa="demo.recipient@bank").am is None


@pytest.mark.parametrize(
    "fields",
    [
        {"pa": "invalid"},
        {"am": "12.25"},
        {"am": Decimal("-0.01")},
        {"am": Decimal("0.001")},
        {"am": Decimal("10000000000.01")},
        {"am": Decimal("NaN")},
        {"cu": "USD"},
        {"mc": "123"},
        {"mc": "１２３４"},
        {"pn": "a" * 257},
    ],
)
def test_upi_payload_validates_fields(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UPIPayload.model_validate({"pa": "demo@bank"} | fields)
