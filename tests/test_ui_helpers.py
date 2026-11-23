import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.ui_helpers import benchmark_rows, history_row, load_benchmark_results, make_scan_request, redacted_summary_bytes
from core.schemas import LinkReport, ScanReport, ScanType, UPIPayload
from core.storage.repository import ScanRepository


@pytest.mark.parametrize(
    ("scan_type", "value", "expected_field", "expected_value"),
    [
        (ScanType.URL, "  https://example.test/Case?key=private  ", "url", "https://example.test/Case?key=private"),
        (ScanType.SMS, "Line one\nLine two", "text", "Line one\nLine two"),
        (ScanType.QR, b"local image bytes", "image_bytes", b"local image bytes"),
        (ScanType.SCREENSHOT, "Reviewed content", "text", "Reviewed content"),
    ],
)
def test_request_helper_respects_core_input_types(scan_type, value, expected_field, expected_value) -> None:
    request = make_scan_request(scan_type, value, ocr_reviewed=scan_type is ScanType.SCREENSHOT)
    assert request.scan_type is scan_type
    assert getattr(request, expected_field) == expected_value


def test_screenshot_review_is_required_before_creating_request() -> None:
    with pytest.raises(ValueError, match="Review"):
        make_scan_request(ScanType.SCREENSHOT, "OCR output")
    with pytest.raises(ValidationError):
        make_scan_request(ScanType.SCREENSHOT, b"raw upload", ocr_reviewed=True)


def test_summary_and_history_use_safe_projections(tmp_path) -> None:
    report = ScanReport(
        scan_type=ScanType.QR,
        timestamp=datetime(2026, 11, 23, 20, 0, tzinfo=timezone.utc),
        upi=UPIPayload(pa="private-payee@upi", pn="private recipient"),
        links=(LinkReport(url="https://example.test/secret?token=private-token"),),
    )
    export = redacted_summary_bytes(report)
    assert json.loads(export)["link_count"] == 1
    assert json.loads(export)["upi_detected"] is True
    assert b"private" not in export
    metadata = ScanRepository(tmp_path / "history.sqlite3").save_scan(report)
    row = history_row(metadata)
    assert row["Scanned"] == "24 Nov 2026 01:30 IST"
    assert row["Input"] == "QR image"
    assert row["Risk"] == "Unassessed"
    assert row["Score"] is None
    assert "private" not in str(row)


def benchmark_artifact() -> dict:
    model = {
        "threshold": 0.9,
        "threshold_policy": "validation_f1_then_lower_fpr_then_higher_threshold",
        "lock_sha256": "a" * 64,
        "metrics": {"recall": 0.75, "precision": 0.5, "false_positive_rate": 0.25},
        "latency_ms": {"mean": 1, "p50": 0.5, "p95": 1.5},
        "splits": {name: {"samples": 12, "groups": 5} for name in ("train", "validation", "test")},
    }
    return {
        "schema_version": 1,
        "protocol": "grouped_holdout_v1",
        "dataset_kind": "synthetic_demo",
        "seed": 42,
        "models": {"url": model, "sms": {**model, "threshold": 0.85}},
    }


def test_benchmark_helper_displays_validated_metrics_and_group_counts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "benchmark_results.json"
    path.write_text(json.dumps(benchmark_artifact()), encoding="utf-8")
    monkeypatch.setenv("SAVDHAAN_BENCHMARK_RESULTS", str(path))
    results = load_benchmark_results()
    url, sms = benchmark_rows(results)
    assert results.dataset_kind == "synthetic_demo"
    assert url["Locked threshold"] == "0.900"
    assert sms["Locked threshold"] == "0.850"
    assert url["False positive rate"] == "25.0%"
    assert url["Test groups"] == 5
    assert url["P95 latency (ms)"] == "1.50"


@pytest.mark.parametrize("invalid", ["lock", "protocol", "threshold", "metric", "latency", "counts"])
def test_benchmark_helper_rejects_invalid_or_unlocked_artifacts(tmp_path, invalid) -> None:
    artifact = benchmark_artifact()
    model = artifact["models"]["url"]
    if invalid == "lock":
        del model["lock_sha256"]
    elif invalid == "protocol":
        artifact["protocol"] = "unlocked"
    elif invalid == "threshold":
        model["threshold"] = 1.5
    elif invalid == "metric":
        model["metrics"]["precision"] = "0.62"
    elif invalid == "latency":
        model["latency_ms"]["p95"] = -0.3
    else:
        model["splits"]["test"]["groups"] = 0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_benchmark_results(path)


def test_benchmark_helper_bounds_display_input(tmp_path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_benchmark_results(path)
