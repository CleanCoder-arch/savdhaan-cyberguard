"""Format the local UI's scan requests, history, and benchmark display data."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.guidance.redactor import build_redacted_report
from core.guidance.rbi_helplines import get_scam_playbook
from core.schemas import ScanReport, ScanRequest, ScanType
from core.storage.repository import ScanMetadata

INDIA_TIME = timezone(timedelta(hours=5, minutes=30), name="IST")
SCAN_TYPE_LABELS = {
    ScanType.URL: "URL",
    ScanType.SMS: "SMS",
    ScanType.QR: "QR image",
    ScanType.SCREENSHOT: "Screenshot",
}
DEFAULT_BENCHMARK_PATH = Path("artifacts") / "benchmarks" / "benchmark_results.json"
_MAX_BENCHMARK_BYTES = 1024 * 1024


def make_scan_request(
    scan_type: ScanType, value: str | bytes, *, ocr_reviewed: bool = False
) -> ScanRequest:
    """Validate the selected input and require explicit review for screenshots."""
    if scan_type is ScanType.SCREENSHOT and not ocr_reviewed:
        raise ValueError("Review the extracted text before starting the scan.")
    if scan_type is ScanType.URL:
        return ScanRequest(scan_type=scan_type, url=value.strip() if isinstance(value, str) else value)
    if scan_type is ScanType.QR:
        return ScanRequest(scan_type=scan_type, image_bytes=value)
    return ScanRequest(scan_type=scan_type, text=value)


def redacted_summary_bytes(report: ScanReport) -> bytes:
    """Serialize the core export allowlist for download."""
    return (json.dumps(build_redacted_report(report), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _local_timestamp(value: datetime | None) -> str:
    return value.astimezone(INDIA_TIME).strftime("%d %b %Y %H:%M IST") if value is not None else "—"


def history_row(metadata: ScanMetadata) -> dict[str, str | int | float | None]:
    """Display saved metadata without merging in any current scanner input."""
    return {
        "Scanned": _local_timestamp(metadata.timestamp),
        "Input": SCAN_TYPE_LABELS[metadata.scan_type],
        "Status": metadata.status.value.title(),
        "Risk": metadata.risk_level.value if metadata.risk_level is not None else "Unassessed",
        "Score": metadata.risk_score,
        "Category": get_scam_playbook(metadata.scam_category).title if metadata.scam_category is not None else "—",
        "Latency (ms)": round(metadata.latency_ms, 1) if metadata.latency_ms is not None else None,
        "Threat intel checked": _local_timestamp(metadata.feed_freshness),
        "Scan ID": str(metadata.scan_id),
    }


_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
_Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_PositiveCount = Annotated[int, Field(gt=0)]


class _BenchmarkSchema(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)


class BenchmarkMetrics(_BenchmarkSchema):
    recall: _Probability
    precision: _Probability
    false_positive_rate: _Probability


class BenchmarkLatency(_BenchmarkSchema):
    mean: _Milliseconds
    p50: _Milliseconds
    p95: _Milliseconds


class BenchmarkSplit(_BenchmarkSchema):
    samples: _PositiveCount
    groups: _PositiveCount


class BenchmarkSplits(_BenchmarkSchema):
    train: BenchmarkSplit
    validation: BenchmarkSplit
    test: BenchmarkSplit


class BenchmarkModel(_BenchmarkSchema):
    threshold: _Probability
    threshold_policy: Literal["validation_f1_then_lower_fpr_then_higher_threshold"]
    lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    metrics: BenchmarkMetrics
    latency_ms: BenchmarkLatency
    splits: BenchmarkSplits


class BenchmarkModels(_BenchmarkSchema):
    url: BenchmarkModel
    sms: BenchmarkModel


class BenchmarkResults(_BenchmarkSchema):
    schema_version: Literal[1]
    protocol: Literal["grouped_holdout_v1"]
    dataset_kind: Literal["synthetic_demo", "provided"]
    seed: int
    models: BenchmarkModels


def load_benchmark_results(path: Path | None = None) -> BenchmarkResults:
    """Read a bounded, typed projection of local benchmark results."""
    if path is None:
        path = Path(os.environ.get("SAVDHAAN_BENCHMARK_RESULTS") or DEFAULT_BENCHMARK_PATH)
    with path.open("rb") as file:
        data = file.read(_MAX_BENCHMARK_BYTES + 1)
    if len(data) > _MAX_BENCHMARK_BYTES:
        raise ValueError("The benchmark results file exceeds the size limit.")
    return BenchmarkResults.model_validate_json(data)


def benchmark_rows(results: BenchmarkResults) -> list[dict[str, str | int]]:
    return [
        {
            "Detector": name,
            "Locked threshold": f"{model.threshold:.3f}",
            "Recall": f"{model.metrics.recall:.1%}",
            "Precision": f"{model.metrics.precision:.1%}",
            "False positive rate": f"{model.metrics.false_positive_rate:.1%}",
            "P50 latency (ms)": f"{model.latency_ms.p50:.2f}",
            "P95 latency (ms)": f"{model.latency_ms.p95:.2f}",
            "Train groups": model.splits.train.groups,
            "Validation groups": model.splits.validation.groups,
            "Test groups": model.splits.test.groups,
            "Test samples": model.splits.test.samples,
        }
        for name, model in (("URL", results.models.url), ("SMS", results.models.sms))
    ]
