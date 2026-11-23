import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from core.models.indian_scam_tagger import ScamCategory
from core.schemas import EvidenceItem, LinkReport, RiskLevel, ScanIssue, ScanReport, ScanStatus, ScanType, UPIPayload
from core.storage.repository import ScanRepository, configured_db_path


def test_sqlite_comprises_only_allowlisted_metadata_and_feedback(tmp_path) -> None:
    path = tmp_path / "scans.sqlite3"
    repository = ScanRepository(path)
    private_url = "https://private.example.test/Agenda?token=SecretSession91"
    private_phone = "+919876543219"
    private_text = "Private SMS Full Content 01234"
    private_upi = "quiet.recipient@bank"
    private_name = "Private Customer Dana"
    report = ScanReport(
        scan_type="sms",
        status="partial",
        issues=(ScanIssue(component="whois", code="offline"),),
        risk_level="HIGH",
        risk_score=79,
        scam_categories=(ScamCategory.UPI_QR, ScamCategory.FAKE_KYC),
        evidence=(EvidenceItem(source="extractor", code="private_text", summary=f"{private_text} {private_phone}"),),
        links=(LinkReport(url=private_url),),
        upi=UPIPayload(pa=private_upi, pn=private_name, am=Decimal("12.34"), cu="INR"),
        latency_ms=27.25,
        feed_freshness=datetime.now(timezone.utc),
    )

    metadata = repository.save_scan(report)
    repository.upsert_feedback(report.scan_id, True)
    assert repository.get_scan(str(report.scan_id)) == metadata
    assert metadata.scan_type is ScanType.SMS
    assert metadata.status is ScanStatus.PARTIAL
    assert metadata.risk_level is RiskLevel.HIGH
    assert metadata.scam_category is ScamCategory.UPI_QR
    assert metadata.feed_freshness == report.feed_freshness
    assert metadata.latency_ms == 27.25

    with closing(sqlite3.connect(path)) as connection:
        assert [column[1] for column in connection.execute("PRAGMA table_info(scans)")] == [
            "scan_id", "scan_type", "timestamp", "status", "risk_level", "risk_score",
            "scam_category", "latency_ms", "feed_freshness",
        ]
        assert [column[1] for column in connection.execute("PRAGMA table_info(scan_feedback)")] == [
            "scan_id", "helpful", "updated_at",
        ]
        tables = connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name").fetchall()
        assert tables == [("scan_feedback",), ("scans",)]
        persisted = "\n".join(connection.iterdump()).encode()
        persisted += b"".join(file.read_bytes() for file in tmp_path.glob("scans.sqlite3*"))
    for private in (private_url, private_phone, private_text, private_upi, private_name, "SecretSession91", "offline"):
        assert private.encode() not in persisted


def test_feedback_replaces_the_previous_answer_without_modifying_scan(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    report = ScanReport(scan_type="url", risk_level="LOW", risk_score=12)
    original = repository.save_scan(report)
    assert repository.get_feedback(report.scan_id) is None
    first = repository.upsert_feedback(report.scan_id, True)
    latest = repository.upsert_feedback(str(report.scan_id), False)

    assert first.helpful is True
    assert latest.helpful is False
    assert latest.updated_at >= first.updated_at
    assert repository.get_feedback(report.scan_id) == latest
    assert repository.get_scan(report.scan_id) == original
    with closing(sqlite3.connect(tmp_path / "scans.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scan_feedback").fetchone() == (1,)


def test_feedback_rejects_unknown_scans_and_non_boolean_values(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    unrecorded = uuid4()
    assert repository.get_scan(unrecorded) is None
    with pytest.raises(KeyError, match="Scan was not found"):
        repository.upsert_feedback(unrecorded, True)
    with pytest.raises(ValueError, match="boolean"):
        repository.upsert_feedback(unrecorded, "false")
    with pytest.raises(ValueError, match="UUID") as error:
        repository.get_scan("private-identifier@example.test")
    assert "private-identifier" not in str(error.value)


def test_scan_insert_is_retry_safe_and_cannot_replace_history(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    report = ScanReport(scan_type="url", risk_level="LOW", risk_score=4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        written = tuple(executor.map(repository.save_scan, [report] * 8))

    assert all(item == written[0] for item in written)
    assert repository.list_scans() == written[:1]
    with pytest.raises(ValueError, match="already uses"):
        repository.save_scan(report.model_copy(update={"risk_score": 5}))
    assert repository.get_scan(report.scan_id).risk_score == 4


def test_report_storage_rejects_raw_dicts_and_unvalidated_fields(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    with pytest.raises(TypeError):
        repository.save_scan({"scan_type": "sms", "text": "Private payload"})
    with pytest.raises(TypeError):
        repository.save_scan(ScanReport.model_construct(scan_type="sms", scan_id="private@example.test"))
    with pytest.raises(ValueError):
        repository.save_scan(ScanReport.model_construct(scan_type="sms", latency_ms="private@example.test"))
    assert repository.list_scans() == ()


def test_history_orders_and_pages_utc_timestamps(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    india_offset = timezone(timedelta(hours=5, minutes=30))
    early = ScanReport(scan_type="sms", timestamp=datetime(2026, 11, 25, 0, 15, tzinfo=india_offset))
    later = ScanReport(scan_type="qr", timestamp=datetime(2026, 11, 24, 19, 0, tzinfo=timezone.utc))
    repository.save_scan(later)
    repository.save_scan(early)
    assert [row.scan_id for row in repository.list_scans()] == [later.scan_id, early.scan_id]
    page, = repository.list_scans(limit=1, offset=1)
    assert page.timestamp == datetime(2026, 11, 24, 18, 45, tzinfo=timezone.utc)
    assert page.risk_score is None
    with pytest.raises(ValueError):
        repository.list_scans(limit=0)


def test_activity_fills_quiet_days_and_counts_only_latest_feedback(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    now = datetime.now(timezone.utc)
    report = ScanReport(
        scan_type="qr", timestamp=now, status="partial", issues=(ScanIssue(component="qr", code="decode_failed"),),
    )
    older = ScanReport(scan_type="sms", timestamp=now - timedelta(days=1))
    repository.save_scan(report)
    repository.save_scan(older)
    repository.upsert_feedback(report.scan_id, True)
    repository.upsert_feedback(report.scan_id, False)
    repository.upsert_feedback(older.scan_id, True)
    quiet, yesterday, current = repository.activity_by_day(3)

    assert quiet.scan_count == 0
    assert yesterday.scan_count == yesterday.feedback_count == yesterday.helpful_count == 1
    assert yesterday.partial_count == 0
    assert current.scan_count == current.partial_count == current.feedback_count == 1
    assert current.helpful_count == 0
    assert current.day == now.date()


def test_default_database_path_is_shared_config_and_ping_is_real(tmp_path, monkeypatch) -> None:
    path = tmp_path / "data" / "scans.sqlite3"
    monkeypatch.setenv("SAVDHAAN_DB_PATH", str(path))
    assert configured_db_path() == path
    repository = ScanRepository()
    assert repository.ping() is True
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("ALTER TABLE scans RENAME TO missing_scans")
    with pytest.raises(sqlite3.Error):
        repository.ping()


def test_activity_uses_local_day_boundaries_when_configured(tmp_path) -> None:
    repository = ScanRepository(tmp_path / "scans.sqlite3")
    ist = timezone(timedelta(minutes=330))
    yesterday = datetime.now(ist).date() - timedelta(days=1)
    midnight = datetime.combine(yesterday, datetime.min.time(), ist)
    repository.save_scan(ScanReport(scan_type="sms", timestamp=midnight - timedelta(microseconds=1)))
    repository.save_scan(ScanReport(scan_type="url", timestamp=midnight))

    days = {entry.day: entry.scan_count for entry in repository.activity_by_day(4, utc_offset_minutes=330)}
    assert days[yesterday] == days[yesterday - timedelta(days=1)] == 1
    utc_days = {entry.day: entry.scan_count for entry in repository.activity_by_day(4)}
    assert utc_days[yesterday - timedelta(days=1)] == 2
    with pytest.raises(ValueError, match="offset"):
        repository.activity_by_day(1, utc_offset_minutes=1440)
