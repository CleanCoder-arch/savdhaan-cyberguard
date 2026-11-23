"""SQLite scan history with a fixed metadata allowlist."""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
from pathlib import Path
from uuid import UUID

from core.models.indian_scam_tagger import ScamCategory
from core.schemas import RiskLevel, ScanReport, ScanStatus, ScanType

DEFAULT_DB_PATH = Path("artifacts") / "scans.sqlite3"

_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS scans (
    scan_id TEXT PRIMARY KEY NOT NULL CHECK (length(scan_id) = 36),
    scan_type TEXT NOT NULL CHECK (scan_type IN ('url', 'sms', 'qr', 'screenshot')),
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'partial')),
    risk_level TEXT CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    scam_category TEXT CHECK (scam_category IN (
        'upi_qr', 'digital_arrest', 'fake_kyc', 'job_task',
        'investment', 'utility', 'loan_app', 'lottery'
    )),
    latency_ms REAL CHECK (latency_ms >= 0),
    feed_freshness TEXT,
    CHECK ((risk_level IS NULL) = (risk_score IS NULL))
);
CREATE INDEX IF NOT EXISTS scans_by_time ON scans (timestamp DESC, scan_id DESC);
CREATE TABLE IF NOT EXISTS scan_feedback (
    scan_id TEXT PRIMARY KEY NOT NULL REFERENCES scans(scan_id) ON DELETE CASCADE,
    helpful INTEGER NOT NULL CHECK (helpful IN (0, 1)),
    updated_at TEXT NOT NULL
);
COMMIT;
"""

_SCAN_COLUMNS = (
    "scan_id, scan_type, timestamp, status, risk_level, risk_score, scam_category, latency_ms, feed_freshness"
)


@dataclass(frozen=True, slots=True)
class ScanMetadata:
    scan_id: UUID
    scan_type: ScanType
    timestamp: datetime
    status: ScanStatus
    risk_level: RiskLevel | None
    risk_score: int | None
    scam_category: ScamCategory | None
    latency_ms: float | None
    feed_freshness: datetime | None


@dataclass(frozen=True, slots=True)
class ScanFeedback:
    scan_id: UUID
    helpful: bool
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DailyScanActivity:
    day: date
    scan_count: int
    partial_count: int
    feedback_count: int
    helpful_count: int


def configured_db_path() -> Path:
    """Return the local scan history database path shared by API and UI."""
    return Path(os.getenv("SAVDHAAN_DB_PATH") or DEFAULT_DB_PATH)


def _scan_id(value: UUID | str) -> str:
    try:
        if isinstance(value, UUID):
            return str(value)
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Expected a valid scan UUID.") from None


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Scan timestamps require a time zone.")
    return value.astimezone(timezone.utc)


def _serialized_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _metadata_from_report(report: ScanReport) -> ScanMetadata:
    if not isinstance(report, ScanReport) or not isinstance(report.scan_id, UUID):
        raise TypeError("A validated scan report is required.")
    score = report.risk_score
    if score is not None and (type(score) is not int or not 0 <= score <= 100):
        raise ValueError("Invalid risk score.")
    latency = report.latency_ms
    if latency is not None and (type(latency) not in {float, int} or not isfinite(latency) or latency < 0):
        raise ValueError("Invalid scan latency.")
    category = next(iter(report.scam_categories), None)
    return ScanMetadata(
        scan_id=report.scan_id,
        scan_type=ScanType(report.scan_type),
        timestamp=_utc_datetime(report.timestamp),
        status=ScanStatus(report.status),
        risk_level=RiskLevel(report.risk_level) if report.risk_level is not None else None,
        risk_score=score,
        scam_category=ScamCategory(category) if category is not None else None,
        latency_ms=float(latency) if latency is not None else None,
        feed_freshness=_utc_datetime(report.feed_freshness) if report.feed_freshness is not None else None,
    )


def _read_scan(row: sqlite3.Row) -> ScanMetadata:
    return ScanMetadata(
        scan_id=UUID(row["scan_id"]),
        scan_type=ScanType(row["scan_type"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        status=ScanStatus(row["status"]),
        risk_level=RiskLevel(row["risk_level"]) if row["risk_level"] is not None else None,
        risk_score=row["risk_score"],
        scam_category=ScamCategory(row["scam_category"]) if row["scam_category"] is not None else None,
        latency_ms=row["latency_ms"],
        feed_freshness=datetime.fromisoformat(row["feed_freshness"]) if row["feed_freshness"] is not None else None,
    )


class ScanRepository:
    """Store report metadata and each scan's latest helpfulness answer separately."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path if path is not None else configured_db_path()).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path, timeout=5)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection

    def ping(self) -> bool:
        """Check that the initialized history database is available."""
        with self._connect() as connection:
            connection.execute("SELECT scan_id FROM scans LIMIT 0")
        return True

    def save_scan(self, report: ScanReport) -> ScanMetadata:
        """Save the allowlisted fields. Repeating the same scan is idempotent."""
        item = _metadata_from_report(report)
        with self._connect() as connection:
            inserted = connection.execute(
                f"INSERT INTO scans ({_SCAN_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (scan_id) DO NOTHING",
                (
                    str(item.scan_id),
                    item.scan_type,
                    _serialized_time(item.timestamp),
                    item.status,
                    item.risk_level,
                    item.risk_score,
                    item.scam_category,
                    item.latency_ms,
                    _serialized_time(item.feed_freshness) if item.feed_freshness is not None else None,
                ),
            )
            if inserted.rowcount == 0:
                previous = connection.execute(
                    f"SELECT {_SCAN_COLUMNS} FROM scans WHERE scan_id = ?", (str(item.scan_id),)
                ).fetchone()
                if _read_scan(previous) != item:
                    raise ValueError("A different result already uses this scan UUID.")
        return item

    def get_scan(self, scan_id: UUID | str) -> ScanMetadata | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans WHERE scan_id = ?", (_scan_id(scan_id),)
            ).fetchone()
        return _read_scan(row) if row is not None else None

    def list_scans(self, *, limit: int = 50, offset: int = 0) -> tuple[ScanMetadata, ...]:
        """Read newest first, with stable ordering for timestamp ties."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("History limit must be an integer from 1 to 500.")
        if type(offset) is not int or not 0 <= offset <= (1 << 63) - 1:
            raise ValueError("History offset must be a nonnegative integer.")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans ORDER BY timestamp DESC, scan_id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return tuple(_read_scan(row) for row in rows)

    def upsert_feedback(self, scan_id: UUID | str, helpful: bool) -> ScanFeedback:
        """Record the latest answer for an existing scan in one write transaction."""
        if type(helpful) is not bool:
            raise ValueError("Helpfulness must be a boolean answer.")
        key = _scan_id(scan_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated_at = datetime.now(timezone.utc)
            cursor = connection.execute(
                """
                INSERT INTO scan_feedback (scan_id, helpful, updated_at)
                SELECT scan_id, ?, ? FROM scans WHERE scan_id = ?
                ON CONFLICT(scan_id) DO UPDATE SET helpful = excluded.helpful, updated_at = excluded.updated_at
                """,
                (helpful, _serialized_time(updated_at), key),
            )
            if cursor.rowcount != 1:
                raise KeyError("Scan was not found.")
        return ScanFeedback(UUID(key), helpful, updated_at)

    def get_feedback(self, scan_id: UUID | str) -> ScanFeedback | None:
        key = _scan_id(scan_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT helpful, updated_at FROM scan_feedback WHERE scan_id = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return ScanFeedback(UUID(key), bool(row["helpful"]), datetime.fromisoformat(row["updated_at"]))

    def activity_by_day(self, days: int = 30, *, utc_offset_minutes: int = 0) -> tuple[DailyScanActivity, ...]:
        """Summarize scan dates at a fixed offset and fill in quiet days."""
        if type(days) is not int or not 1 <= days <= 366:
            raise ValueError("Activity window must be an integer from 1 to 366 days.")
        if type(utc_offset_minutes) is not int or not -1439 <= utc_offset_minutes <= 1439:
            raise ValueError("Activity offset must be an integer number of minutes within a day.")
        zone = timezone(timedelta(minutes=utc_offset_minutes))
        today = datetime.now(zone).date()
        first = today - timedelta(days=days - 1)
        start = datetime.combine(first, time.min, tzinfo=zone).astimezone(timezone.utc)
        end = datetime.combine(today + timedelta(days=1), time.min, tzinfo=zone).astimezone(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT date(substr(s.timestamp, 1, 19), ?) AS day,
                       COUNT(*) AS scan_count,
                       SUM(s.status = 'partial') AS partial_count,
                       COUNT(f.scan_id) AS feedback_count,
                       COALESCE(SUM(f.helpful), 0) AS helpful_count
                FROM scans AS s
                LEFT JOIN scan_feedback AS f ON s.scan_id = f.scan_id
                WHERE s.timestamp >= ? AND s.timestamp < ?
                GROUP BY day
                """,
                (f"{utc_offset_minutes:+d} minutes", _serialized_time(start), _serialized_time(end)),
            ).fetchall()
        activity = {
            date.fromisoformat(row["day"]): DailyScanActivity(
                date.fromisoformat(row["day"]),
                row["scan_count"],
                row["partial_count"],
                row["feedback_count"],
                row["helpful_count"],
            )
            for row in rows
        }
        return tuple(
            activity.get(day, DailyScanActivity(day, 0, 0, 0, 0))
            for day in (first + timedelta(days=offset) for offset in range(days))
        )
