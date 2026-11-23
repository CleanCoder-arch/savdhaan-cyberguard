"""Authenticate API keys and keep atomic rate and webhook delivery state."""

import hmac
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from uuid import UUID, uuid4


class APIKeys:
    """Compare derived credentials without retaining keys in the control database."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        if any(not key or len(key) > 1024 for key in keys):
            raise ValueError("Configured API keys must contain between 1 and 1024 characters.")
        self._digests = tuple(self._digest(key) for key in dict.fromkeys(keys))

    @staticmethod
    def _digest(key: str) -> bytes:
        return hmac.digest(key.encode(), b"savdhaan:api-rate:v1", "sha256")

    @property
    def configured(self) -> bool:
        return bool(self._digests)

    def authenticate(self, key: str | None) -> bytes | None:
        if not key or len(key) > 1024:
            return None
        digest = self._digest(key)
        matches = tuple(hmac.compare_digest(digest, allowed) for allowed in self._digests)
        return digest if any(matches) else None


@dataclass(frozen=True, slots=True)
class RateDecision:
    limit: int
    remaining: int
    retry_after: int | None = None

    @property
    def allowed(self) -> bool:
        return self.retry_after is None

    @property
    def headers(self) -> dict[str, str]:
        headers = {"X-RateLimit-Limit": str(self.limit), "X-RateLimit-Remaining": str(self.remaining)}
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


@dataclass(frozen=True, slots=True)
class WebhookClaim:
    scan_id: UUID
    token: bytes | None
    complete: bool


class ControlStore:
    """Persistent sliding windows and idempotency leases shared by API workers."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self._path = Path(path)
        self._clock = clock
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_rate_event (
                    key_digest BLOB NOT NULL CHECK (length(key_digest) = 32),
                    arrived_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS api_rate_key_time ON api_rate_event (key_digest, arrived_at);
                CREATE INDEX IF NOT EXISTS api_rate_expiry ON api_rate_event (arrived_at);
                CREATE TABLE IF NOT EXISTS whatsapp_delivery (
                    event_digest BLOB PRIMARY KEY CHECK (length(event_digest) = 32),
                    lease_token BLOB CHECK (length(lease_token) = 16),
                    expires_at REAL NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS whatsapp_delivery_expiry ON whatsapp_delivery (expires_at);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path, timeout=1)) as connection, connection:
            yield connection

    def ping(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1 FROM api_rate_event LIMIT 1").fetchone()
            connection.execute("SELECT 1 FROM whatsapp_delivery LIMIT 1").fetchone()
        return True

    def consume(self, key_digest: bytes, *, limit: int, window_seconds: int) -> RateDecision:
        if limit < 1 or window_seconds < 1:
            raise ValueError("A positive rate and window are required.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            connection.execute("DELETE FROM api_rate_event WHERE arrived_at <= ?", (now - window_seconds,))
            used, oldest = connection.execute(
                "SELECT COUNT(*), MIN(arrived_at) FROM api_rate_event WHERE key_digest = ?",
                (key_digest,),
            ).fetchone()
            if used >= limit:
                return RateDecision(limit, 0, max(1, ceil(oldest + window_seconds - now)))
            connection.execute(
                "INSERT INTO api_rate_event VALUES (?, ?)", (key_digest, now)
            )
            return RateDecision(limit, limit - used - 1)

    def claim_delivery(self, digest: bytes) -> WebhookClaim:
        # The deterministic opaque scan ID makes storage recoverable after a process interruption.
        scan_id = UUID(bytes=digest[:16], version=4)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = self._clock()
            connection.execute("DELETE FROM whatsapp_delivery WHERE expires_at <= ?", (now,))
            row = connection.execute(
                "SELECT lease_token FROM whatsapp_delivery WHERE event_digest = ?", (digest,)
            ).fetchone()
            if row is not None:
                return WebhookClaim(scan_id, token=None, complete=row[0] is None)

            token = uuid4().bytes
            connection.execute(
                "INSERT INTO whatsapp_delivery VALUES (?, ?, ?)", (digest, token, now + 90)
            )
            return WebhookClaim(scan_id, token=token, complete=False)

    def finish_delivery(self, digest: bytes, token: bytes, *, complete: bool) -> None:
        with self._connect() as connection:
            if complete:
                connection.execute(
                    "UPDATE whatsapp_delivery SET lease_token = NULL, expires_at = ? "
                    "WHERE event_digest = ? AND lease_token = ?",
                    (self._clock() + 7 * 24 * 60 * 60, digest, token),
                )
            else:
                connection.execute(
                    "DELETE FROM whatsapp_delivery WHERE event_digest = ? AND lease_token = ?",
                    (digest, token),
                )
