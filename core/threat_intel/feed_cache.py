"""Store sanitized threat lookup outcomes with per-provider freshness limits."""

import hashlib
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

import httpx

from core.threat_intel.live_providers import IntelProvider, IntelResult, IntelStatus
from core.validators import validate_url

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS threat_lookup (
        provider TEXT NOT NULL CHECK (provider IN ('whois', 'urlhaus')),
        site_key BLOB NOT NULL CHECK (length(site_key) = 32),
        checked_at REAL NOT NULL,
        domain_age_days INTEGER,
        listed INTEGER,
        PRIMARY KEY (provider, site_key),
        CHECK (
            (provider = 'whois' AND domain_age_days IS NOT NULL AND domain_age_days >= 0 AND listed IS NULL)
            OR (provider = 'urlhaus' AND domain_age_days IS NULL AND listed IS NOT NULL AND listed IN (0, 1))
        )
    ) WITHOUT ROWID
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(moment: datetime) -> float:
    if moment.utcoffset() is None:
        raise ValueError("Threat cache timestamps must have a time zone.")
    return moment.timestamp()


def _site_key(host: str) -> bytes:
    if not isinstance(host, str) or not host or len(host) > 254 or "%" in host:
        raise ValueError("Provide a hostname or IP address as the cache key.")
    try:
        canonical = ip_address(host).compressed
    except ValueError:
        if any(character in host for character in "/\\@:#?[]"):
            raise ValueError("Provide a hostname or IP address as the cache key.") from None
        try:
            validate_url(f"https://{host}")
            canonical = httpx.URL(scheme="https", host=host).raw_host.decode("ascii").removesuffix(".")
        except (ValueError, UnicodeError, httpx.InvalidURL):
            raise ValueError("Provide a valid hostname as the cache key.") from None
    return hashlib.sha256(canonical.encode("ascii")).digest()


@dataclass(frozen=True, slots=True)
class CachedIntel:
    """A cached outcome and its freshness at lookup time."""

    result: IntelResult
    expires_at: datetime
    is_fresh: bool


class FeedCache:
    """An on-disk lookup cache for successful observations, keyed by a hostname digest."""

    def __init__(
        self,
        path: str | Path,
        *,
        whois_ttl: timedelta = timedelta(days=1),
        urlhaus_ttl: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._path = Path(path).resolve()
        self._clock = clock
        self._ttl = {IntelProvider.WHOIS: whois_ttl, IntelProvider.URLHAUS: urlhaus_ttl}
        if any(ttl <= timedelta(0) for ttl in self._ttl.values()):
            raise ValueError("Threat cache freshness limits must be positive.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(_CREATE_TABLE)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path, timeout=0.25)) as connection, connection:
            yield connection

    def lookup(self, provider: IntelProvider, host: str) -> CachedIntel | None:
        """Read an observation with an explicit freshness flag."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT checked_at, domain_age_days, listed FROM threat_lookup WHERE provider = ? AND site_key = ?",
                (provider.value, _site_key(host)),
            ).fetchone()
        if row is None:
            return None

        timestamp, age, listed = row
        checked_at = datetime.fromtimestamp(timestamp, timezone.utc)
        expires_at = checked_at + self._ttl[provider]
        result = IntelResult(
            provider=provider,
            status=IntelStatus.OK,
            checked_at=checked_at,
            domain_age_days=age,
            listed=None if listed is None else bool(listed),
        )
        return CachedIntel(result, expires_at, timestamp <= _timestamp(self._clock()) < expires_at.timestamp())

    def get(self, provider: IntelProvider, host: str) -> IntelResult | None:
        """Return only an observation still eligible for scoring."""
        entry = self.lookup(provider, host)
        return entry.result if entry is not None and entry.is_fresh else None

    def put(self, host: str, result: IntelResult) -> None:
        """Record a successful observation without replacing a newer one."""
        if result.status is not IntelStatus.OK:
            return
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO threat_lookup (provider, site_key, checked_at, domain_age_days, listed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (provider, site_key) DO UPDATE SET
                    checked_at = excluded.checked_at,
                    domain_age_days = excluded.domain_age_days,
                    listed = excluded.listed
                WHERE excluded.checked_at > threat_lookup.checked_at
                """,
                (
                    result.provider.value,
                    _site_key(host),
                    _timestamp(result.checked_at),
                    result.domain_age_days,
                    result.listed,
                ),
            )

    def prune_expired(self) -> int:
        """Remove naturally expired observations and return the number removed."""
        now = _timestamp(self._clock())
        with self._connection() as connection:
            deleted = connection.executemany(
                "DELETE FROM threat_lookup WHERE provider = ? AND checked_at <= ?",
                ((provider.value, now - ttl.total_seconds()) for provider, ttl in self._ttl.items()),
            )
            return deleted.rowcount
