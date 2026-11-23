from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from core.threat_intel.feed_cache import FeedCache
from core.threat_intel.live_providers import IntelProvider, IntelResult, IntelStatus

_NOW = datetime(2026, 11, 24, 9, tzinfo=timezone.utc)
_HOST = "session-private.example.com"


def _whois(age: int = 10, checked_at: datetime = _NOW) -> IntelResult:
    return IntelResult(IntelProvider.WHOIS, IntelStatus.OK, checked_at, domain_age_days=age)


def _urlhaus(listed: bool = False) -> IntelResult:
    return IntelResult(IntelProvider.URLHAUS, IntelStatus.OK, _NOW, listed=listed)


def test_cache_persists_sanitized_outcomes_and_separates_providers(tmp_path) -> None:
    path = tmp_path / "lookup.sqlite3"
    cache = FeedCache(path, clock=lambda: _NOW)
    assert cache.get(IntelProvider.WHOIS, _HOST) is None
    cache.put(_HOST.upper() + ".", _whois())
    cache.put(_HOST, _urlhaus())

    reopened = FeedCache(path, clock=lambda: _NOW)
    assert reopened.get(IntelProvider.WHOIS, _HOST) == _whois()
    assert reopened.get(IntelProvider.URLHAUS, _HOST) == _urlhaus()
    assert reopened.get(IntelProvider.URLHAUS, "different.example.com") is None
    assert _HOST.encode() not in path.read_bytes().lower()


def test_cache_canonicalizes_idna_and_ip_addresses(tmp_path) -> None:
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: _NOW)
    cache.put("für.example.com", _whois())
    cache.put("2001:db8:0:0:0:0:0:1", _urlhaus(True))
    assert cache.get(IntelProvider.WHOIS, "xn--fr-xka.example.com") == _whois()
    assert cache.get(IntelProvider.URLHAUS, "2001:db8::1") == _urlhaus(True)


def test_cache_keeps_distinct_internationalized_hosts_separate(tmp_path) -> None:
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: _NOW)
    cache.put("straße.de", _urlhaus(True))
    assert cache.get(IntelProvider.URLHAUS, "xn--strae-oqa.de") == _urlhaus(True)
    assert cache.get(IntelProvider.URLHAUS, "strasse.de") is None


def test_freshness_tracks_observation_time_and_provider_ttl(tmp_path) -> None:
    now = _NOW
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: now)
    cache.put(_HOST, _whois())
    cache.put(_HOST, _urlhaus(True))

    now += timedelta(minutes=15)
    urlhaus = cache.lookup(IntelProvider.URLHAUS, _HOST)
    assert urlhaus is not None
    assert urlhaus.result.checked_at == _NOW
    assert urlhaus.expires_at == now
    assert not urlhaus.is_fresh
    assert cache.get(IntelProvider.URLHAUS, _HOST) is None
    assert cache.get(IntelProvider.WHOIS, _HOST) == _whois()

    assert cache.prune_expired() == 1
    assert cache.lookup(IntelProvider.URLHAUS, _HOST) is None
    now += timedelta(days=1)
    assert cache.prune_expired() == 1
    assert cache.get(IntelProvider.WHOIS, _HOST) is None


def test_rate_limited_response_does_not_poison_cache(tmp_path) -> None:
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: _NOW)
    cache.put(_HOST, _urlhaus(True))
    cache.put(_HOST, IntelResult(IntelProvider.URLHAUS, IntelStatus.RATE_LIMITED, _NOW))
    cache.put(_HOST, IntelResult(IntelProvider.WHOIS, IntelStatus.UNAVAILABLE, _NOW))
    assert cache.get(IntelProvider.URLHAUS, _HOST) == _urlhaus(True)
    assert cache.get(IntelProvider.WHOIS, _HOST) is None


def test_older_observations_cannot_overwrite_new_ones(tmp_path) -> None:
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: _NOW)
    older = _whois(1, _NOW - timedelta(hours=1))
    newest = _whois(20)
    with ThreadPoolExecutor(2) as threads:
        list(threads.map(lambda result: cache.put(_HOST, result), (newest, older)))
    assert cache.get(IntelProvider.WHOIS, _HOST) == newest


@pytest.mark.parametrize(
    "key", ["", "https://example.com/", "example.com:443", "example.com?token=private", "me@example.com",
            "fe80::1%zone", "example.com\n", "example.com/folder"]
)
def test_cache_rejects_raw_urls_and_invalid_host_keys(tmp_path, key) -> None:
    cache = FeedCache(tmp_path / "lookup.db", clock=lambda: _NOW)
    with pytest.raises(ValueError):
        cache.put(key, _whois())


def test_cache_rejects_invalid_expiry_policy(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        FeedCache(tmp_path / "lookup.db", urlhaus_ttl=timedelta(0))
