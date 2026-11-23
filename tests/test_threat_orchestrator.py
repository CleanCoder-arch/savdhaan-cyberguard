import asyncio
from datetime import datetime, timezone
from threading import Event

import pytest

from core.schemas import ScanIssue
from core.threat_intel.live_providers import IntelProvider, IntelResult, IntelStatus
from core.threat_intel.orchestrator import ThreatIntelRunner

_CHECKED_AT = datetime(2026, 10, 10, tzinfo=timezone.utc)
_AGE = IntelResult(IntelProvider.WHOIS, IntelStatus.OK, _CHECKED_AT, domain_age_days=180)
_LISTED = IntelResult(IntelProvider.URLHAUS, IntelStatus.OK, _CHECKED_AT, listed=True)
_RATE_LIMITED = IntelResult(IntelProvider.URLHAUS, IntelStatus.RATE_LIMITED, _CHECKED_AT)
_SCAN_URL = "https://Login.Example.CO.IN:443/Agenda?access_token=private#other"


class _Provider:
    def __init__(self, result):
        self.result = result
        self.targets = []

    async def lookup(self, target):
        self.targets.append(target)
        return self.result


class _Cache:
    def __init__(self, entries=None):
        self.entries = entries or {}
        self.requests = []
        self.writes = []

    def get(self, provider, host):
        self.requests.append((provider, host))
        return self.entries.get((provider, host))

    def put(self, host, result):
        self.writes.append((host, result))
        self.entries[result.provider, host] = result


def test_runner_calls_both_providers_in_parallel_with_only_public_lookup_keys():
    async def run():
        whois_ready = asyncio.Event()
        urlhaus_ready = asyncio.Event()

        class Whois:
            async def lookup(self, target):
                assert target == "example.co.in"
                whois_ready.set()
                await urlhaus_ready.wait()
                return _AGE

        class Urlhaus:
            async def lookup(self, target):
                assert target == "login.example.co.in"
                urlhaus_ready.set()
                await whois_ready.wait()
                return _LISTED

        return await ThreatIntelRunner(whois=Whois(), urlhaus=Urlhaus(), timeout=1).run(_SCAN_URL)

    report = asyncio.run(run())
    assert report.results == (_AGE, _LISTED)
    assert report.issues == ()
    assert report.freshness == _CHECKED_AT
    assert not report.is_partial
    assert "access_token" not in repr(report)


def test_runner_preserves_success_when_provider_is_rate_limited():
    cache = _Cache()
    runner = ThreatIntelRunner(cache=cache, whois=_Provider(_AGE), urlhaus=_Provider(_RATE_LIMITED))
    report = asyncio.run(runner.run(_SCAN_URL))
    assert report.results == (_AGE, _RATE_LIMITED)
    assert report.issues == (ScanIssue(component="urlhaus", code="rate_limited"),)
    assert report.is_partial
    assert report.freshness == _CHECKED_AT
    assert cache.writes == [("example.co.in", _AGE)]


def test_runner_reports_timeout_and_cancels_slow_lookup():
    async def run():
        stopped = asyncio.Event()

        class Slow:
            async def lookup(self, target):
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        runner = ThreatIntelRunner(whois=Slow(), urlhaus=_Provider(_LISTED), timeout=0.03)
        report = await runner.run(_SCAN_URL)
        assert stopped.is_set()
        return report

    report = asyncio.run(run())
    assert report.results[0].status is IntelStatus.TIMEOUT
    assert report.results[1] is _LISTED
    assert report.issues == (ScanIssue(component="whois", code="timeout"),)


def test_runner_cancellation_propagates_and_cancels_both_providers():
    async def run():
        ready = asyncio.Event()
        callers = 0
        cancelled = 0

        class Pending:
            async def lookup(self, target):
                nonlocal callers, cancelled
                callers += 1
                if callers == 2:
                    ready.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled += 1

        runner = ThreatIntelRunner(whois=Pending(), urlhaus=Pending())
        task = asyncio.create_task(runner.run(_SCAN_URL))
        await asyncio.wait_for(ready.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == 2

    asyncio.run(run())


def test_runner_limits_lookups_across_simultaneous_scans():
    async def run():
        active = 0
        peak = 0

        class Counting:
            def __init__(self, result):
                self.result = result

            async def lookup(self, target):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return self.result

        runner = ThreatIntelRunner(
            whois=Counting(_AGE), urlhaus=Counting(_LISTED), max_concurrent_checks=2
        )
        reports = await asyncio.gather(*(runner.run(_SCAN_URL) for _ in range(5)))
        assert peak == 2
        assert all(not report.is_partial for report in reports)

    asyncio.run(run())


def test_runner_reuses_instance_across_event_loops_after_slot_contention():
    class Deliberate:
        def __init__(self, result):
            self.result = result

        async def lookup(self, target):
            await asyncio.sleep(0.01)
            return self.result

    runner = ThreatIntelRunner(
        whois=Deliberate(_AGE), urlhaus=Deliberate(_LISTED), max_concurrent_checks=1
    )
    assert asyncio.run(runner.run(_SCAN_URL)).results == (_AGE, _LISTED)
    assert asyncio.run(runner.run(_SCAN_URL)).results == (_AGE, _LISTED)


def test_blocked_cache_read_does_not_delay_the_other_provider():
    async def run():
        loop = asyncio.get_running_loop()
        cache_waiting = asyncio.Event()
        release_cache = Event()

        class DelayedCache(_Cache):
            def get(self, provider, host):
                if provider is IntelProvider.WHOIS:
                    loop.call_soon_threadsafe(cache_waiting.set)
                    if not release_cache.wait(timeout=1):
                        raise TimeoutError("The event loop could not run the independent check.")
                return super().get(provider, host)

        class Independent:
            async def lookup(self, target):
                await cache_waiting.wait()
                release_cache.set()
                return _LISTED

        runner = ThreatIntelRunner(cache=DelayedCache(), whois=_Provider(_AGE), urlhaus=Independent())
        return await runner.run(_SCAN_URL)

    report = asyncio.run(run())
    assert report.results == (_AGE, _LISTED)
    assert report.issues == ()


def test_cache_write_deadline_preserves_completed_provider_result():
    async def run():
        release_cache = Event()

        class DelayedCache(_Cache):
            def put(self, host, result):
                if result.provider is IntelProvider.WHOIS:
                    release_cache.wait(timeout=1)
                super().put(host, result)

        runner = ThreatIntelRunner(
            cache=DelayedCache(), whois=_Provider(_AGE), urlhaus=_Provider(_LISTED), timeout=0.1
        )
        try:
            return await runner.run(_SCAN_URL)
        finally:
            release_cache.set()

    report = asyncio.run(run())
    assert report.results == (_AGE, _LISTED)
    assert report.issues == (ScanIssue(component="threat_intel", code="cache_write"),)
    assert report.is_partial


def test_offline_cache_hit_is_used_even_when_other_check_is_unavailable():
    cache = _Cache({(IntelProvider.WHOIS, "example.co.in"): _AGE})
    whois, urlhaus = _Provider(_AGE), _Provider(_LISTED)
    runner = ThreatIntelRunner(whois=whois, urlhaus=urlhaus, cache=cache)
    report = asyncio.run(runner.run(_SCAN_URL, offline=True))
    assert report.results[0] is _AGE
    assert report.results[1].status is IntelStatus.OFFLINE
    assert report.issues == (ScanIssue(component="urlhaus", code="offline"),)
    assert report.freshness == _CHECKED_AT
    assert whois.targets == urlhaus.targets == cache.writes == []
    assert set(cache.requests) == {
        (IntelProvider.WHOIS, "example.co.in"),
        (IntelProvider.URLHAUS, "login.example.co.in"),
    }


def test_offline_without_any_cache_does_not_call_live_providers():
    whois, urlhaus = _Provider(_AGE), _Provider(_LISTED)
    runner = ThreatIntelRunner(whois=whois, urlhaus=urlhaus)
    report = asyncio.run(runner.run(_SCAN_URL, offline=True))
    assert tuple(result.status for result in report.results) == (IntelStatus.OFFLINE, IntelStatus.OFFLINE)
    assert report.is_partial
    assert report.freshness is None
    assert whois.targets == urlhaus.targets == []


@pytest.mark.parametrize("url", ["https://127.0.0.1/secret", "https://demo.test/?session=secret"])
def test_runner_avoids_cache_and_network_for_nonpublic_targets(url):
    whois, urlhaus, cache = _Provider(_AGE), _Provider(_LISTED), _Cache()
    runner = ThreatIntelRunner(whois=whois, urlhaus=urlhaus, cache=cache)
    report = asyncio.run(runner.run(url))
    assert tuple(result.status for result in report.results) == (IntelStatus.NO_DATA, IntelStatus.NO_DATA)
    assert whois.targets == urlhaus.targets == cache.requests == []


def test_public_ipv4_checks_urlhaus_only():
    whois, urlhaus = _Provider(_AGE), _Provider(_LISTED)
    runner = ThreatIntelRunner(whois=whois, urlhaus=urlhaus)
    report = asyncio.run(runner.run("https://8.8.8.8/private?key=secret"))
    assert report.results[0].status is IntelStatus.NO_DATA
    assert report.results[1] is _LISTED
    assert whois.targets == []
    assert urlhaus.targets == ["8.8.8.8"]


def test_invalid_provider_response_becomes_partial_result():
    runner = ThreatIntelRunner(whois=_Provider(_LISTED), urlhaus=_Provider(None))
    report = asyncio.run(runner.run(_SCAN_URL))
    assert tuple(result.status for result in report.results) == (
        IntelStatus.INVALID_RESPONSE,
        IntelStatus.INVALID_RESPONSE,
    )
    assert report.freshness is None


def test_provider_exception_preserves_other_result_without_exposing_exception_details():
    class Broken:
        async def lookup(self, target):
            raise RuntimeError("private payload")

    runner = ThreatIntelRunner(whois=_Provider(_AGE), urlhaus=Broken())
    report = asyncio.run(runner.run(_SCAN_URL))
    assert report.results[0] is _AGE
    assert report.issues == (ScanIssue(component="urlhaus", code="unavailable"),)
    assert "private" not in repr(report)


def test_cache_failures_are_reported_once_and_do_not_replace_provider_results():
    class BrokenCache:
        def get(self, provider, host):
            raise RuntimeError("cache read failure")

        def put(self, host, result):
            raise RuntimeError("cache write failure")

    runner = ThreatIntelRunner(cache=BrokenCache(), whois=_Provider(_AGE), urlhaus=_Provider(_LISTED))
    report = asyncio.run(runner.run(_SCAN_URL))
    assert report.results == (_AGE, _LISTED)
    assert report.issues == (
        ScanIssue(component="threat_intel", code="cache_read"),
        ScanIssue(component="threat_intel", code="cache_write"),
    )
    assert report.is_partial


def test_cache_hit_retains_timestamp_and_report_freshness_uses_oldest_success():
    result = IntelResult(
        IntelProvider.URLHAUS,
        IntelStatus.OK,
        datetime(2026, 10, 11, tzinfo=timezone.utc),
        listed=False,
    )
    cache = _Cache({(IntelProvider.WHOIS, "example.co.in"): _AGE})
    whois, urlhaus = _Provider(_AGE), _Provider(result)
    runner = ThreatIntelRunner(cache=cache, whois=whois, urlhaus=urlhaus)
    report = asyncio.run(runner.run(_SCAN_URL))
    assert report.results == (_AGE, result)
    assert report.freshness == _AGE.checked_at
    assert not report.is_partial
    assert whois.targets == []
    assert urlhaus.targets == ["login.example.co.in"]
    assert cache.writes == [("login.example.co.in", result)]


@pytest.mark.parametrize("value", [0, True, -1, 1.5, "2"])
def test_runner_requires_positive_integer_concurrency(value):
    with pytest.raises(ValueError, match="concurrency"):
        ThreatIntelRunner(max_concurrent_checks=value)


@pytest.mark.parametrize("value", [0, True, -1, float("nan"), "2"])
def test_runner_requires_positive_finite_timeout(value):
    with pytest.raises(ValueError, match="timeout"):
        ThreatIntelRunner(timeout=value)
