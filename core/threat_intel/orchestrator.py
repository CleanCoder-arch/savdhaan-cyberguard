"""Run independent threat-intelligence checks with bounded concurrency."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from threading import Lock
from typing import Protocol
from weakref import WeakValueDictionary

from core.schemas import ScanIssue
from core.threat_intel.live_providers import (
    IntelProvider,
    IntelResult,
    IntelStatus,
    UrlhausProvider,
    WhoisProvider,
    lookup_targets,
)


class IntelCache(Protocol):
    def get(self, provider: IntelProvider, host: str) -> IntelResult | None: ...

    def put(self, host: str, result: IntelResult) -> None: ...


class Provider(Protocol):
    async def lookup(self, target: str) -> IntelResult: ...


@dataclass(frozen=True, slots=True)
class IntelReport:
    """Independent successful results and explicit reasons for incomplete checks."""

    results: tuple[IntelResult, ...]
    cache_issues: tuple[ScanIssue, ...] = ()

    @property
    def issues(self) -> tuple[ScanIssue, ...]:
        return tuple(
            ScanIssue(component=result.provider.value, code=result.status.value)
            for result in self.results
            if result.status is not IntelStatus.OK
        ) + self.cache_issues

    @property
    def is_partial(self) -> bool:
        return bool(self.cache_issues) or any(result.status is not IntelStatus.OK for result in self.results)

    @property
    def freshness(self) -> datetime | None:
        """Return the oldest successful lookup timestamp, including cache hits."""
        return min(
            (result.checked_at for result in self.results if result.status is IntelStatus.OK),
            default=None,
        )


class ThreatIntelRunner:
    """Look up public targets with a concurrency limit on each active event loop."""

    def __init__(
        self,
        *,
        cache: IntelCache | None = None,
        whois: Provider | None = None,
        urlhaus: Provider | None = None,
        timeout: float = 3.0,
        max_concurrent_checks: int = 8,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
            raise ValueError("A positive finite threat-intelligence timeout is required.")
        if type(max_concurrent_checks) is not int or max_concurrent_checks < 1:
            raise ValueError("The concurrency limit must be a positive integer.")
        self._cache = cache
        self._timeout = timeout
        self._max_concurrent_checks = max_concurrent_checks
        self._loop_slots: WeakValueDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakValueDictionary()
        self._slots_lock = Lock()
        self._whois = whois if whois is not None else WhoisProvider(timeout=timeout)
        self._urlhaus = urlhaus if urlhaus is not None else UrlhausProvider(timeout=timeout)

    async def run(self, url: str, *, offline: bool = False) -> IntelReport:
        """Use fresh cached results first, then query the live services in parallel."""
        targets = lookup_targets(url)
        loop = asyncio.get_running_loop()
        with self._slots_lock:
            slots = self._loop_slots.get(loop)
            if slots is None:
                slots = asyncio.Semaphore(self._max_concurrent_checks)
                self._loop_slots[loop] = slots
        outcomes = await asyncio.gather(
            self._run_provider(IntelProvider.WHOIS, self._whois, targets.registrable_domain, offline, slots),
            self._run_provider(IntelProvider.URLHAUS, self._urlhaus, targets.urlhaus_host, offline, slots),
        )
        results = tuple(result for result, _ in outcomes)
        cache_codes = dict.fromkeys(code for _, codes in outcomes for code in codes)
        cache_issues = tuple(ScanIssue(component="threat_intel", code=code) for code in cache_codes)
        return IntelReport(results, cache_issues)

    async def _run_provider(
        self,
        provider: IntelProvider,
        lookup: Provider,
        target: str | None,
        offline: bool,
        slots: asyncio.Semaphore,
    ) -> tuple[IntelResult, list[str]]:
        cache_codes: list[str] = []
        if target is None:
            return _incomplete(provider, IntelStatus.NO_DATA), cache_codes

        cache_operation = None
        result = None
        try:
            async with asyncio.timeout(self._timeout), slots:
                if self._cache is not None:
                    cache_operation = "cache_read"
                    try:
                        cached = await asyncio.to_thread(self._cache.get, provider, target)
                        if cached is not None:
                            if not self._is_valid(cached, provider) or cached.status is not IntelStatus.OK:
                                raise ValueError("The cache returned an invalid provider result.")
                            return cached, cache_codes
                    except Exception:
                        cache_codes.append(cache_operation)
                    cache_operation = None
                if offline:
                    return _incomplete(provider, IntelStatus.OFFLINE), cache_codes

                try:
                    result = await lookup.lookup(target)
                except TimeoutError:
                    result = _incomplete(provider, IntelStatus.TIMEOUT)
                except Exception:
                    # Report unknown provider failures without exposing details or suppressing cancellation.
                    result = _incomplete(provider, IntelStatus.UNAVAILABLE)
                if not self._is_valid(result, provider):
                    result = _incomplete(provider, IntelStatus.INVALID_RESPONSE)

                if self._cache is not None and result.status is IntelStatus.OK:
                    cache_operation = "cache_write"
                    try:
                        await asyncio.to_thread(self._cache.put, target, result)
                    except Exception:
                        cache_codes.append(cache_operation)
        except TimeoutError:
            if cache_operation is not None:
                cache_codes.append(cache_operation)
            if result is None:
                result = _incomplete(provider, IntelStatus.TIMEOUT)
        return result, cache_codes

    @staticmethod
    def _is_valid(result: object, provider: IntelProvider) -> bool:
        return isinstance(result, IntelResult) and result.provider is provider


def _incomplete(provider: IntelProvider, status: IntelStatus) -> IntelResult:
    return IntelResult(provider, status, datetime.now(timezone.utc))
