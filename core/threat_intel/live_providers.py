"""Check public registration age and URLhaus host records without opening scan URLs."""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from ipaddress import IPv4Address, ip_address
from math import isfinite
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

import httpx
from tldextract import TLDExtract

from core.validators import validate_url

_PUBLIC_SUFFIX = TLDExtract(cache_dir=None, suffix_list_urls=())
_DNS_HOST = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_URLHAUS_HOST_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/host/"
_IANA_RDAP_REGISTRY = "https://data.iana.org/rdap/dns.json"
# IANA Root Zone Database records for these TLDs advertise port-43 service.
_WHOIS_REGISTRIES = {
    "com": "whois.verisign-grs.com",
    "net": "whois.verisign-grs.com",
    "in": "whois.nixiregistry.in",
    "org": "whois.publicinterestregistry.org",
}
_WHOIS_CREATION_KEYS = {
    "created", "created on", "creation date", "domain created", "registered", "registered on",
    "registration date",
}
_WHOIS_RESPONSE_LIMIT = 256 * 1024
_HTTP_RESPONSE_LIMIT = 2 * 1024 * 1024
_RDAP_REGISTRY_TTL = 24 * 60 * 60


class IntelProvider(StrEnum):
    WHOIS = "whois"
    URLHAUS = "urlhaus"


class IntelStatus(StrEnum):
    OK = "ok"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTH_REQUIRED = "auth_required"
    NO_DATA = "no_data"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True, slots=True)
class IntelResult:
    """Sanitized provider output. A successful check has exactly one typed value."""

    provider: IntelProvider
    status: IntelStatus
    checked_at: datetime
    domain_age_days: int | None = None
    listed: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, IntelProvider) or not isinstance(self.status, IntelStatus):
            raise ValueError("A known threat-intelligence provider and status are required.")
        if not isinstance(self.checked_at, datetime) or self.checked_at.utcoffset() is None:
            raise ValueError("The check timestamp must be timezone-aware.")
        if self.status is not IntelStatus.OK:
            if self.domain_age_days is not None or self.listed is not None:
                raise ValueError("An incomplete check cannot carry a verdict.")
        elif self.provider is IntelProvider.WHOIS:
            if type(self.domain_age_days) is not int or self.domain_age_days < 0 or self.listed is not None:
                raise ValueError("A domain registration result requires a nonnegative age.")
        elif type(self.listed) is not bool or self.domain_age_days is not None:
            raise ValueError("A URLhaus result requires a listing decision.")


@dataclass(frozen=True, slots=True)
class LookupTargets:
    registrable_domain: str | None = None
    urlhaus_host: str | None = None


def _targets_for_host(host: str) -> LookupTargets:
    try:
        canonical = httpx.URL(scheme="https", host=host).raw_host.decode("ascii").removesuffix(".")
    except (UnicodeError, httpx.InvalidURL):
        return LookupTargets()
    try:
        address = ip_address(canonical)
    except ValueError:
        if len(canonical) > 253 or not _DNS_HOST.fullmatch(canonical):
            return LookupTargets()
        extracted = _PUBLIC_SUFFIX(canonical)
        registered = extracted.top_domain_under_public_suffix
        # A public suffix and a registrable label are required before disclosing a DNS name.
        if not registered or any(label.startswith("-") or label.endswith("-") for label in canonical.split(".")):
            return LookupTargets()
        return LookupTargets(registered, canonical)
    if isinstance(address, IPv4Address) and address.is_global and not address.is_multicast:
        return LookupTargets(urlhaus_host=str(address))
    return LookupTargets()


def lookup_targets(url: str) -> LookupTargets:
    """Choose public lookup keys without carrying credentials, paths, or query fields."""
    host = urlsplit(validate_url(url)).hostname
    return _targets_for_host(host) if host else LookupTargets()


class _ProviderFailure(Exception):
    def __init__(self, status: IntelStatus) -> None:
        super().__init__(status.value)
        self.status = status


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _valid_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError("A positive finite threat-intelligence timeout is required.")
    return float(timeout)


@asynccontextmanager
async def _http_client(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient() as managed:
            yield managed


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    data: dict[str, str] | None = None,
    auth_key: str | None = None,
) -> dict:
    headers = {"Accept": "application/rdap+json, application/json", "Accept-Encoding": "identity"}
    if auth_key is not None:
        headers["Auth-Key"] = auth_key
    async with client.stream(method, url, data=data, headers=headers, follow_redirects=False) as response:
        if response.status_code == 429:
            raise _ProviderFailure(IntelStatus.RATE_LIMITED)
        if response.status_code == 404 and method == "GET":
            raise _ProviderFailure(IntelStatus.NO_DATA)
        if response.status_code in {401, 403} and auth_key is not None:
            raise _ProviderFailure(IntelStatus.AUTH_REQUIRED)
        if not response.is_success:
            raise _ProviderFailure(IntelStatus.UNAVAILABLE)
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)

        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > _HTTP_RESPONSE_LIMIT:
                raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
            body.extend(chunk)
    try:
        document = json.loads(body)
    except (ValueError, RecursionError):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE) from None
    if not isinstance(document, dict):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    return document


def _urlhaus_listing(response: dict) -> bool:
    status = response.get("query_status")
    if status == "no_results":
        return False
    if status in ("no_auth_key", "unknown_auth_key", "invalid_auth_key"):
        raise _ProviderFailure(IntelStatus.AUTH_REQUIRED)
    if status != "ok":
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)

    entries = response.get("urls")
    if not isinstance(entries, list) or not entries:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    states = [entry.get("url_status") if isinstance(entry, dict) else None for entry in entries]
    if any(state not in ("online", "offline", "unknown") for state in states):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    if "online" in states:
        return True
    total = response.get("url_count")
    if isinstance(total, str) and re.fullmatch(r"[0-9]{1,12}", total):
        total = int(total)
    if type(total) is not int or total < len(states):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    if "unknown" in states or total > len(states):
        raise _ProviderFailure(IntelStatus.NO_DATA)
    return False


class UrlhausProvider:
    """Look up active URLhaus entries on a public hostname or IPv4 address."""

    provider = IntelProvider.URLHAUS

    def __init__(
        self, auth_key: str | None = None, *, timeout: float = 3.0, client: httpx.AsyncClient | None = None
    ) -> None:
        self._auth_key = auth_key if auth_key is not None else os.environ.get("URLHAUS_AUTH_KEY")
        self._timeout = _valid_timeout(timeout)
        self._client = client

    async def lookup(self, target: str) -> IntelResult:
        host = _targets_for_host(target).urlhaus_host
        if host is None:
            return IntelResult(self.provider, IntelStatus.NO_DATA, _now())
        key = self._auth_key
        if not key or not key.isascii() or not key.isprintable() or key.strip() != key:
            return IntelResult(self.provider, IntelStatus.AUTH_REQUIRED, _now())

        try:
            async with asyncio.timeout(self._timeout), _http_client(self._client) as client:
                response = await _request_json(
                    client, "POST", _URLHAUS_HOST_ENDPOINT, data={"host": host}, auth_key=key
                )
                listed = _urlhaus_listing(response)
        except (TimeoutError, httpx.TimeoutException):
            return IntelResult(self.provider, IntelStatus.TIMEOUT, _now())
        except (OSError, httpx.RequestError):
            return IntelResult(self.provider, IntelStatus.UNAVAILABLE, _now())
        except _ProviderFailure as failure:
            return IntelResult(self.provider, failure.status, _now())
        return IntelResult(self.provider, IntelStatus.OK, _now(), listed=listed)


async def _query_whois(server: str, domain: str) -> str:
    reader, writer = await asyncio.open_connection(server, 43)
    try:
        writer.write(f"{domain}\r\n".encode("ascii"))
        await writer.drain()
        body = bytearray()
        while chunk := await reader.read(8 * 1024):
            if len(body) + len(chunk) > _WHOIS_RESPONSE_LIMIT:
                raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
            body.extend(chunk)
        return body.decode("utf-8", errors="replace")
    finally:
        writer.close()


def _whois_creation(response: str, domain: str) -> str:
    lowered = response.lower()
    if any(
        phrase in lowered
        for phrase in ("limit exceeded", "quota exceeded", "too many queries", "rate limit exceeded")
    ):
        raise _ProviderFailure(IntelStatus.RATE_LIMITED)
    created = []
    for line in response.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "domain name" and value.rstrip(".").lower() != domain:
            raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
        if key in _WHOIS_CREATION_KEYS:
            created.append(value)
    if not created:
        raise _ProviderFailure(IntelStatus.NO_DATA)
    if len(set(created)) != 1:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    return created[0]


def _rdap_creation(response: dict, domain: str) -> str:
    name = response.get("ldhName")
    if response.get("objectClassName") != "domain" or not isinstance(name, str) or name.lower() != domain:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    events = response.get("events", [])
    if not isinstance(events, list):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    dates = [
        event.get("eventDate")
        for event in events
        if isinstance(event, dict) and event.get("eventAction") == "registration"
    ]
    if not dates:
        raise _ProviderFailure(IntelStatus.NO_DATA)
    if any(not isinstance(date, str) for date in dates) or len(set(dates)) != 1:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    return dates[0]


def _age_days(value: str, now: datetime) -> int:
    try:
        created = datetime.fromisoformat(value)
    except ValueError:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE) from None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    elapsed = now - created
    if elapsed.days < 0:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    return elapsed.days


def _rdap_endpoints(registry: dict) -> dict[str, str]:
    entries = registry.get("services")
    if not isinstance(entries, list):
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    endpoints: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        suffixes, urls = entry
        if not isinstance(suffixes, list) or not isinstance(urls, list):
            continue
        for url in urls:
            if not isinstance(url, str) or not url.startswith("https://"):
                continue
            try:
                parts = urlsplit(validate_url(url))
                if parts.port not in (443, None) or parts.query or parts.fragment:
                    continue
            except ValueError:
                continue
            host = parts.hostname
            if not host or _targets_for_host(host).registrable_domain is None:
                continue
            for suffix in suffixes:
                if isinstance(suffix, str):
                    endpoints[suffix.lower()] = url.rstrip("/") + "/"
            break
    if not endpoints:
        raise _ProviderFailure(IntelStatus.INVALID_RESPONSE)
    return endpoints


class WhoisProvider:
    """Retrieve a registration date from trusted WHOIS or official registry RDAP."""

    provider = IntelProvider.WHOIS

    def __init__(self, *, timeout: float = 3.0, client: httpx.AsyncClient | None = None) -> None:
        self._timeout = _valid_timeout(timeout)
        self._client = client
        self._rdap_registry: tuple[float, dict[str, str]] = (0.0, {})
        self._loop_locks: WeakValueDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakValueDictionary()
        self._locks_guard = Lock()

    async def lookup(self, target: str) -> IntelResult:
        domain = _targets_for_host(target).registrable_domain
        if domain is None:
            return IntelResult(self.provider, IntelStatus.NO_DATA, _now())
        try:
            async with asyncio.timeout(self._timeout):
                return await self._registration_age(domain)
        except (TimeoutError, httpx.TimeoutException):
            return IntelResult(self.provider, IntelStatus.TIMEOUT, _now())
        except (OSError, httpx.RequestError):
            return IntelResult(self.provider, IntelStatus.UNAVAILABLE, _now())
        except _ProviderFailure as failure:
            return IntelResult(self.provider, failure.status, _now())

    async def _registration_age(self, domain: str) -> IntelResult:
        tld = domain.rsplit(".", 1)[-1]
        whois_server = _WHOIS_REGISTRIES.get(tld)
        if whois_server is not None:
            try:
                async with asyncio.timeout(min(1.0, self._timeout / 2)):
                    response = await _query_whois(whois_server, domain)
                    checked_at = _now()
                    age = _age_days(_whois_creation(response, domain), checked_at)
                    return IntelResult(self.provider, IntelStatus.OK, checked_at, domain_age_days=age)
            except (OSError, _ProviderFailure) as exc:
                if isinstance(exc, _ProviderFailure) and exc.status is IntelStatus.RATE_LIMITED:
                    raise
                # An unavailable port-43 result can still be available from the official RDAP server.

        async with _http_client(self._client) as client:
            base_url = await self._rdap_base_url(client, tld)
            response = await _request_json(client, "GET", f"{base_url}domain/{domain}")
        checked_at = _now()
        age = _age_days(_rdap_creation(response, domain), checked_at)
        return IntelResult(self.provider, IntelStatus.OK, checked_at, domain_age_days=age)

    async def _rdap_base_url(self, client: httpx.AsyncClient, tld: str) -> str:
        loop = asyncio.get_running_loop()
        with self._locks_guard:
            lock = self._loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._loop_locks[loop] = lock
        async with lock:
            expires, endpoints = self._rdap_registry
            if monotonic() >= expires:
                registry = await _request_json(client, "GET", _IANA_RDAP_REGISTRY)
                endpoints = _rdap_endpoints(registry)
                self._rdap_registry = (monotonic() + _RDAP_REGISTRY_TTL, endpoints)
            base_url = endpoints.get(tld)
        if base_url is None:
            raise _ProviderFailure(IntelStatus.NO_DATA)
        return base_url
