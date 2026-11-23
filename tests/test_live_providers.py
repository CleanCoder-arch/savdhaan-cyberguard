import asyncio
from datetime import datetime, timezone
from urllib.parse import parse_qs

import httpx
import pytest

from core.threat_intel import live_providers
from core.threat_intel.live_providers import (
    IntelProvider,
    IntelResult,
    IntelStatus,
    LookupTargets,
    UrlhausProvider,
    WhoisProvider,
    lookup_targets,
)


@pytest.mark.parametrize(
    ("url", "targets"),
    [
        (
            "https://Sub.Example.CO.IN.:8443/Agenda?token=sensitive#private",
            LookupTargets("example.co.in", "sub.example.co.in"),
        ),
        ("https://मदद.in/path", LookupTargets("xn--j2baw.in", "xn--j2baw.in")),
        ("https://straße.de/path", LookupTargets("xn--strae-oqa.de", "xn--strae-oqa.de")),
        ("https://strasse.de/path", LookupTargets("strasse.de", "strasse.de")),
        ("https://8.8.8.8/path", LookupTargets(urlhaus_host="8.8.8.8")),
        ("https://[2001:4860:4860::8888]/", LookupTargets()),
        ("https://127.0.0.1/", LookupTargets()),
        ("https://224.0.0.1/", LookupTargets()),
        ("https://localhost/", LookupTargets()),
        ("https://db.local/", LookupTargets()),
        ("https://demo.test/", LookupTargets()),
    ],
)
def test_public_lookup_targets(url, targets):
    assert lookup_targets(url) == targets


def test_lookup_rejects_url_credentials_before_egress():
    with pytest.raises(ValueError, match="credentials"):
        lookup_targets("https://name:secret@example.com/path?session=private")


@pytest.mark.parametrize(
    ("payload", "expected_status", "listed"),
    [
        ({"query_status": "no_results"}, IntelStatus.OK, False),
        (
            {"query_status": "ok", "urls": [{"url_status": "online"}], "url_count": "1"},
            IntelStatus.OK,
            True,
        ),
        (
            {"query_status": "ok", "urls": [{"url_status": "offline"}], "url_count": "1"},
            IntelStatus.OK,
            False,
        ),
        (
            {"query_status": "ok", "urls": [{"url_status": "offline"}], "url_count": "101"},
            IntelStatus.NO_DATA,
            None,
        ),
        (
            {"query_status": "ok", "urls": [{"url_status": "unknown"}], "url_count": "1"},
            IntelStatus.NO_DATA,
            None,
        ),
        ({"query_status": "unknown_auth_key"}, IntelStatus.AUTH_REQUIRED, None),
        ({"query_status": "invalid_host"}, IntelStatus.INVALID_RESPONSE, None),
        ({"query_status": []}, IntelStatus.INVALID_RESPONSE, None),
        ({"query_status": "ok", "urls": "bad"}, IntelStatus.INVALID_RESPONSE, None),
        ({"query_status": "ok", "urls": [{"url_status": {}}]}, IntelStatus.INVALID_RESPONSE, None),
    ],
)
def test_urlhaus_interprets_current_host_activity(payload, expected_status, listed):
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://urlhaus-api.abuse.ch/v1/host/"
        assert request.method == "POST"
        assert request.headers["auth-key"] == "authorized-key"
        assert parse_qs(request.content.decode()) == {"host": ["sub.example.co.in"]}
        return httpx.Response(200, json=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await UrlhausProvider("authorized-key", client=client).lookup("Sub.Example.Co.In.")

    result = asyncio.run(run())
    assert result.provider is IntelProvider.URLHAUS
    assert result.status is expected_status
    assert result.listed is listed
    assert result.checked_at.utcoffset() is not None
    assert "authorized-key" not in repr(result)


@pytest.mark.parametrize(
    ("status", "content", "expected_status"),
    [
        (429, b"", IntelStatus.RATE_LIMITED),
        (401, b"", IntelStatus.AUTH_REQUIRED),
        (403, b"", IntelStatus.AUTH_REQUIRED),
        (503, b"", IntelStatus.UNAVAILABLE),
        (200, b"not JSON", IntelStatus.INVALID_RESPONSE),
        (200, b"[]", IntelStatus.INVALID_RESPONSE),
    ],
)
def test_urlhaus_exposes_failed_requests_without_a_verdict(status, content, expected_status):
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(status, content=content))
        async with httpx.AsyncClient(transport=transport) as client:
            return await UrlhausProvider("authorized-key", client=client).lookup("example.com")

    result = asyncio.run(run())
    assert result.status is expected_status
    assert result.listed is None


def test_urlhaus_uses_configured_environment_key(monkeypatch):
    monkeypatch.setenv("URLHAUS_AUTH_KEY", "key-from-environment")

    def respond(request):
        assert request.headers["auth-key"] == "key-from-environment"
        return httpx.Response(200, json={"query_status": "no_results"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await UrlhausProvider(client=client).lookup("example.com")

    assert asyncio.run(run()).listed is False


def test_urlhaus_does_not_request_without_authentication_or_for_private_hosts(monkeypatch):
    monkeypatch.delenv("URLHAUS_AUTH_KEY", raising=False)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_http_request)) as client:
            return (
                await UrlhausProvider(client=client).lookup("example.com"),
                await UrlhausProvider("authorized-key", client=client).lookup("10.20.30.40"),
                await UrlhausProvider("authorized-key", client=client).lookup("secret.example.test"),
            )

    results = asyncio.run(run())
    assert tuple(result.status for result in results) == (
        IntelStatus.AUTH_REQUIRED,
        IntelStatus.NO_DATA,
        IntelStatus.NO_DATA,
    )


def _reject_http_request(request: httpx.Request):
    pytest.fail("The provider attempted an unexpected HTTP request.")


def test_urlhaus_does_not_follow_redirects_with_authentication():
    def respond(request):
        assert request.url.host == "urlhaus-api.abuse.ch"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
            return await UrlhausProvider("authorized-key", client=client).lookup("example.com")

    assert asyncio.run(run()).status is IntelStatus.UNAVAILABLE


def test_urlhaus_limited_response_bytes(monkeypatch):
    monkeypatch.setattr(live_providers, "_HTTP_RESPONSE_LIMIT", 16)

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 17))
        async with httpx.AsyncClient(transport=transport) as client:
            return await UrlhausProvider("authorized-key", client=client).lookup("example.com")

    assert asyncio.run(run()).status is IntelStatus.INVALID_RESPONSE


@pytest.mark.parametrize(
    ("error", "status"),
    [(httpx.ReadTimeout, IntelStatus.TIMEOUT), (httpx.ConnectError, IntelStatus.UNAVAILABLE)],
)
def test_urlhaus_distinguishes_timeout_and_network_error(error, status):
    def respond(request):
        raise error("sensitive details", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await UrlhausProvider("authorized-key", client=client).lookup("example.com")

    result = asyncio.run(run())
    assert result.status is status
    assert result.listed is None
    assert "sensitive" not in repr(result)


def test_urlhaus_enforces_total_deadline_even_if_transport_has_no_timeout():
    async def respond(request):
        await asyncio.Event().wait()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), timeout=None) as client:
            return await UrlhausProvider("authorized-key", timeout=0.02, client=client).lookup("example.com")

    assert asyncio.run(run()).status is IntelStatus.TIMEOUT


def test_whois_uses_trusted_port_43_with_registrable_domain(monkeypatch):
    payload = (
        b"Domain Name: EXAMPLE.CO.IN\r\nCreation Date: 2020-01-01T00:00:00Z\r\n"
        b"Registrant Email: private@example.com\r\nRegistrar WHOIS Server: localhost\r\n"
    )
    sent = []
    closed = []

    class Reader:
        def __init__(self):
            self.chunks = iter((payload[:40], payload[40:], b""))

        async def read(self, limit):
            return next(self.chunks)

    class Writer:
        def write(self, data):
            sent.append(data)

        async def drain(self):
            return None

        def close(self):
            closed.append(True)

    async def connect(server, port):
        assert (server, port) == ("whois.nixiregistry.in", 43)
        return Reader(), Writer()

    now = datetime(2026, 11, 24, tzinfo=timezone.utc)
    monkeypatch.setattr(live_providers.asyncio, "open_connection", connect)
    monkeypatch.setattr(live_providers, "_now", lambda: now)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_http_request)) as client:
            return await WhoisProvider(client=client).lookup("login.Example.Co.In.")

    result = asyncio.run(run())
    assert result == IntelResult(
        IntelProvider.WHOIS,
        IntelStatus.OK,
        now,
        domain_age_days=(now - datetime(2020, 1, 1, tzinfo=timezone.utc)).days,
    )
    assert sent == [b"example.co.in\r\n"]
    assert closed == [True]
    assert "private" not in repr(result)


def test_whois_rate_limit_is_not_retried(monkeypatch):
    async def query(server, domain):
        return "WHOIS LIMIT EXCEEDED"

    monkeypatch.setattr(live_providers, "_query_whois", query)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_http_request)) as client:
            return await WhoisProvider(client=client).lookup("example.in")

    assert asyncio.run(run()).status is IntelStatus.RATE_LIMITED


def test_whois_falls_back_to_authenticated_iana_registry_for_rdap(monkeypatch):
    async def query(server, domain):
        raise OSError("WHOIS port unavailable")

    monkeypatch.setattr(live_providers, "_query_whois", query)
    requests = []

    def respond(request):
        requests.append(str(request.url))
        if request.url.host == "data.iana.org":
            return httpx.Response(200, json={"services": [[["in"], [
                "http://rdap.example.com/",
                "https://rdap.example.com:444/invalid/",
                "https://rdap.example.com/rdap/",
            ]]]})
        return httpx.Response(
            200,
            json={
                "objectClassName": "domain",
                "ldhName": "EXAMPLE.IN",
                "events": [
                    {"eventAction": "last changed", "eventDate": "2026-11-23T05:20:32Z"},
                    {"eventAction": "registration", "eventDate": "2020-01-01T00:00:00Z"},
                ],
                "links": [{"href": "http://127.0.0.1/"}],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            provider = WhoisProvider(client=client)
            return await provider.lookup("login.example.in"), await provider.lookup("pay.example.in")

    first, second = asyncio.run(run())
    assert first.status is second.status is IntelStatus.OK
    assert first.domain_age_days == second.domain_age_days
    assert first.domain_age_days > 0
    assert requests == [
        "https://data.iana.org/rdap/dns.json",
        "https://rdap.example.com/rdap/domain/example.in",
        "https://rdap.example.com/rdap/domain/example.in",
    ]


@pytest.mark.parametrize(
    ("record", "status"),
    [
        (
            {"objectClassName": "domain", "ldhName": "example.io", "events": []},
            IntelStatus.NO_DATA,
        ),
        (
            {"objectClassName": "entity", "ldhName": "example.io"},
            IntelStatus.INVALID_RESPONSE,
        ),
        (
            {"objectClassName": "domain", "ldhName": "other.io"},
            IntelStatus.INVALID_RESPONSE,
        ),
        (
            {"objectClassName": "domain", "ldhName": "example.io", "events": [
                {"eventAction": "registration", "eventDate": "2999-01-01T00:00:00Z"},
            ]},
            IntelStatus.INVALID_RESPONSE,
        ),
        (
            {"objectClassName": "domain", "ldhName": "example.io", "events": [
                {"eventAction": "registration", "eventDate": []},
            ]},
            IntelStatus.INVALID_RESPONSE,
        ),
    ],
)
def test_rdap_validates_domain_and_registration_event(record, status):
    def respond(request):
        if request.url.host == "data.iana.org":
            return httpx.Response(200, json={"services": [[["io"], ["https://rdap.example.com/"]]]})
        return httpx.Response(200, json=record)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await WhoisProvider(client=client).lookup("example.io")

    result = asyncio.run(run())
    assert result.status is status
    assert result.domain_age_days is None


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "3"])
def test_providers_require_positive_finite_timeout(value):
    with pytest.raises(ValueError, match="timeout"):
        WhoisProvider(timeout=value)
    with pytest.raises(ValueError, match="timeout"):
        UrlhausProvider(timeout=value)


def test_intel_result_enforces_payload_for_provider_and_status():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="registration"):
        IntelResult(IntelProvider.WHOIS, IntelStatus.OK, now, listed=False)
    with pytest.raises(ValueError, match="URLhaus"):
        IntelResult(IntelProvider.URLHAUS, IntelStatus.OK, now)
    with pytest.raises(ValueError, match="verdict"):
        IntelResult(IntelProvider.URLHAUS, IntelStatus.TIMEOUT, now, listed=False)
    with pytest.raises(ValueError, match="timezone"):
        IntelResult(IntelProvider.URLHAUS, IntelStatus.OK, datetime.now(), listed=False)
