import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.auth import APIKeys, ControlStore
from api.main import APISettings, create_app
from api.sse import scan_events
from core.schemas import LinkReport, ScanIssue, ScanReport, ScanRequest
from core.storage.repository import ScanRepository


class RecordingEngine:
    readiness = {"url_model": True, "sms_model": True}

    def __init__(self) -> None:
        self.requests: list[tuple[ScanRequest, bool]] = []
        self.fail = False

    async def scan(self, request: ScanRequest, *, offline: bool = False) -> ScanReport:
        self.requests.append((request, offline))
        if self.fail:
            raise RuntimeError("private engine details")
        return ScanReport(
            scan_type=request.scan_type, risk_level="MEDIUM", risk_score=45,
            status="partial", issues=(ScanIssue(component="urlhaus", code="rate_limited"),),
            links=(LinkReport(url=request.url),) if request.url else (),
        )


@pytest.fixture
def settings(tmp_path) -> APISettings:
    return APISettings(
        api_keys=("TEST_ONLY_private_key", "TEST_ONLY_secondary_key"),
        db_path=tmp_path / "scans.sqlite3", control_db_path=tmp_path / "control.sqlite3", model_dir=tmp_path,
        requests_per_window=3,
    )


@pytest.fixture
def service(settings):
    engine = RecordingEngine()
    repository = ScanRepository(settings.db_path)
    with TestClient(create_app(settings, engine=engine, repository=repository)) as client:
        yield client, engine, repository


def _auth(settings: APISettings, index: int = 0) -> dict[str, str]:
    return {"X-API-Key": settings.api_keys[index]}


def test_health_and_openapi_report_real_service_configuration(service) -> None:
    client, _, _ = service
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {
            "url_model": True, "sms_model": True, "storage": True, "rate_limiting": True,
            "scan_api_key": True, "whatsapp_webhook": False, "whatsapp_media": False,
        },
    }
    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/api/v1/scan"]["post"]
    assert endpoint["requestBody"]["content"]["application/json"]["schema"]["properties"]["scan_type"]["$ref"] == (
        "#/components/schemas/ScanType"
    )
    assert "ScanType" in schema["components"]["schemas"]
    assert endpoint["security"] == [{"APIKeyHeader": []}]
    assert "text/event-stream" in endpoint["responses"]["200"]["content"]


def test_health_checks_actual_database_access(settings) -> None:
    settings.db_path.mkdir()
    with TestClient(create_app(settings, engine=RecordingEngine())) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        assert response.json()["checks"]["storage"] is False


def test_scan_api_fails_closed_without_configured_key(settings) -> None:
    engine = RecordingEngine()
    with TestClient(create_app(replace(settings, api_keys=()), engine=engine)) as client:
        response = client.post("/api/v1/scan", headers=_auth(settings), json={"scan_type": "sms", "text": "Private"})
        assert response.status_code == 503
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert engine.requests == []


def test_unauthorized_calls_cannot_scan_or_echo_key(service) -> None:
    client, engine, _ = service
    for headers in ({}, {"X-API-Key": "secret_wrong_value"}):
        response = client.post("/api/v1/scan", headers=headers, json={"scan_type": "sms", "text": "Private"})
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "APIKey"
        assert "secret_wrong_value" not in response.text
    assert engine.requests == []


def test_scan_delegates_preserves_partial_and_saves_metadata(service, settings) -> None:
    client, engine, repository = service
    value = "https://example.test/Agenda?token=private-request-marker"
    response = client.post("/api/v1/scan?offline=true", headers=_auth(settings), json={"scan_type": "url", "url": value})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-RateLimit-Remaining"] == "2"
    report = ScanReport.model_validate_json(response.content)
    assert report.links[0].url == value
    assert report.issues[0].code == "rate_limited"
    assert engine.requests == [(ScanRequest(scan_type="url", url=value), True)]
    metadata = repository.get_scan(report.scan_id)
    assert metadata is not None
    assert metadata.risk_score == 45
    assert b"private-request-marker" not in settings.db_path.read_bytes()
    assert settings.api_keys[0].encode() not in settings.control_db_path.read_bytes()


def test_scan_accepts_base64_images(service, settings) -> None:
    client, engine, _ = service
    image = b"synthetic test image bytes"
    response = client.post(
        "/api/v1/scan", headers=_auth(settings),
        json={"scan_type": "qr", "image_bytes": base64.urlsafe_b64encode(image).decode()},
    )
    assert response.status_code == 200
    assert engine.requests[0][0].image_bytes == image
    assert image not in settings.db_path.read_bytes()


def test_scan_sanitizes_invalid_json_fields_values_and_query(service, settings) -> None:
    client, engine, _ = service
    cases = [
        ("/api/v1/scan", {"scan_type": "sms", "text": "valid", "secret_extra_key": "secret_extra_value"}),
        ("/api/v1/scan", {"scan_type": "url", "url": "secret_invalid_url"}),
        ("/api/v1/scan?stream=secret_bad_boolean", {"scan_type": "sms", "text": "valid"}),
    ]
    for path, payload in cases:
        response = client.post(path, headers=_auth(settings), json=payload)
        assert response.status_code == 422
        assert "secret_" not in response.text
    assert engine.requests == []


def test_scan_rejects_unexpected_media_type_and_bounds_stream_input(service, settings, monkeypatch) -> None:
    client, engine, _ = service
    monkeypatch.setattr("api.main._MAX_SCAN_BODY", 30)
    response = client.post("/api/v1/scan", headers=_auth(settings), content=b"invalid")
    assert response.status_code == 415
    response = client.post(
        "/api/v1/scan", headers={**_auth(settings), "Content-Type": "application/json"},
        content=iter((b"x" * 20, b"y" * 20)),
    )
    assert response.status_code == 413
    assert engine.requests == []


def test_api_limits_requests_per_key_and_persists_across_restart(settings) -> None:
    payload = {"scan_type": "sms", "text": "A synthetic sample."}
    engine = RecordingEngine()
    with TestClient(create_app(settings, engine=engine)) as client:
        for _ in range(settings.requests_per_window):
            assert client.post("/api/v1/scan", headers=_auth(settings), json=payload).status_code == 200
        denied = client.post("/api/v1/scan", headers=_auth(settings), json=payload)
        assert denied.status_code == 429
        assert 1 <= int(denied.headers["Retry-After"]) <= settings.window_seconds
        assert client.post("/api/v1/scan", headers=_auth(settings, 1), json=payload).status_code == 200
    with TestClient(create_app(settings, engine=engine)) as client:
        assert client.post("/api/v1/scan", headers=_auth(settings), json=payload).status_code == 429
    assert len(engine.requests) == 4


def test_atomic_sliding_window_counts_individual_expiries(tmp_path) -> None:
    now = 1000.0
    path = tmp_path / "control.sqlite3"
    key = APIKeys(("test-only-long-private-key",)).authenticate("test-only-long-private-key")
    store = ControlStore(path, clock=lambda: now)
    assert key is not None
    options = {"limit": 2, "window_seconds": 60}
    assert store.consume(key, **options).allowed
    now += 20
    assert store.consume(key, **options).allowed
    now += 39.5
    assert store.consume(key, **options).retry_after == 1
    now += 0.5
    assert store.consume(key, **options).allowed
    assert store.consume(key, **options).retry_after == 20
    now += 90
    with ThreadPoolExecutor(max_workers=8) as workers:
        verdicts = list(workers.map(lambda _: store.consume(key, **options), range(8)))
    assert sum(result.allowed for result in verdicts) == options["limit"]
    assert b"test-only-long-private-key" not in path.read_bytes()


@pytest.mark.parametrize("selection", ["query", "accept"])
def test_scan_stream_emits_progress_and_real_result(service, settings, selection) -> None:
    client, engine, repository = service
    path = "/api/v1/scan?stream=true" if selection == "query" else "/api/v1/scan"
    headers = _auth(settings)
    if selection == "accept":
        headers["Accept"] = "text/event-stream"
    response = client.post(path, headers=headers, json={"scan_type": "sms", "text": "Synthetic SMS"})
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert response.headers["X-Accel-Buffering"] == "no"
    progress, result = response.text.strip().split("\n\n")
    assert progress == 'event: progress\ndata: {"stage": "scanning"}'
    assert result.startswith("event: result\ndata: ")
    report = json.loads(result.partition("data: ")[2])
    assert report["risk_score"] == 45
    assert repository.get_scan(report["scan_id"]) is not None
    assert len(engine.requests) == 1


def test_scan_failures_report_generic_errors_in_both_formats(service, settings) -> None:
    client, engine, _ = service
    engine.fail = True
    payload = {"scan_type": "sms", "text": "Private text"}
    json_response = client.post("/api/v1/scan", headers=_auth(settings), json=payload)
    assert json_response.status_code == 503
    stream_response = client.post("/api/v1/scan?stream=true", headers=_auth(settings), json=payload)
    assert stream_response.status_code == 200
    assert "event: error\n" in stream_response.text
    assert "event: result\n" not in stream_response.text
    assert "private engine details" not in json_response.text + stream_response.text


def test_sse_closing_stream_cancels_unfinished_scan() -> None:
    async def check() -> None:
        cancelled = asyncio.Event()

        async def long_scan():
            try:
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        stream = scan_events(long_scan, heartbeat=0.01)
        assert "event: progress" in await anext(stream)
        assert ": keep-alive" in await anext(stream)
        await stream.aclose()
        assert cancelled.is_set()

    asyncio.run(check())
