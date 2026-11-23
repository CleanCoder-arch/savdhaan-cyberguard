import asyncio
import hmac
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api.auth import ControlStore
from api.main import APISettings, create_app
from api.whatsapp import MediaUnavailable, WhatsAppMediaClient, delivery_digest
from core.schemas import ScanReport, ScanRequest, ScanType
from core.storage.repository import ScanRepository

_SECRET = "meta_test_signing_secret"
_VERIFY = "meta_test_verification_token"
_WAMID = "wamid.test-personal-looking-id"
_PHONE = "919512345678"
_MEDIA = b"\x89PNG\r\nsynthetic-image-data"


class WebhookEngine:
    readiness = {"url_model": True, "sms_model": True}

    def __init__(self) -> None:
        self.requests: list[ScanRequest] = []
        self.fail = False

    async def scan(self, request: ScanRequest, *, offline: bool = False) -> ScanReport:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("private transient issue")
        return ScanReport(scan_type=request.scan_type, risk_level="HIGH", risk_score=79)


def _settings(tmp_path, **credentials) -> APISettings:
    return APISettings(
        db_path=tmp_path / "scans.sqlite3", control_db_path=tmp_path / "control.sqlite3",
        model_dir=tmp_path, whatsapp_app_secret=_SECRET, whatsapp_verify_token=_VERIFY,
        **credentials,
    )


def _text(body: str, message_id: str = _WAMID) -> dict:
    return {"id": message_id, "from": _PHONE, "type": "text", "text": {"body": body}}


def _notification(messages=None, statuses=None) -> bytes:
    value = {"messaging_product": "whatsapp", "metadata": {"phone_number_id": "778899"}}
    if messages is not None:
        value["messages"] = messages
    if statuses is not None:
        value["statuses"] = statuses
    return json.dumps({
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"field": "messages", "value": value}]}],
    }).encode()


def _signed(body: bytes, *, key: str = _SECRET) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": "sha256=" + hmac.digest(key.encode(), body, "sha256").hex(),
    }


def test_meta_subscription_challenge_is_verified(tmp_path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, engine=WebhookEngine())) as client:
        params = {"hub.mode": "subscribe", "hub.challenge": "validation123", "hub.verify_token": _VERIFY}
        response = client.get("/api/v1/webhooks/whatsapp", params=params)
        assert response.status_code == 200
        assert response.text == "validation123"
        params["hub.verify_token"] = "wrong_private_token"
        response = client.get("/api/v1/webhooks/whatsapp", params=params)
        assert response.status_code == 403
        assert "wrong_private_token" not in response.text


@pytest.mark.parametrize("mode", ["missing", "wrong", "modified"])
def test_meta_rejects_invalid_signatures_before_processing(tmp_path, mode) -> None:
    engine = WebhookEngine()
    body = _notification([_text("private test message")])
    headers = _signed(body, key="wrong" if mode == "wrong" else _SECRET)
    if mode == "missing":
        del headers["X-Hub-Signature-256"]
    payload = body[:-1] + b" }" if mode == "modified" else body
    with TestClient(create_app(_settings(tmp_path), engine=engine)) as client:
        response = client.post("/api/v1/webhooks/whatsapp", content=payload, headers=headers)
        assert response.status_code == 401
        assert "private test message" not in response.text
        assert engine.requests == []


def test_webhook_processes_text_links_and_ignores_status_notifications(tmp_path) -> None:
    settings = _settings(tmp_path)
    engine = WebhookEngine()
    url = "https://example.test/CaseSensitive?token=private-query-value"
    sms = "Private test SMS marker 316159"
    body = _notification([_text(url), _text(sms, f"{_WAMID}-second")])
    with TestClient(create_app(settings, engine=engine)) as client:
        response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
        assert response.status_code == 200
        assert response.json() == {"accepted": 2, "duplicates": 0, "ignored": 0}
        assert engine.requests == [
            ScanRequest(scan_type="url", url=url), ScanRequest(scan_type="sms", text=sms),
        ]
        status = _notification(statuses=[{"id": "wamid.outbound", "status": "delivered"}])
        response = client.post("/api/v1/webhooks/whatsapp", content=status, headers=_signed(status))
        assert response.status_code == 200
        assert response.json() == {"accepted": 0, "duplicates": 0, "ignored": 0}
    combined = settings.db_path.read_bytes() + settings.control_db_path.read_bytes()
    for private in (_PHONE, _WAMID, _SECRET, sms, url, "private-query-value"):
        assert private.encode() not in combined
    assert len(ScanRepository(settings.db_path).list_scans()) == 2


def test_webhook_deduplicates_retries_across_api_workers(tmp_path) -> None:
    settings = _settings(tmp_path)
    body = _notification([_text("A simulated forwarded SMS.")])
    engine = WebhookEngine()
    for expected in ({"accepted": 1, "duplicates": 0, "ignored": 0}, {"accepted": 0, "duplicates": 1, "ignored": 0}):
        with TestClient(create_app(settings, engine=engine)) as client:
            response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
            assert response.status_code == 200
            assert response.json() == expected
    assert len(engine.requests) == 1
    assert len(ScanRepository(settings.db_path).list_scans()) == 1


def test_transient_scan_failure_allows_meta_to_retry(tmp_path) -> None:
    settings = _settings(tmp_path)
    body = _notification([_text("A simulated forwarded SMS.")])
    engine = WebhookEngine()
    engine.fail = True
    with TestClient(create_app(settings, engine=engine)) as client:
        response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
        assert response.status_code == 503
        assert "private transient issue" not in response.text
        engine.fail = False
        response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
        assert response.status_code == 200
        assert response.json()["accepted"] == 1
    assert len(engine.requests) == 2
    assert len(ScanRepository(settings.db_path).list_scans()) == 1


def test_competing_webhook_deliveries_use_unique_atomic_lease(tmp_path) -> None:
    controls = ControlStore(tmp_path / "control.sqlite3")
    digest = delivery_digest(_WAMID, _SECRET)
    first = controls.claim_delivery(digest)
    second = controls.claim_delivery(digest)
    assert first.token is not None
    assert second.token is None and not second.complete
    assert second.scan_id == first.scan_id
    controls.finish_delivery(digest, token=b"invalid", complete=True)
    assert not controls.claim_delivery(digest).complete
    controls.finish_delivery(digest, token=first.token, complete=True)
    assert controls.claim_delivery(digest).complete


def test_webhook_processes_signed_image_using_graph_media_identifier(tmp_path, monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    async def retrieve(_self, media, phone_number_id):
        seen.append((media, phone_number_id))
        return _MEDIA

    monkeypatch.setattr(WhatsAppMediaClient, "retrieve_image", retrieve)
    settings = _settings(tmp_path, whatsapp_access_token="test_graph_access_token")
    engine = WebhookEngine()
    message = {"id": _WAMID, "type": "image", "from": _PHONE, "image": {"id": "123456", "mime_type": "image/png"}}
    body = _notification([message])
    with TestClient(create_app(settings, engine=engine)) as client:
        response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
        assert response.status_code == 200
        assert response.json()["accepted"] == 1
    assert seen == [("123456", "778899")]
    assert engine.requests == [ScanRequest(scan_type=ScanType.QR, image_bytes=_MEDIA)]
    assert _MEDIA not in settings.db_path.read_bytes()


def test_unsupported_and_unconfigured_media_are_acknowledged_without_scanning(tmp_path) -> None:
    engine = WebhookEngine()
    body = _notification([
        {"id": f"{_WAMID}-image", "type": "image", "image": {"id": "1234"}},
        {"id": f"{_WAMID}-audio", "type": "audio", "audio": {"id": "5678"}},
    ])
    with TestClient(create_app(_settings(tmp_path), engine=engine)) as client:
        response = client.post("/api/v1/webhooks/whatsapp", content=body, headers=_signed(body))
        assert response.status_code == 200
        assert response.json() == {"accepted": 0, "duplicates": 0, "ignored": 2}
    assert engine.requests == []


class _ResponseBytes(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def __aiter__(self):
        yield self._content


def _mock_graph(monkeypatch, *, address: str, redirect: bool = False) -> list[httpx.Request]:
    requests: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "graph.facebook.com":
            body = json.dumps({"mime_type": "image/png", "url": address}).encode()
            return httpx.Response(200, stream=_ResponseBytes(body))
        if redirect:
            return httpx.Response(307, headers={"Location": "https://external.example.test/steal"})
        return httpx.Response(200, headers={"Content-Type": "image/png"}, stream=_ResponseBytes(_MEDIA))

    def client_factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("api.whatsapp.httpx.AsyncClient", client_factory)
    return requests


def test_media_download_uses_short_lived_meta_address_and_auth_header(monkeypatch) -> None:
    address = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1234"
    requests = _mock_graph(monkeypatch, address=address)
    client = WhatsAppMediaClient("test_graph_token", version="v27.0")
    assert asyncio.run(client.retrieve_image("1234", "5678")) == _MEDIA
    assert [str(request.url) for request in requests] == [
        "https://graph.facebook.com/v27.0/1234?phone_number_id=5678", address,
    ]
    assert all(request.headers["Authorization"] == "Bearer test_graph_token" for request in requests)


@pytest.mark.parametrize("address", [
    "https://external.example.test/media", "http://lookaside.fbsbx.com/media",
    "https://lookaside.fbsbx.com.external.example.test/media", "https://lookaside.fbsbx.com:8443/media",
    "https://user@lookaside.fbsbx.com/media",
])
def test_media_rejects_addresses_that_might_leak_graph_credentials(monkeypatch, address) -> None:
    requests = _mock_graph(monkeypatch, address=address)
    with pytest.raises(MediaUnavailable):
        asyncio.run(WhatsAppMediaClient("test_graph_token").retrieve_image("1234"))
    assert [request.url.host for request in requests] == ["graph.facebook.com"]


def test_media_never_follows_redirects(monkeypatch) -> None:
    requests = _mock_graph(monkeypatch, address="https://lookaside.fbsbx.com/media", redirect=True)
    with pytest.raises(MediaUnavailable):
        asyncio.run(WhatsAppMediaClient("test_graph_token").retrieve_image("1234"))
    assert [request.url.host for request in requests] == ["graph.facebook.com", "lookaside.fbsbx.com"]
