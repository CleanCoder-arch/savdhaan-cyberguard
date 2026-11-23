"""FastAPI entry points for the shared scan engine and metadata repository."""

import asyncio
import hmac
import os
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from api.auth import APIKeys, ControlStore
from api.sse import scan_events
from api.whatsapp import (
    GRAPH_VERSION,
    WhatsAppEvent,
    WhatsAppMediaClient,
    delivery_digest,
    parse_notifications,
    verify_signature,
)
from core.models.artifacts import DEFAULT_MODEL_DIR, configured_model_dir
from core.scanner import ScanEngine
from core.schemas import ScanReport, ScanRequest, ScanType
from core.storage.repository import DEFAULT_DB_PATH, ScanRepository, configured_db_path
from core.validators import MAX_IMAGE_BYTES

_NO_STORE = {"Cache-Control": "no-store"}
_MAX_SCAN_BODY = (MAX_IMAGE_BYTES + 2) // 3 * 4 + 256 * 1024
_MAX_WEBHOOK_BODY = 3 * 1024 * 1024
_MAX_SCAN_SECONDS = 30
_API_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_SCAN_SCHEMA = ScanRequest.model_json_schema(ref_template="#/components/schemas/{model}")
_SCAN_SCHEMA.pop("$defs", None)


class ScanService(Protocol):
    @property
    def readiness(self) -> dict[str, bool]: ...

    async def scan(self, request: ScanRequest, *, offline: bool = False) -> ScanReport: ...


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None


@dataclass(frozen=True, slots=True)
class APISettings:
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    db_path: Path = DEFAULT_DB_PATH
    control_db_path: Path = Path("artifacts") / "api-control.sqlite3"
    model_dir: Path = DEFAULT_MODEL_DIR
    requests_per_window: int = 10
    window_seconds: int = 3600
    offline: bool = False
    whatsapp_app_secret: str | None = field(default=None, repr=False)
    whatsapp_verify_token: str | None = field(default=None, repr=False)
    whatsapp_access_token: str | None = field(default=None, repr=False)
    whatsapp_graph_version: str = GRAPH_VERSION

    def __post_init__(self) -> None:
        if self.requests_per_window < 1 or self.window_seconds < 1:
            raise ValueError("API rate limits must have a positive request count and window.")

    @classmethod
    def from_env(cls) -> "APISettings":
        primary_key = os.environ.get("SAVDHAAN_API_KEY", "")
        additional_keys = os.environ.get("SAVDHAAN_API_KEYS", "").split(",")
        api_keys = tuple(key.strip() for key in (primary_key, *additional_keys) if key.strip())
        return cls(
            api_keys=api_keys,
            db_path=configured_db_path(),
            control_db_path=Path(os.environ.get("SAVDHAAN_API_DB_PATH", "artifacts/api-control.sqlite3")),
            model_dir=configured_model_dir(),
            requests_per_window=_env_int("SAVDHAAN_RATE_LIMIT", 10),
            window_seconds=_env_int("SAVDHAAN_RATE_WINDOW_SECONDS", 3600),
            offline=os.environ.get("SAVDHAAN_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"},
            whatsapp_app_secret=os.environ.get("SAVDHAAN_WHATSAPP_APP_SECRET") or None,
            whatsapp_verify_token=os.environ.get("SAVDHAAN_WHATSAPP_VERIFY_TOKEN") or None,
            whatsapp_access_token=os.environ.get("SAVDHAAN_WHATSAPP_ACCESS_TOKEN") or None,
            whatsapp_graph_version=os.environ.get("SAVDHAAN_WHATSAPP_GRAPH_VERSION", GRAPH_VERSION),
        )


class ServiceUnavailable(RuntimeError):
    """The scan or metadata write did not complete."""


@dataclass(slots=True)
class _Runtime:
    settings: APISettings
    engine: ScanService
    repository: ScanRepository | None
    controls: ControlStore | None
    api_keys: APIKeys
    media: WhatsAppMediaClient | None
    scan_slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(8))

    async def scan(
        self, request: ScanRequest, *, offline: bool = False, scan_id: UUID | None = None,
    ) -> ScanReport:
        if self.repository is None:
            raise ServiceUnavailable("Scan metadata storage is unavailable.")
        try:
            async with asyncio.timeout(_MAX_SCAN_SECONDS), self.scan_slots:
                report = await self.engine.scan(request, offline=self.settings.offline or offline)
                if scan_id is not None:
                    report = report.model_copy(update={"scan_id": scan_id})
                await asyncio.to_thread(self.repository.save_scan, report)
                return report
        except Exception:
            # Provider and storage errors may include sensitive request details.
            raise ServiceUnavailable("The scan service could not complete the request.") from None


def _start(
    settings: APISettings, engine: ScanService | None, repository: ScanRepository | None,
) -> _Runtime:
    if repository is None:
        try:
            repository = ScanRepository(settings.db_path)
        except (OSError, sqlite3.Error):
            repository = None
    try:
        controls = ControlStore(settings.control_db_path)
    except (OSError, sqlite3.Error):
        controls = None
    media = (
        WhatsAppMediaClient(settings.whatsapp_access_token, settings.whatsapp_graph_version)
        if settings.whatsapp_access_token else None
    )
    return _Runtime(
        settings, engine if engine is not None else ScanEngine.from_artifacts(settings.model_dir),
        repository, controls, APIKeys(settings.api_keys), media,
    )


async def _read_json(request: Request, maximum: int) -> bytes:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise HTTPException(415, "Send a JSON request body.", headers=_NO_STORE)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise HTTPException(400, "Invalid request body length.", headers=_NO_STORE)
        if len(content_length) > 9 or int(content_length) > maximum:
            raise HTTPException(413, "Request body exceeds the supported size.", headers=_NO_STORE)
    body = bytearray()
    try:
        async with asyncio.timeout(15):
            async for chunk in request.stream():
                if len(body) + len(chunk) > maximum:
                    raise HTTPException(413, "Request body exceeds the supported size.", headers=_NO_STORE)
                body.extend(chunk)
    except (ClientDisconnect, TimeoutError):
        raise HTTPException(408, "The request body was interrupted.", headers=_NO_STORE) from None
    return bytes(body)


async def _scan_whatsapp_event(runtime: _Runtime, event: WhatsAppEvent, secret: str) -> str:
    if event.request is None and (event.media_id is None or runtime.media is None):
        return "ignored"
    controls, repository = runtime.controls, runtime.repository
    if controls is None or repository is None:
        raise ServiceUnavailable("Webhook metadata storage is unavailable.")
    digest = delivery_digest(event.message_id, secret)
    claim = await asyncio.to_thread(controls.claim_delivery, digest)
    if claim.complete:
        return "duplicates"
    if claim.token is None:
        raise ServiceUnavailable("The webhook notification is already being processed.")
    complete = False
    try:
        if await asyncio.to_thread(repository.get_scan, claim.scan_id) is not None:
            complete = True
            return "duplicates"
        scan_request = event.request
        if scan_request is None:
            image = await runtime.media.retrieve_image(event.media_id, event.phone_number_id)
            scan_request = ScanRequest(scan_type=ScanType.QR, image_bytes=image)
        await runtime.scan(scan_request, scan_id=claim.scan_id)
        complete = True
        return "accepted"
    finally:
        await asyncio.shield(asyncio.to_thread(controls.finish_delivery, digest, claim.token, complete=complete))


async def _storage_ready(storage: ScanRepository | ControlStore | None) -> bool:
    if storage is None:
        return False
    try:
        return await asyncio.to_thread(storage.ping)
    except (OSError, sqlite3.Error):
        return False


def create_app(
    settings: APISettings | None = None, *, engine: ScanService | None = None, repository: ScanRepository | None = None,
) -> FastAPI:
    """Create the service with optional local engine and repository injection."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = await asyncio.to_thread(_start, settings or APISettings.from_env(), engine, repository)
        yield

    application = FastAPI(
        title="Savdhaan CyberGuard", version="0.1.0", lifespan=lifespan,
        description="Analyze suspicious URLs, SMS messages, QR codes, and reviewed screenshot text.",
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "Invalid request parameters."}, status_code=422, headers=_NO_STORE)

    @application.get("/api/v1/health")
    async def health(request: Request) -> JSONResponse:
        runtime: _Runtime = request.app.state.runtime
        storage, limiter = await asyncio.gather(
            _storage_ready(runtime.repository), _storage_ready(runtime.controls),
        )
        models = runtime.engine.readiness
        whatsapp_ready = storage and limiter and bool(
            runtime.settings.whatsapp_app_secret and runtime.settings.whatsapp_verify_token
        )
        checks = {
            **models,
            "storage": storage,
            "rate_limiting": limiter,
            "scan_api_key": runtime.api_keys.configured,
            "whatsapp_webhook": whatsapp_ready,
            "whatsapp_media": whatsapp_ready and runtime.media is not None,
        }
        operational = storage and limiter
        ready = operational and runtime.api_keys.configured and all(models.values())
        return JSONResponse(
            {"status": "ok" if ready else "degraded", "checks": checks},
            status_code=200 if operational else 503,
            headers=_NO_STORE,
        )

    @application.post(
        "/api/v1/scan", response_model=ScanReport,
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": _SCAN_SCHEMA}}}},
        responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
    )
    async def scan(
        request: Request, api_key: Annotated[str | None, Security(_API_HEADER)], stream: bool = False,
        offline: bool = False,
    ):
        runtime: _Runtime = request.app.state.runtime
        if not runtime.api_keys.configured or runtime.controls is None or runtime.repository is None:
            raise HTTPException(503, "The scan API is not configured or available.", headers=_NO_STORE)
        digest = runtime.api_keys.authenticate(api_key) if len(request.headers.getlist("x-api-key")) == 1 else None
        if digest is None:
            raise HTTPException(401, "A valid API key is required.", headers={**_NO_STORE, "WWW-Authenticate": "APIKey"})
        try:
            decision = await asyncio.to_thread(
                runtime.controls.consume, digest, limit=runtime.settings.requests_per_window,
                window_seconds=runtime.settings.window_seconds,
            )
        except (sqlite3.Error, OSError):
            raise HTTPException(503, "Rate limiting is temporarily unavailable.", headers=_NO_STORE) from None
        headers = {**_NO_STORE, **decision.headers}
        if not decision.allowed:
            raise HTTPException(429, "API request limit reached.", headers=headers)
        body = await _read_json(request, _MAX_SCAN_BODY)
        try:
            scan_request = ScanRequest.model_validate_json(body)
        except ValidationError:
            raise HTTPException(422, "Invalid scan request.", headers=headers) from None

        async def run_scan() -> ScanReport:
            return await runtime.scan(scan_request, offline=offline)

        accepts_sse = any(
            part.partition(";")[0].strip().lower() == "text/event-stream"
            for part in request.headers.get("accept", "").split(",")
        )
        if stream or accepts_sse:
            return StreamingResponse(
                scan_events(run_scan), media_type="text/event-stream", headers={**headers, "X-Accel-Buffering": "no"},
            )
        try:
            report = await run_scan()
        except ServiceUnavailable:
            raise HTTPException(503, "The scan could not be completed. Try again later.", headers=headers) from None
        return JSONResponse(report.model_dump(mode="json"), headers=headers)

    @application.get("/api/v1/webhooks/whatsapp", response_class=PlainTextResponse)
    async def verify_whatsapp(request: Request) -> PlainTextResponse:
        runtime: _Runtime = request.app.state.runtime
        expected = runtime.settings.whatsapp_verify_token
        supplied = request.query_params.get("hub.verify_token", "")
        challenge = request.query_params.get("hub.challenge", "")
        if not expected:
            raise HTTPException(503, "WhatsApp verification is not configured.", headers=_NO_STORE)
        if (
            request.query_params.get("hub.mode") != "subscribe"
            or not hmac.compare_digest(supplied.encode(), expected.encode())
            or not 1 <= len(challenge) <= 256
            or not challenge.isascii()
            or not challenge.isprintable()
        ):
            raise HTTPException(403, "WhatsApp verification failed.", headers=_NO_STORE)
        return PlainTextResponse(challenge, headers=_NO_STORE)

    @application.post("/api/v1/webhooks/whatsapp")
    async def receive_whatsapp(request: Request) -> JSONResponse:
        runtime: _Runtime = request.app.state.runtime
        secret = runtime.settings.whatsapp_app_secret
        if not secret:
            raise HTTPException(503, "WhatsApp ingress is not configured.", headers=_NO_STORE)
        signatures = request.headers.getlist("x-hub-signature-256")
        if len(signatures) != 1:
            raise HTTPException(401, "Invalid WhatsApp signature.", headers=_NO_STORE)
        body = await _read_json(request, _MAX_WEBHOOK_BODY)
        if not verify_signature(body, signatures[0], secret):
            raise HTTPException(401, "Invalid WhatsApp signature.", headers=_NO_STORE)
        try:
            events, ignored = parse_notifications(body)
        except (ValueError, RecursionError):
            raise HTTPException(422, "Invalid WhatsApp notification.", headers=_NO_STORE) from None
        counts = {"accepted": 0, "duplicates": 0, "ignored": ignored}
        try:
            async with asyncio.timeout(55):
                for event in events:
                    counts[await _scan_whatsapp_event(runtime, event, secret)] += 1
        except Exception:
            raise HTTPException(503, "The WhatsApp notification could not be processed.", headers=_NO_STORE) from None
        return JSONResponse(counts, headers=_NO_STORE)

    return application


app = create_app()
