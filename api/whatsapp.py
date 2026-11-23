"""Parse signed WhatsApp notifications and retrieve authorized inbound QR images."""

import asyncio
import hmac
import json
import re
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from core.schemas import ScanRequest, ScanType
from core.validators import MAX_IMAGE_BYTES, validate_url

GRAPH_VERSION = "v27.0"
_META_MEDIA_HOSTS = frozenset({"lookaside.fbsbx.com", "graph.facebook.com"})
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png"})
_GRAPH_ID = re.compile(r"[0-9]{1,32}")
_SIGNATURE = re.compile(r"sha256=[a-f0-9]{64}")
_MAX_MESSAGES = 1000


@dataclass(frozen=True, slots=True)
class WhatsAppEvent:
    message_id: str = field(repr=False)
    request: ScanRequest | None = field(default=None, repr=False)
    media_id: str | None = field(default=None, repr=False)
    phone_number_id: str | None = field(default=None, repr=False)


class MediaUnavailable(RuntimeError):
    """The Graph API could not provide a supported image."""


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    if not _SIGNATURE.fullmatch(signature):
        return False
    expected = "sha256=" + hmac.digest(secret.encode(), payload, "sha256").hex()
    return hmac.compare_digest(expected, signature)


def delivery_digest(message_id: str, secret: str) -> bytes:
    return hmac.digest(secret.encode(), b"savdhaan:whatsapp:v1:" + message_id.encode(), "sha256")


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _array(value: object) -> list:
    return value if isinstance(value, list) else []


def _graph_id(value: object) -> str | None:
    return value if isinstance(value, str) and _GRAPH_ID.fullmatch(value) else None


def _scan_text(text: str) -> ScanRequest | None:
    try:
        url = validate_url(text.strip())
    except ValueError:
        try:
            return ScanRequest(scan_type=ScanType.SMS, text=text)
        except ValidationError:
            return None
    return ScanRequest(scan_type=ScanType.URL, url=url)


def _parse_event(message: dict, phone_number_id: str | None) -> WhatsAppEvent | None:
    message_id = message.get("id")
    if not isinstance(message_id, str) or not 1 <= len(message_id) <= 512 or not message_id.isascii():
        return None
    message_type = message.get("type")
    if message_type == "text":
        body = _object(message.get("text")).get("body")
        if isinstance(body, str) and (request := _scan_text(body)) is not None:
            return WhatsAppEvent(message_id, request=request)
    elif message_type == "image":
        media_id = _graph_id(_object(message.get("image")).get("id"))
        if media_id is not None:
            return WhatsAppEvent(message_id, media_id=media_id, phone_number_id=phone_number_id)
    return None


def parse_notifications(payload: bytes) -> tuple[tuple[WhatsAppEvent, ...], int]:
    """Return actionable notifications and the count of skipped message types."""
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("Expected an event object.")
    if document.get("object") != "whatsapp_business_account":
        return (), 0
    events: list[WhatsAppEvent] = []
    ignored = 0
    for entry in _array(document.get("entry")):
        for change in _array(_object(entry).get("changes")):
            change = _object(change)
            if change.get("field") != "messages":
                continue
            value = _object(change.get("value"))
            phone_number_id = _graph_id(_object(value.get("metadata")).get("phone_number_id"))
            for message in _array(value.get("messages")):
                if len(events) + ignored >= _MAX_MESSAGES:
                    raise ValueError("Webhook message limit exceeded.")
                event = _parse_event(_object(message), phone_number_id)
                if event is None:
                    ignored += 1
                else:
                    events.append(event)
    return tuple(events), ignored


def _media_url(value: object) -> httpx.URL:
    if not isinstance(value, str):
        raise MediaUnavailable("Meta did not return a media address.")
    url = httpx.URL(value)
    if url.scheme != "https" or url.host not in _META_MEDIA_HOSTS or url.port not in {None, 443} or url.userinfo:
        raise MediaUnavailable("Meta returned an unexpected media address.")
    return url


async def _limited_download(
    client: httpx.AsyncClient,
    url: str | httpx.URL,
    maximum: int,
    *,
    params: dict[str, str] | None = None,
    image: bool = False,
) -> bytes:
    async with client.stream("GET", url, params=params) as response:
        response.raise_for_status()
        if response.headers.get("content-encoding", "").lower() not in {"", "identity"}:
            raise MediaUnavailable("The media response used unsupported encoding.")
        if image and response.headers.get("content-type", "").partition(";")[0].lower() not in _IMAGE_TYPES:
            raise MediaUnavailable("The media response was not a supported image.")
        length = response.headers.get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > maximum):
            raise MediaUnavailable("The media response exceeded its size limit.")
        content = bytearray()
        async for chunk in response.aiter_raw():
            if len(content) + len(chunk) > maximum:
                raise MediaUnavailable("The media response exceeded its size limit.")
            content.extend(chunk)
        if not content:
            raise MediaUnavailable("The media response was empty.")
        return bytes(content)


class WhatsAppMediaClient:
    """Retrieve images from Meta only, keeping access tokens out of URL parameters."""

    def __init__(self, access_token: str, version: str = GRAPH_VERSION) -> None:
        if not re.fullmatch(r"v[0-9]{1,3}\.[0-9]{1,2}", version):
            raise ValueError("The Meta Graph version must have the form v27.0.")
        self._access_token = access_token
        self._version = version

    async def retrieve_image(self, media_id: str, phone_number_id: str | None = None) -> bytes:
        if _graph_id(media_id) is None or (phone_number_id is not None and _graph_id(phone_number_id) is None):
            raise MediaUnavailable("The media identifier was invalid.")
        try:
            async with asyncio.timeout(10), httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=False,
                trust_env=False,
                headers={"Authorization": f"Bearer {self._access_token}", "Accept-Encoding": "identity"},
            ) as client:
                metadata = _object(json.loads(await _limited_download(
                    client, f"https://graph.facebook.com/{self._version}/{media_id}", 64 * 1024,
                    params={"phone_number_id": phone_number_id} if phone_number_id is not None else None,
                )))
                if metadata.get("mime_type") not in _IMAGE_TYPES:
                    raise MediaUnavailable("Meta did not identify a supported image.")
                return await _limited_download(client, _media_url(metadata.get("url")), MAX_IMAGE_BYTES, image=True)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TimeoutError, RecursionError):
            raise MediaUnavailable("The Meta image could not be retrieved.") from None
