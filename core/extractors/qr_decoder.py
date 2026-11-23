"""Decode a local QR image or inspect the fields of a UPI payment URI."""

import re
from decimal import Decimal
from urllib.parse import parse_qsl, urlsplit

import cv2
import numpy as np
from pydantic import ValidationError

from core.extractors._images import ImageInputError, load_image
from core.schemas import UPIPayload
from core.validators import validate_text


_MAX_UPI_URI_LENGTH = 4096
_UPI_FIELDS = frozenset({"pa", "pn", "am", "cu", "mc"})
_INVALID_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_DECIMAL_AMOUNT = re.compile(r"[0-9]+(?:\.[0-9]{1,2})?")


class QRCodeError(ValueError):
    """The image does not contain exactly one decodable QR payload."""


class QRCodeNotFoundError(QRCodeError):
    """OpenCV could not find a decodable QR payload."""


class MultipleQRCodesError(QRCodeError):
    """The image contains multiple QR codes."""


class InvalidUPIPayloadError(ValueError):
    """The decoded text cannot be parsed as a UPI payment URI."""


def decode_qr(image_bytes: bytes) -> str:
    """Decode one QR payload as untrusted UTF-8 text for local analysis."""
    try:
        image = load_image(image_bytes)
    except ImageInputError as exc:
        raise QRCodeError(str(exc)) from None

    try:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        detector = cv2.QRCodeDetector()
        _, payloads, points, _ = detector.detectAndDecodeMulti(gray)
        if points is not None and len(points) > 1:
            raise MultipleQRCodesError("Provide an image with a single QR code.")
        payload = payloads[0] if payloads else ""
        if not payload:
            payload, _, _ = detector.detectAndDecode(gray)
    except UnicodeError:
        raise QRCodeError("The QR payload is not valid UTF-8 text.") from None
    except cv2.error:
        raise QRCodeError("OpenCV could not process the QR image.") from None

    if not payload:
        raise QRCodeNotFoundError("No decodable QR code was found in the image.")
    try:
        return validate_text(payload)
    except ValueError as exc:
        raise QRCodeError(str(exc)) from None


def parse_upi_uri(uri: str) -> UPIPayload:
    """Extract known UPI fields for analysis without initiating a payment."""
    if (
        not isinstance(uri, str)
        or not uri
        or len(uri) > _MAX_UPI_URI_LENGTH
        or "#" in uri
        or any(char.isspace() or not char.isprintable() for char in uri)
    ):
        raise InvalidUPIPayloadError("The UPI URI is malformed or too long.")

    try:
        parts = urlsplit(uri)
        if parts.scheme != "upi" or parts.netloc.lower() != "pay" or parts.path:
            raise InvalidUPIPayloadError("Provide a upi://pay URI.")
        if not parts.query or _INVALID_ESCAPE.search(parts.query):
            raise InvalidUPIPayloadError("The UPI URI has an invalid query.")
        pairs = parse_qsl(
            parts.query,
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=32,
        )
    except InvalidUPIPayloadError:
        raise
    except ValueError:
        raise InvalidUPIPayloadError("The UPI URI has an invalid query.") from None

    fields: dict[str, str] = {}
    for name, value in pairs:
        if name in _UPI_FIELDS:
            if name in fields:
                raise InvalidUPIPayloadError("A UPI field was supplied more than once.")
            fields[name] = value

    amount = fields.get("am")
    if amount and not _DECIMAL_AMOUNT.fullmatch(amount):
        raise InvalidUPIPayloadError("The UPI amount must have up to two decimal places.")

    try:
        return UPIPayload(
            pa=fields.get("pa", ""),
            pn=fields.get("pn") or None,
            am=Decimal(amount) if amount else None,
            cu=fields.get("cu") or None,
            mc=fields.get("mc") or None,
        )
    except ValidationError:
        raise InvalidUPIPayloadError("One or more UPI fields are invalid.") from None
