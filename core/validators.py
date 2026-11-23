"""Validate untrusted scan inputs without network access or echoing their values."""

import re
import unicodedata
from ipaddress import IPv4Address, IPv6Address
from urllib.parse import urlsplit

MAX_TEXT_LENGTH = 32_768
MAX_URL_LENGTH = 4_096
MAX_IMAGE_BYTES = 10 * 1024 * 1024

_HTTP_SCHEMES = {"http", "https"}
_DNS_LABEL = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?")
_IPV4_ENDING = re.compile(r"(?:[0-9]+|0[xX][0-9a-fA-F]+)")
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_MOBILE = re.compile(r"(?:\+91|0091|91|0)?([6-9][0-9]{9})")
_UPI_ADDRESS = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}@[a-zA-Z0-9][a-zA-Z0-9.-]{0,62}"
)


def validate_text(value: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Check a text input while preserving whitespace, script, and case."""
    if not isinstance(value, str) or len(value) > max_length or not value.strip():
        raise ValueError("Expected nonempty text within the length limit.")
    if any(
        character not in "\t\r\n" and unicodedata.category(character) in {"Cc", "Cs"}
        for character in value
    ):
        raise ValueError("Text contains unsupported control characters.")
    return value


def validate_port(port: int | str) -> int:
    """Return a numeric TCP port in the range 1 to 65535."""
    if isinstance(port, str) and re.fullmatch(r"[0-9]{1,5}", port):
        port = int(port)
    if type(port) is not int or not 1 <= port <= 65_535:
        raise ValueError("Port must be an integer from 1 to 65535.")
    return port


def _validate_hostname(hostname: str) -> None:
    hostname = hostname.removesuffix(".")
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("URL has an invalid hostname.") from None

    labels = ascii_hostname.split(".")
    if len(ascii_hostname) > 253 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("URL has an invalid hostname.")
    if _IPV4_ENDING.fullmatch(labels[-1]):
        try:
            IPv4Address(ascii_hostname)
        except ValueError:
            raise ValueError("URL has an invalid IPv4 address.") from None


def _validate_authority(authority: str) -> None:
    if not authority or "@" in authority:
        raise ValueError("URL requires a hostname and cannot contain credentials.")

    if authority.startswith("["):
        address, separator, suffix = authority[1:].partition("]")
        if not separator or (suffix and not suffix.startswith(":")) or "%" in address:
            raise ValueError("URL has an invalid IPv6 address.")
        try:
            IPv6Address(address)
        except ValueError:
            raise ValueError("URL has an invalid IPv6 address.") from None
        if suffix:
            validate_port(suffix[1:])
        return

    hostname, separator, port = authority.partition(":")
    _validate_hostname(hostname)
    if separator:
        validate_port(port)


def validate_url(value: str) -> str:
    """Validate an absolute HTTP(S) URL and return it with its original spelling."""
    if not isinstance(value, str) or not value or len(value) > MAX_URL_LENGTH:
        raise ValueError("Expected a URL within the length limit.")
    if any(
        character in '\\<>"'
        or character.isspace()
        or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    ):
        raise ValueError("URL contains unsupported characters.")
    if _INVALID_PERCENT_ESCAPE.search(value):
        raise ValueError("URL contains an invalid percent escape.")

    try:
        parts = urlsplit(value)
    except ValueError:
        raise ValueError("URL has an invalid authority.") from None
    if parts.scheme.lower() not in _HTTP_SCHEMES:
        raise ValueError("URL must use HTTP or HTTPS.")
    _validate_authority(parts.netloc)
    return value


def validate_indian_mobile(value: str) -> str:
    """Normalize an Indian mobile number to +91 followed by ten ASCII digits."""
    if not isinstance(value, str) or len(value) > 24:
        raise ValueError("Expected an Indian mobile number.")
    compact = value.replace(" ", "").replace("-", "")
    match = _MOBILE.fullmatch(compact)
    if match is None:
        raise ValueError("Expected an Indian mobile number.")
    return f"+91{match[1]}"


def validate_upi_address(value: str) -> str:
    """Check VPA syntax locally without resolving or verifying the payee."""
    if not isinstance(value, str) or not _UPI_ADDRESS.fullmatch(value):
        raise ValueError("Expected a valid UPI virtual payment address.")
    handle = value.rsplit("@", 1)[1]
    if handle.endswith((".", "-")) or ".." in handle:
        raise ValueError("Expected a valid UPI virtual payment address.")
    return value
