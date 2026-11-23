"""Evaluate local scam cues and verified domain age without storing raw inputs."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit

from core.models.indian_scam_tagger import ScamCategory, tag_indian_scams
from core.schemas import UPIPayload
from core.validators import validate_text, validate_url

NEW_DOMAIN_DAYS = 30

_URGENCY = re.compile(
    r"\b(?:urgent(?:ly)?|immediately|act\s+now|final\s+warning|last\s+chance|"
    r"within\s+\d{1,3}\s*(?:minutes?|mins?|hours?|hrs?))\b"
)
_CREDENTIAL_REQUEST = re.compile(
    r"\b(?:share|send|provide|enter|confirm|tell\s+us|reply\s+with)\s+"
    r"(?:(?:your|the|this)\s+)?(?:otp|upi[-\s]+pin|cvv|password|verification\s+code)\b"
)
_NEGATED_ACTION = re.compile(
    r"\b(?:never|do\s+not|don['’]t)\s+"
    r"(?:share|send|provide|enter|confirm|tell\s+us|reply\s+with|scan|use|approve|accept|authorize)\b"
)
_PAYMENT_ACTION = re.compile(r"\b(?:scan|approve|authorize|enter)\b")
_RECEIVE_MONEY = re.compile(
    r"\b(?:receive|claim|collect|get)\s+(?:(?:your|this|a|the)\s+)?"
    r"(?:money|payment|refund|cashback|prize|bonus|rewards?)\b"
)
_HOST_TOKEN_BREAK = re.compile(r"[.-]+")

# These allowlists include current and legacy first-party banking domains.
# SBI's .sbi TLD is exclusive to the bank.
_BANK_DOMAINS = (
    (frozenset({"sbi", "statebankofindia", "sbibank"}), ("sbi.bank.in", "sbi.co.in", "bank.sbi")),
    (frozenset({"hdfc", "hdfcbank"}), ("hdfc.bank.in", "hdfcbank.com")),
    (frozenset({"icici", "icicibank"}), ("icici.bank.in", "icicibank.com")),
)


@dataclass(frozen=True, slots=True)
class RuleHit:
    """A stable rule identifier, its signed point value, and its explanation."""

    code: str
    points: int
    summary: str

    def __post_init__(self) -> None:
        if type(self.points) is not int or not -100 <= self.points <= 100:
            raise ValueError("Rule points must be an integer from -100 to 100.")


_URGENT_MESSAGE = RuleHit(
    "message.urgency", 8, "The message uses a deadline or language that pressures you to act quickly."
)
_REQUESTS_CREDENTIALS = RuleHit(
    "message.credentials_requested", 18, "The message asks you to share or enter a sensitive credential."
)
_DIGITAL_ARREST = RuleHit(
    "message.digital_arrest", 22, "The message contains a remote arrest or investigation cue."
)
_UPI_RECEIVE_REQUEST = RuleHit(
    "upi.receive_request", 22, "The message asks you to use a payment QR or approve a request to receive money."
)
_UPI_PAYMENT_TO_RECEIVE = RuleHit(
    "upi.payment_to_receive", 28, "A QR that specifies a payment amount accompanies a request to receive money."
)
_HTTP_LINK = RuleHit("url.unencrypted_http", 5, "A link uses unencrypted HTTP.")
_IP_LINK = RuleHit("url.ip_literal", 10, "A link uses an IP address as its hostname.")
_BANK_NAME_MISMATCH = RuleHit(
    "url.bank_name_mismatch",
    24,
    "A link uses a recognized bank name in a hostname outside the bank domains known to this rule.",
)
_RECENT_DOMAIN = RuleHit(
    "domain.recent_registration",
    15,
    f"The available registration data puts the checked domain under {NEW_DOMAIN_DAYS} days old.",
)


def _bank_name_mismatch(host: str) -> bool:
    tokens = set(_HOST_TOKEN_BREAK.split(host))
    for bank_tokens, domains in _BANK_DOMAINS:
        if not bank_tokens & tokens:
            continue
        if "sbi" in bank_tokens and host.endswith(".sbi"):
            continue
        if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return True
    return False


def _is_ip(host: str) -> bool:
    try:
        ip_address(host)
    except ValueError:
        return False
    return True


def evaluate_rules(
    *,
    text: str | None = None,
    urls: Iterable[str] = (),
    upi: UPIPayload | None = None,
    domain_age_days: int | None = None,
) -> tuple[RuleHit, ...]:
    """Evaluate each local indicator once, adding a registration cue only for known ages."""
    if domain_age_days is not None and (type(domain_age_days) is not int or domain_age_days < 0):
        raise ValueError("Domain age must be a nonnegative number of whole days.")

    hits = []
    if text is not None:
        normalized = unicodedata.normalize("NFKC", validate_text(text)).casefold()
        requests = _NEGATED_ACTION.sub(" ", normalized)
        categories = {tag.category for tag in tag_indian_scams(requests)}
        if _URGENCY.search(normalized):
            hits.append(_URGENT_MESSAGE)
        if _CREDENTIAL_REQUEST.search(requests):
            hits.append(_REQUESTS_CREDENTIALS)
        if ScamCategory.DIGITAL_ARREST in categories:
            hits.append(_DIGITAL_ARREST)
        if (
            upi is not None
            and upi.am is not None
            and upi.am > 0
            and _PAYMENT_ACTION.search(requests)
            and _RECEIVE_MONEY.search(requests)
        ):
            hits.append(_UPI_PAYMENT_TO_RECEIVE)
        elif ScamCategory.UPI_QR in categories:
            hits.append(_UPI_RECEIVE_REQUEST)

    insecure, ip_literal, bank_mismatch = False, False, False
    for url in urls:
        parsed = urlsplit(validate_url(url))
        host = (parsed.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        insecure |= parsed.scheme == "http"
        ip_literal |= _is_ip(host)
        bank_mismatch |= _bank_name_mismatch(host)
    if insecure:
        hits.append(_HTTP_LINK)
    if ip_literal:
        hits.append(_IP_LINK)
    if bank_mismatch:
        hits.append(_BANK_NAME_MISMATCH)
    if domain_age_days is not None and domain_age_days < NEW_DOMAIN_DAYS:
        hits.append(_RECENT_DOMAIN)
    return tuple(hits)
