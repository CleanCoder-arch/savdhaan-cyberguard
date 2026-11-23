"""Scrub common sensitive patterns and export allowlisted scan metadata."""

import re
import textwrap
import unicodedata
from io import BytesIO
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont

from core.guidance.rbi_helplines import CYBER_FRAUD_HELPLINE, CYBERCRIME_PORTAL, get_scam_playbook
from core.schemas import ScanReport, ScanStatus

_URL = re.compile(
    r"(?<![\w@])(?:[a-z][a-z0-9+.-]{1,31}://|//|www\.)[^\s<>'\"\x60]+"
    r"|(?<![\w@.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?[/?#][^\s<>'\"\x60]*"
    r"|(?<![\w@.-])(?:[a-z0-9](?:[a-z0-9-]{0,62})\.){1,20}[a-z]{2,63}"
    r"(?=[:/?#]|$|[\s.,!?)\]}])(?::[0-9]{1,5})?(?:[/?#][^\s<>'\"\x60]*)?",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"(?<![\w@%+.-])[\w.+%-]+(?:@|%40)[\w-]+(?:\.[\w-]+)*",
    re.IGNORECASE,
)
_AADHAAR = re.compile(r"(?<![\w+])(?:\d{4}|[xX*]{4})[ -]?(?:\d{4}|[xX*]{4})[ -]?\d{4}(?!\w)")
_PAN = re.compile(r"(?<!\w)[A-Z]{5}[0-9]{4}[A-Z](?!\w)", re.IGNORECASE | re.ASCII)
_PHONE_SEPARATOR = r"[ .()-]{0,3}"
_PHONE = re.compile(
    rf"(?<![\w+])(?:"
    rf"(?:\+|00)\d(?:{_PHONE_SEPARATOR}\d){{7,14}}"
    rf"|(?:(?:0|91){_PHONE_SEPARATOR})?[6-9६-९](?:{_PHONE_SEPARATOR}\d){{9}}"
    rf"|0{_PHONE_SEPARATOR}[1-5](?:{_PHONE_SEPARATOR}\d){{9}}"
    rf")(?!\w)"
)
_LABELED_PHONE = re.compile(
    r"(?<!\w)(?P<label>(?:phone|mobile|contact|tel)(?:[ \t]+(?:number|no\.?))?[ \t]*[:=][ \t]*)"
    rf"\(?(?:\+|00)?\d(?:{_PHONE_SEPARATOR}\d){{7,14}}(?!\w)",
    re.IGNORECASE,
)
_OTP_LABEL = r"(?:otp|one[- ]time (?:password|code|pin)|verification code|security code|upi pin)"
_OTP_VALUE = r"\d(?:[ -]?\d){3,7}"
_OTP_AFTER_LABEL = re.compile(
    rf"(?<!\w)(?P<label>{_OTP_LABEL}\b[ \t]*(?:is\b[ \t]*)?[:=#-]?[ \t]*"
    rf"(?:(?:\r\n?|\n)[ \t]*)?){_OTP_VALUE}(?!\w)",
    re.IGNORECASE,
)
_OTP_BEFORE_LABEL = re.compile(
    rf"(?<!\w){_OTP_VALUE}(?P<label>[ \t]+is[ \t]+(?:your[ \t]+)?{_OTP_LABEL}\b)",
    re.IGNORECASE,
)
_SECRET_PARAMETER = re.compile(
    r"(?<!\w)(?P<label>(?:access[_-]?token|refresh[_-]?token|id[_-]?token|token|session[_-]?id|api[_-]?key)"
    r"[ \t]*=[ \t]*)(?!\[REDACTED SECRET\])[^\s&#<>'\"\x60]+",
    re.IGNORECASE,
)
_INVISIBLE_TEXT = str.maketrans("", "", "\u200b\u2060\ufeff")


def redact_text(text: str) -> str:
    """Remove recognized URLs, addresses, identity numbers, phones, and labelled secrets."""
    redacted = unicodedata.normalize("NFKC", text).translate(_INVISIBLE_TEXT)
    redacted = _URL.sub("[REDACTED URL]", redacted)
    redacted = _ADDRESS.sub("[REDACTED ADDRESS]", redacted)
    redacted = _SECRET_PARAMETER.sub(r"\g<label>[REDACTED SECRET]", redacted)
    redacted = _OTP_AFTER_LABEL.sub(r"\g<label>[REDACTED OTP]", redacted)
    redacted = _OTP_BEFORE_LABEL.sub(r"[REDACTED OTP]\g<label>", redacted)
    redacted = _LABELED_PHONE.sub(r"\g<label>[REDACTED PHONE]", redacted)
    redacted = _AADHAAR.sub("[REDACTED AADHAAR]", redacted)
    redacted = _PAN.sub("[REDACTED PAN]", redacted)
    return _PHONE.sub("[REDACTED PHONE]", redacted)


class RedactedReport(TypedDict):
    scan_type: str
    status: str
    risk_level: str | None
    risk_score: int | None
    scam_categories: list[str]
    evidence_count: int
    link_count: int
    issue_count: int
    upi_detected: bool


def build_redacted_report(report: ScanReport) -> RedactedReport:
    """Return JSON-ready fields that are safe to share outside the scan session."""
    return {
        "scan_type": report.scan_type.value,
        "status": report.status.value,
        "risk_level": report.risk_level.value if report.risk_level is not None else None,
        "risk_score": report.risk_score,
        "scam_categories": [category.value for category in dict.fromkeys(report.scam_categories)],
        "evidence_count": len(report.evidence),
        "link_count": len(report.links),
        "issue_count": len(report.issues),
        "upi_detected": report.upi is not None,
    }


_RISK_COLORS = {
    "LOW": "#03624C",
    "MEDIUM": "#8C4704",
    "HIGH": "#A2251E",
    "CRITICAL": "#77163E",
}


def generate_alert_card(report: ScanReport) -> bytes:
    """Render a PNG in memory from risk metadata and curated public guidance."""
    risk_level = report.risk_level.value if report.risk_level is not None else None
    risk_color = _RISK_COLORS.get(risk_level, "#38546C")
    risk_text = f"{risk_level.title()} risk" if risk_level else "Risk level unavailable"
    if report.risk_score is not None:
        risk_text = f"{risk_text}  |  {report.risk_score}/100"
    status_text = "Partial scan" if report.status == ScanStatus.PARTIAL else "Scan complete"

    category_text = ", ".join(
        get_scam_playbook(category).title for category in dict.fromkeys(report.scam_categories)
    )
    category_lines = textwrap.wrap(category_text, width=48)
    contact_y = 247 + 38 * len(category_lines)

    heading = ImageFont.load_default(size=44)
    body = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=22)
    with BytesIO() as output, Image.new("RGB", (900, contact_y + 129), "#F7F8FD") as card:
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card.width, 9), fill=risk_color)
        draw.text((48, 39), "SAVDHAAN AI", font=small, fill="#38546C")
        draw.text((48, 76), "Scam alert", font=heading, fill="#131B2B")
        draw.text((48, 145), risk_text, font=body, fill=risk_color)
        draw.text((48, 193), status_text, font=small, fill="#38546C")
        for index, line in enumerate(category_lines):
            draw.text((48, 231 + 38 * index), line, font=body, fill="#131B2B")
        draw.line((48, contact_y, 852, contact_y), fill="#BFC8D3", width=2)
        draw.text(
            (48, contact_y + 23),
            f"If you lost money, contact your bank and call {CYBER_FRAUD_HELPLINE}.",
            font=body,
            fill="#131B2B",
        )
        draw.text((48, contact_y + 65), CYBERCRIME_PORTAL, font=small, fill="#38546C")
        card.save(output, format="PNG")
        return output.getvalue()
