"""Typed contracts shared by the scan engine and its callers."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self, TypeVar
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from core.models.indian_scam_tagger import ScamCategory
from core.validators import (
    MAX_IMAGE_BYTES,
    validate_indian_mobile,
    validate_text,
    validate_upi_address,
    validate_url,
)

_HttpUrl = Annotated[str, AfterValidator(validate_url)]
_Text = Annotated[str, AfterValidator(validate_text)]
_Identifier = Annotated[str, Field(max_length=64, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")]
_RiskScore = Annotated[int, Field(ge=0, le=100)]
_ImageBytes = Annotated[bytes, Field(min_length=1, max_length=MAX_IMAGE_BYTES)]
_IndianMobile = Annotated[str, AfterValidator(validate_indian_mobile)]
_UpiAddress = Annotated[str, AfterValidator(validate_upi_address)]
_EnumType = TypeVar("_EnumType", bound=StrEnum)


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string enum value.")
    return value


_StringEnum = Annotated[_EnumType, Field(strict=False), BeforeValidator(_require_string)]


class ScanType(StrEnum):
    URL = "url"
    SMS = "sms"
    QR = "qr"
    SCREENSHOT = "screenshot"


class ScanStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceSource(StrEnum):
    VALIDATION = "validation"
    EXTRACTOR = "extractor"
    TAXONOMY = "taxonomy"
    RULES = "rules"
    ML = "ml"
    THREAT_INTEL = "threat_intel"


class _ScanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class ScanRequest(_ScanModel):
    """A single input, with OCR text accepted after screenshot review."""

    model_config = ConfigDict(val_json_bytes="base64", ser_json_bytes="base64")

    scan_type: _StringEnum[ScanType]
    text: _Text | None = Field(default=None, repr=False)
    url: _HttpUrl | None = Field(default=None, repr=False)
    image_bytes: _ImageBytes | None = Field(default=None, repr=False)
    sender_phone: _IndianMobile | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_input_kind(self) -> Self:
        provided = {
            name
            for name in ("text", "url", "image_bytes")
            if getattr(self, name) is not None
        }
        if len(provided) != 1:
            raise ValueError("Provide exactly one scan input.")
        allowed_inputs = {
            ScanType.URL: {"url"},
            ScanType.SMS: {"text"},
            ScanType.QR: {"image_bytes"},
            ScanType.SCREENSHOT: {"image_bytes", "text"},
        }
        if provided - allowed_inputs[self.scan_type]:
            raise ValueError("The input field does not match the scan type.")
        return self


class EvidenceItem(_ScanModel):
    source: _StringEnum[EvidenceSource]
    code: _Identifier
    summary: _Text = Field(max_length=500, repr=False)


class ScanIssue(_ScanModel):
    """Machine-readable explanation of an incomplete component."""

    component: _Identifier
    code: _Identifier


class _RiskAssessment(_ScanModel):
    risk_level: _StringEnum[RiskLevel] | None = None
    risk_score: _RiskScore | None = None

    @model_validator(mode="after")
    def validate_risk_pair(self) -> Self:
        if (self.risk_level is None) != (self.risk_score is None):
            raise ValueError("Risk level and score must be set together.")
        return self


class LinkReport(_RiskAssessment):
    url: _HttpUrl = Field(repr=False)
    evidence: tuple[EvidenceItem, ...] = ()


class UPIPayload(_ScanModel):
    """Locally decoded payment fields. The payee name is supplied by the QR."""

    pa: _UpiAddress = Field(repr=False)
    pn: _Text | None = Field(default=None, max_length=256, repr=False)
    am: Decimal | None = Field(default=None, ge=0, max_digits=12, decimal_places=2, repr=False)
    cu: Literal["INR"] | None = None
    mc: str | None = Field(default=None, pattern=r"^[0-9]{4}$")


class ScanReport(_RiskAssessment):
    scan_id: UUID = Field(default_factory=uuid4)
    scan_type: _StringEnum[ScanType]
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: _StringEnum[ScanStatus] = ScanStatus.COMPLETE
    scam_categories: tuple[_StringEnum[ScamCategory], ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    links: tuple[LinkReport, ...] = ()
    upi: UPIPayload | None = Field(default=None, repr=False)
    issues: tuple[ScanIssue, ...] = ()
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    feed_freshness: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        if (self.status is ScanStatus.PARTIAL) != bool(self.issues):
            raise ValueError("Only partial reports must include component issues.")
        return self
