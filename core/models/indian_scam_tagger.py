"""Tag Indian scam archetypes from text using deterministic local rules."""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class ScamCategory(StrEnum):
    UPI_QR = "upi_qr"
    DIGITAL_ARREST = "digital_arrest"
    FAKE_KYC = "fake_kyc"
    JOB_TASK = "job_task"
    INVESTMENT = "investment"
    UTILITY = "utility"
    LOAN_APP = "loan_app"
    LOTTERY = "lottery"


@dataclass(frozen=True, slots=True)
class ScamTag:
    """A category and the stable identifiers of its matching rules."""

    category: ScamCategory
    rule_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Rule:
    category: ScamCategory
    rule_id: str
    patterns: tuple[re.Pattern[str], ...]


def _rule(category: ScamCategory, rule_id: str, *patterns: str) -> _Rule:
    return _Rule(
        category,
        rule_id,
        tuple(re.compile(rf"(?<!\w)(?:{pattern})(?!\w)") for pattern in patterns),
    )


_RECEIVE_MONEY = (
    r"(?:receive|claim|collect|get)\s+(?:(?:your|this|a|the)\s+)?"
    r"(?:money|payment|refund|cashback|prize|bonus|rewards?)"
)
_KYC = r"(?:re[-\s]?)?kyc|know\s+your\s+customer"
_INVESTMENT = r"invest(?:ments?|ing)?|trading|stocks?|crypto(?:currency)?|ipo|forex"
_LOAN = (
    r"loan\s+(?:app|approval|offer|disbursal|disbursement)|"
    r"(?:instant|personal|online)\s+loan"
)
_FEE_DEMAND = (
    r"(?:pay|deposit|transfer|submit)\s+(?:(?:a|an|the|your)\s+)?"
    r"(?:refundable\s+)?(?:registration|training|processing|joining|release|"
    r"security)\s+(?:fee|deposit|charge)"
)
_PRIZE = r"lottery|jackpot|sweepstakes|kbc(?:\s+lottery)?|(?:cash\s+)?prize"

_RULES = (
    _rule(
        ScamCategory.UPI_QR,
        "upi_qr.scan_to_receive",
        r"(?:scan|use)\s+(?:(?:this|the|our|a)\s+)?(?:upi\s+)?qr(?:\s+code)?",
        _RECEIVE_MONEY,
    ),
    _rule(
        ScamCategory.UPI_QR,
        "upi_qr.authorize_to_receive",
        r"(?:approve|accept)\s+(?:(?:your|this|the)\s+)?(?:upi\s+)?collect\s+request|"
        r"(?:enter|share|provide|send|confirm)\s+(?:(?:your|the)\s+)?upi[-\s]+pin",
        _RECEIVE_MONEY,
    ),
    _rule(
        ScamCategory.UPI_QR,
        "upi_qr.scan_to_receive_hinglish",
        r"qr(?:\s+code)?\s+scan\s+karo",
        r"paise\s+(?:milenge|lene)",
    ),
    _rule(
        ScamCategory.DIGITAL_ARREST,
        "digital_arrest.arrest_claim",
        r"(?:under|facing)\s+(?:a\s+)?digital[-\s]+arrest|digital[-\s]+arrest\s+(?:order|warrant)",
    ),
    _rule(
        ScamCategory.DIGITAL_ARREST,
        "digital_arrest.remote_investigation",
        r"police|cbi|ncb|customs|cyber\s+cell|enforcement\s+directorate",
        r"arrest|warrant|money\s+laundering|(?:illegal|drug)\s+parcel",
        r"video\s+call|skype|stay\s+on\s+(?:the\s+)?call|secure\s+account",
    ),
    _rule(
        ScamCategory.FAKE_KYC,
        "fake_kyc.suspension_link",
        _KYC,
        r"block(?:ed)?|suspend(?:ed)?|deactivat(?:e|ed)|freez(?:e|ing)|frozen|expir(?:e|ed|es)|locked",
        r"(?:click|tap|open|visit|follow)\s+(?:(?:the|this|our|below|your)\s+)?(?:link|url)",
    ),
    _rule(
        ScamCategory.FAKE_KYC,
        "fake_kyc.credential_request",
        _KYC,
        r"(?:share|send|provide|enter|confirm)\s+(?:(?:your|the)\s+)?(?:otp|upi[-\s]+pin|cvv)",
    ),
    _rule(
        ScamCategory.JOB_TASK,
        "job_task.paid_online_tasks",
        r"part[-\s]+time\s+(?:job|work)|online\s+(?:job|work)|work\s+from\s+home|review\s+tasks?",
        r"earn\s+(?:(?:₹|rs\.?|inr)\s*)?\d[\d,]*\s+(?:daily|per\s+day|a\s+day|per\s+task)|"
        r"(?:like|rate|review)\s+(?:(?:our|a|the|some)\s+)?(?:youtube\s+videos?|google\s+maps\s+listings?)|"
        r"(?:prepaid|recharge)\s+tasks?",
    ),
    _rule(
        ScamCategory.JOB_TASK,
        "job_task.advance_fee",
        r"job|interview|recruitment|hiring|employment|offer\s+letter",
        _FEE_DEMAND,
    ),
    _rule(
        ScamCategory.INVESTMENT,
        "investment.guaranteed_returns",
        _INVESTMENT,
        r"(?:guaranteed|assured|risk[-\s]+free)\s+"
        r"(?:\d{1,3}(?:\.\d+)?\s*%\s+)?(?:(?:daily|weekly|monthly|high|fixed)\s+)?"
        r"(?:returns?|profits?|gains?)",
    ),
    _rule(
        ScamCategory.INVESTMENT,
        "investment.multiply_money",
        r"(?:double|triple)\s+(?:(?:your|the)\s+)?(?:money|investment|capital)",
    ),
    _rule(
        ScamCategory.UTILITY,
        "utility.disconnection_demand",
        r"(?:electricity|power|water|gas|utility)\s+(?:connection|supply|service|bill)|electricity",
        r"disconnect(?:ed|ion)?|cut\s+off|suspended",
        r"(?:call|contact|whatsapp)\s+(?:(?:our|this|the)\s+)?(?:officer|agent|number)|"
        r"pay\s+(?:now|immediately|online|via)|"
        r"(?:update|clear|pay)\s+(?:(?:your|the)\s+)?(?:pending\s+)?bill",
    ),
    _rule(
        ScamCategory.LOAN_APP,
        "loan_app.advance_fee",
        _LOAN,
        _FEE_DEMAND,
    ),
    _rule(
        ScamCategory.LOAN_APP,
        "loan_app.contact_access",
        _LOAN,
        r"(?:access|upload|share)\s+(?:(?:your|the)\s+)?(?:contacts?|contact\s+list|gallery|photos?)",
    ),
    _rule(
        ScamCategory.LOTTERY,
        "lottery.prize_claim",
        r"you(?:['’]ve|\s+have)?\s+won|winner\s+of|selected\s+for",
        _PRIZE,
    ),
    _rule(
        ScamCategory.LOTTERY,
        "lottery.advance_fee",
        _PRIZE,
        _FEE_DEMAND,
    ),
)


def tag_indian_scams(text: str) -> tuple[ScamTag, ...]:
    """Return matching categories in taxonomy order with all matching rule IDs."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    matches: dict[ScamCategory, list[str]] = {}
    for rule in _RULES:
        if all(pattern.search(normalized) for pattern in rule.patterns):
            matches.setdefault(rule.category, []).append(rule.rule_id)

    return tuple(
        ScamTag(category, tuple(matches[category]))
        for category in ScamCategory
        if category in matches
    )
