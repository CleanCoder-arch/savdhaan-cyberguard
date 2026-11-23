"""Official reporting contacts and consumer guidance for scams in India."""

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping

from core.models.indian_scam_tagger import ScamCategory

CYBER_FRAUD_HELPLINE = "1930"
CYBERCRIME_PORTAL = "https://cybercrime.gov.in/"
I4C_REPORTING_SOURCE = "https://ncrp-grievanceredressal.mha.gov.in/"
RBI_COMMERCIAL_BANK_DIRECTIONS = "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=13140"
RBI_COMMERCIAL_BANK_AMENDMENT = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=13543"


@dataclass(frozen=True, slots=True)
class BankHelpline:
    bank_name: str
    phone_numbers: tuple[str, ...]
    source_url: str


BANK_HELPLINES: Mapping[str, BankHelpline] = MappingProxyType({
    "sbi": BankHelpline(
        "State Bank of India", ("1800 11 1109",), "https://sbi.bank.in/web/customer-care/contact-us"
    ),
    "hdfc": BankHelpline(
        "HDFC Bank",
        ("1800 1600", "1800 2600"),
        "https://www.hdfc.bank.in/contact-us/unauthorized-electronic-banking",
    ),
    "icici": BankHelpline(
        "ICICI Bank", ("1800 2662",), "https://www.icici.bank.in/customer-care"
    ),
})


def get_bank_helpline(bank_name: str) -> BankHelpline | None:
    """Resolve a directory key or the bank's official name."""
    name = " ".join(bank_name.casefold().split())
    if name in BANK_HELPLINES:
        return BANK_HELPLINES[name]
    return next((bank for bank in BANK_HELPLINES.values() if bank.bank_name.casefold() == name), None)


REPORTING_STEPS = (
    "If you lost money or see an unfamiliar debit, contact your bank or payment provider immediately "
    "using its official app, website, or the number on your card. Ask it to block any affected payment access.",
    f"For financial cyber fraud, call {CYBER_FRAUD_HELPLINE} promptly and file a complaint at {CYBERCRIME_PORTAL}",
    "Keep the message, screenshots, transaction references, and complaint acknowledgements for official reporting.",
)


@dataclass(frozen=True, slots=True)
class ScamPlaybook:
    title: str
    actions: tuple[str, ...]
    source_urls: tuple[str, ...]


_CATEGORY_ADVICE = {
    ScamCategory.UPI_QR: (
        "UPI / QR scam",
        "Reject unfamiliar collect requests and requests to scan a payment QR code to receive money. "
        "Only enter a UPI PIN to approve an intended payment.",
        "https://www.npci.org.in/fraud-awareness",
    ),
    ScamCategory.DIGITAL_ARREST: (
        "Digital arrest scam",
        "End the call. Call the named agency on a number you find independently before sharing information or sending money.",
        "https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=2020570&lang=2&reg=48",
    ),
    ScamCategory.FAKE_KYC: (
        "Fake KYC scam",
        "Open the bank's app or visit its branch to check your KYC status. Keep OTPs and identity documents away from the sender.",
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=57244",
    ),
    ScamCategory.JOB_TASK: (
        "Job / task scam",
        "Verify the role through the employer's own careers page. Decline recruitment fees and deposits to unlock payments.",
        None,
    ),
    ScamCategory.INVESTMENT: (
        "Investment scam",
        "Hold off on deposits. Check securities advisers and platforms against official SEBI records before investing.",
        "https://investor.sebi.gov.in/spot-any-scam.html",
    ),
    ScamCategory.UTILITY: (
        "Utility scam",
        "Check the bill in your utility provider's official app or website. Use the phone number printed on a previous bill.",
        None,
    ),
    ScamCategory.LOAN_APP: (
        "Loan app scam",
        "Check the lender and confirm the app on that lender's official site before sharing KYC details or making a payment.",
        "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=12848&Mode=0",
    ),
    ScamCategory.LOTTERY: (
        "Lottery scam",
        "Ignore requests to pay a release fee or share bank details for a prize you did not enter to win.",
        None,
    ),
}


def get_scam_playbook(category: ScamCategory | None = None) -> ScamPlaybook:
    """Return category-specific advice with official reporting steps."""
    if category is None:
        return ScamPlaybook("Suspected scam", REPORTING_STEPS, (I4C_REPORTING_SOURCE,))
    title, action, source = _CATEGORY_ADVICE[category]
    sources = (I4C_REPORTING_SOURCE, source) if source else (I4C_REPORTING_SOURCE,)
    return ScamPlaybook(title, (action, *REPORTING_STEPS), sources)


@dataclass(frozen=True, slots=True)
class LiabilityGuidance:
    scope: str
    rules: tuple[str, ...]
    source_url: str


_COMMERCIAL_BANK_SCOPE = "Commercial banks covered by RBI's Commercial Banks Responsible Business Conduct Directions."
_AMENDMENT_EFFECTIVE = date(2027, 1, 1)
_COMPENSATION_END = date(2028, 1, 1)
_BANK_FAULT = (
    "The customer has zero liability where the bank's fraud, negligence, or deficient controls caused the loss. "
    "This applies even if the customer has yet to report the transaction."
)

_PRIOR_LIABILITY = LiabilityGuidance(
    _COMMERCIAL_BANK_SCOPE,
    (
        _BANK_FAULT,
        "For an unauthorised transaction caused by a third-party breach where neither you nor your bank is responsible, "
        "you have zero liability if you notify the bank within three working days after receiving the bank's alert. "
        "Count the working days of your home branch, starting after the day you received the alert.",
        "If you report such a third-party breach in four to seven working days, your liability cannot exceed the lower of "
        "the transaction amount and the RBI's account-specific cap. After seven working days the bank's policy applies.",
        "If sharing payment credentials or other customer negligence caused the loss, the customer bears losses up to "
        "the time of reporting. The bank bears losses from unauthorised transactions after the report.",
    ),
    RBI_COMMERCIAL_BANK_DIRECTIONS,
)
_AMENDED_RULES = (
    _BANK_FAULT,
    "For an unauthorised transaction caused by a third-party breach, the customer has zero liability if the bank "
    "receives the report within five calendar days of the transaction. After that, the bank's policy applies.",
    "If customer negligence caused the loss, liability before reporting is subject to any eligible compensation. "
    "The bank bears losses from unauthorised transactions after the report.",
)
_SMALL_LOSS_RULE = (
    "For eligible customer-negligence fraud involving gross losses up to INR 50,000, an individual may receive a "
    "once-in-a-lifetime payment of 85% of net loss up to INR 25,000. The bank must verify eligibility, including that "
    "you reported to the bank and filed a cybercrime complaint through 1930 or the national portal within five calendar days."
)


def get_rbi_liability_guidance(transaction_date: date) -> LiabilityGuidance:
    """Summarize the cited commercial bank framework for the transaction date."""
    if transaction_date < _AMENDMENT_EFFECTIVE:
        return _PRIOR_LIABILITY
    rules = _AMENDED_RULES
    if transaction_date < _COMPENSATION_END:
        rules += (_SMALL_LOSS_RULE,)
    return LiabilityGuidance(_COMMERCIAL_BANK_SCOPE, rules, RBI_COMMERCIAL_BANK_AMENDMENT)
