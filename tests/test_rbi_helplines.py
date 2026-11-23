from datetime import date
from urllib.parse import urlsplit

import pytest

from core.guidance.rbi_helplines import (
    BANK_HELPLINES,
    CYBER_FRAUD_HELPLINE,
    CYBERCRIME_PORTAL,
    I4C_REPORTING_SOURCE,
    RBI_COMMERCIAL_BANK_AMENDMENT,
    RBI_COMMERCIAL_BANK_DIRECTIONS,
    REPORTING_STEPS,
    get_bank_helpline,
    get_rbi_liability_guidance,
    get_scam_playbook,
)
from core.models.indian_scam_tagger import ScamCategory


@pytest.mark.parametrize(
    ("category", "advice"),
    [
        (ScamCategory.UPI_QR, "UPI PIN"),
        (ScamCategory.DIGITAL_ARREST, "End the call"),
        (ScamCategory.FAKE_KYC, "check your KYC status"),
        (ScamCategory.JOB_TASK, "careers page"),
        (ScamCategory.INVESTMENT, "SEBI"),
        (ScamCategory.UTILITY, "previous bill"),
        (ScamCategory.LOAN_APP, "lender's official site"),
        (ScamCategory.LOTTERY, "release fee"),
    ],
)
def test_category_playbook_includes_tailored_action_and_reporting(category, advice):
    playbook = get_scam_playbook(category)
    assert playbook.title.endswith("scam")
    assert advice in playbook.actions[0]
    assert playbook.actions[1:] == REPORTING_STEPS
    assert playbook.source_urls[0] == I4C_REPORTING_SOURCE
    assert all(urlsplit(source).scheme == "https" for source in playbook.source_urls)


def test_generic_playbook_identifies_bank_helpline_and_portal():
    playbook = get_scam_playbook()
    assert playbook.title == "Suspected scam"
    assert playbook.actions == REPORTING_STEPS
    assert "contact your bank" in playbook.actions[0]
    assert CYBER_FRAUD_HELPLINE == "1930"
    assert CYBER_FRAUD_HELPLINE in playbook.actions[1]
    assert urlsplit(CYBERCRIME_PORTAL).hostname == "cybercrime.gov.in"
    assert CYBERCRIME_PORTAL in playbook.actions[1]


@pytest.mark.parametrize(
    ("bank", "key", "numbers", "source_host"),
    [
        (" SBI ", "sbi", ("1800 11 1109",), "sbi.bank.in"),
        ("hDfC  Bank", "hdfc", ("1800 1600", "1800 2600"), "www.hdfc.bank.in"),
        ("ICICI", "icici", ("1800 2662",), "www.icici.bank.in"),
    ],
)
def test_directory_resolves_official_name_or_key(bank, key, numbers, source_host):
    helpline = get_bank_helpline(bank)
    assert helpline == BANK_HELPLINES[key]
    assert helpline.phone_numbers == numbers
    assert urlsplit(helpline.source_url).hostname == source_host


@pytest.mark.parametrize("bank", ["", "Unlisted Bank", "SBI Support: 12345"])
def test_unknown_bank_has_no_directory_match(bank):
    assert get_bank_helpline(bank) is None


def test_existing_rbi_timing_depends_on_bank_communication_and_responsibility():
    guidance = get_rbi_liability_guidance(date(2026, 12, 31))
    assert guidance.source_url == RBI_COMMERCIAL_BANK_DIRECTIONS
    assert guidance.scope.startswith("Commercial banks")
    assert "bank's fraud, negligence, or deficient controls" in guidance.rules[0]
    assert "three working days" in guidance.rules[1]
    assert "bank's alert" in guidance.rules[1]
    assert "home branch" in guidance.rules[1]
    assert "four to seven working days" in guidance.rules[2]
    assert "account-specific cap" in guidance.rules[2]
    assert "sharing payment credentials" in guidance.rules[3]
    assert "after the report" in guidance.rules[3]


def test_final_rbi_amendment_uses_transaction_date_and_conditional_compensation():
    guidance = get_rbi_liability_guidance(date(2027, 1, 1))
    assert guidance.source_url == RBI_COMMERCIAL_BANK_AMENDMENT
    assert "five calendar days of the transaction" in guidance.rules[1]
    assert "third-party breach" in guidance.rules[1]
    assert "INR 50,000" in guidance.rules[-1]
    assert "85% of net loss up to INR 25,000" in guidance.rules[-1]
    assert "once-in-a-lifetime" in guidance.rules[-1]
    assert "the bank and filed a cybercrime complaint" in guidance.rules[-1]


def test_original_compensation_window_is_time_limited():
    assert len(get_rbi_liability_guidance(date(2027, 12, 31)).rules) == 4
    guidance = get_rbi_liability_guidance(date(2028, 1, 1))
    assert guidance.source_url == RBI_COMMERCIAL_BANK_AMENDMENT
    assert len(guidance.rules) == 3
