from decimal import Decimal

import pytest

from core.schemas import UPIPayload
from core.scoring.rules_engine import NEW_DOMAIN_DAYS, RuleHit, evaluate_rules


def test_message_rules_return_stable_explanations_without_echoing_input() -> None:
    private_credential = "123456"
    text = (
        "You are under digital arrest. Final warning: send your OTP immediately. "
        f"Enter your password too. Your reference is {private_credential}."
    )
    hits = evaluate_rules(text=text)
    assert [hit.code for hit in hits] == [
        "message.urgency",
        "message.credentials_requested",
        "message.digital_arrest",
    ]
    assert [hit.points for hit in hits] == [8, 18, 22]
    assert private_credential not in repr(hits)


@pytest.mark.parametrize(
    "text",
    [
        "We saved the agenda for our meeting.",
        "The insurgent group was mentioned in the article.",
        "Never share your OTP with anyone.",
        "Don't scan the QR code to receive money.",
        "Do not enter your UPI PIN to receive a refund.",
        "Scan this QR code to pay for your order.",
        "Send your full name to the organizer.",
    ],
)
def test_common_neutral_messages_and_warnings_have_no_rule_hits(text: str) -> None:
    assert evaluate_rules(text=text) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Final warning. Open your account.",
        "Submit the renewal within 15 minutes.",
        "Complete verification ＩＭＭＥＤＩＡＴＥＬＹ.",
    ],
)
def test_urgency_normalizes_case_width_and_words(text: str) -> None:
    assert [hit.code for hit in evaluate_rules(text=text)] == ["message.urgency"]


def test_fixed_upi_amount_is_attributed_once_in_receive_money_context() -> None:
    upi = UPIPayload(pa="private-example@upi", am=Decimal("100.00"), pn="Private Example")
    hits = evaluate_rules(text="Scan this QR code to receive your refund.", upi=upi)
    assert [hit.code for hit in hits] == ["upi.payment_to_receive"]
    assert hits[0].points == 28
    assert upi.pa not in repr(hits)
    assert upi.pn not in repr(hits)


def test_dynamic_upi_qr_alone_is_not_a_scam_signal() -> None:
    upi = UPIPayload(pa="seller@bank", am=Decimal("80"))
    assert evaluate_rules(upi=upi) == ()
    assert evaluate_rules(text="Scan this code to pay for your order.", upi=upi) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Scan this QR code to receive your refund.",
        "Approve the UPI collect request to receive a payment.",
        "QR scan karo, paise milenge.",
    ],
)
def test_upi_request_patterns_are_one_signal_without_a_fixed_amount(text: str) -> None:
    assert [hit.code for hit in evaluate_rules(text=text)] == ["upi.receive_request"]


@pytest.mark.parametrize(
    "url",
    [
        "https://sbi.bank.in/NetBanking",
        "https://corporate.sbi/",
        "https://bank.sbi/",
        "https://www.sbi.co.in./",
        "https://netbanking.hdfc.bank.in/",
        "https://netbanking.hdfcbank.com/",
        "https://icici.bank.in/",
        "https://www.icicibank.com/",
        "https://ihdfc.example.test/",
        "https://example.test/claims/SBI",
    ],
)
def test_bank_rule_honors_real_dns_boundaries_and_brand_tokens(url: str) -> None:
    assert evaluate_rules(urls=(url,)) == ()


@pytest.mark.parametrize(
    "url",
    [
        "https://sbi.bank.in.example.test/",
        "https://hdfc-verify.example.test/",
        "https://secure.icicibank.example.test/",
    ],
)
def test_bank_rule_detects_brand_tokens_on_outside_hostname(url: str) -> None:
    assert [hit.code for hit in evaluate_rules(urls=(url,))] == ["url.bank_name_mismatch"]


def test_url_rules_deduplicate_and_do_not_leak_original_links() -> None:
    urls = (
        "http://127.0.0.1/path?token=secret",
        "http://[2001:db8::1]/",
        "https://sbi-reset.example.test/",
        "https://hdfc-reset.example.test/",
    )
    hits = evaluate_rules(urls=iter(urls))
    assert [hit.code for hit in hits] == [
        "url.unencrypted_http",
        "url.ip_literal",
        "url.bank_name_mismatch",
    ]
    assert all(url not in repr(hits) for url in urls)
    assert "secret" not in repr(hits)


@pytest.mark.parametrize("age", [None, NEW_DOMAIN_DAYS, NEW_DOMAIN_DAYS + 365])
def test_missing_or_older_registration_does_not_add_points(age: int | None) -> None:
    assert evaluate_rules(domain_age_days=age) == ()


@pytest.mark.parametrize("age", [0, NEW_DOMAIN_DAYS - 1])
def test_verified_young_registration_adds_points_once(age: int) -> None:
    assert [hit.code for hit in evaluate_rules(domain_age_days=age)] == ["domain.recent_registration"]


@pytest.mark.parametrize("age", [-1, True, 0.5, "30"])
def test_invalid_domain_age_fails_explicitly(age: object) -> None:
    with pytest.raises(ValueError, match="Domain age"):
        evaluate_rules(domain_age_days=age)


@pytest.mark.parametrize("points", [-101, 101, False, 2.5, "10"])
def test_rule_weights_are_bounded_integers(points: object) -> None:
    with pytest.raises(ValueError, match="Rule points"):
        RuleHit("example.invalid", points, "Synthetic rule.")
