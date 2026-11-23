import pytest

from core.models.indian_scam_tagger import ScamCategory, ScamTag, tag_indian_scams


@pytest.mark.parametrize(
    ("text", "category", "rule_id"),
    [
        (
            "Scan this QR code to receive your refund.",
            ScamCategory.UPI_QR,
            "upi_qr.scan_to_receive",
        ),
        (
            "Approve the UPI collect request to receive a payment.",
            ScamCategory.UPI_QR,
            "upi_qr.authorize_to_receive",
        ),
        (
            "QR scan karo, paise milenge.",
            ScamCategory.UPI_QR,
            "upi_qr.scan_to_receive_hinglish",
        ),
        (
            "The caller claims you are under digital arrest.",
            ScamCategory.DIGITAL_ARREST,
            "digital_arrest.arrest_claim",
        ),
        (
            "The CBI claims there is a money laundering warrant. Stay on the call.",
            ScamCategory.DIGITAL_ARREST,
            "digital_arrest.remote_investigation",
        ),
        (
            "Your KYC expires today. Click the link to keep the account active.",
            ScamCategory.FAKE_KYC,
            "fake_kyc.suspension_link",
        ),
        (
            "For re-KYC, enter your OTP.",
            ScamCategory.FAKE_KYC,
            "fake_kyc.credential_request",
        ),
        (
            "Part-time job. Earn ₹2,500 per day for online reviews.",
            ScamCategory.JOB_TASK,
            "job_task.paid_online_tasks",
        ),
        (
            "Your interview is confirmed once you pay the refundable registration fee.",
            ScamCategory.JOB_TASK,
            "job_task.advance_fee",
        ),
        (
            "Crypto trading with guaranteed 25% weekly returns.",
            ScamCategory.INVESTMENT,
            "investment.guaranteed_returns",
        ),
        (
            "Double your money within a week.",
            ScamCategory.INVESTMENT,
            "investment.multiply_money",
        ),
        (
            "Your electricity connection will be disconnected tonight. Contact this number.",
            ScamCategory.UTILITY,
            "utility.disconnection_demand",
        ),
        (
            "You have loan approval. Pay the processing fee before disbursement.",
            ScamCategory.LOAN_APP,
            "loan_app.advance_fee",
        ),
        (
            "For an instant loan, let the app access your contact list.",
            ScamCategory.LOAN_APP,
            "loan_app.contact_access",
        ),
        (
            "You have won ₹25 lakh in the KBC lottery.",
            ScamCategory.LOTTERY,
            "lottery.prize_claim",
        ),
        (
            "To release the lottery prize, pay the processing fee.",
            ScamCategory.LOTTERY,
            "lottery.advance_fee",
        ),
    ],
)
def test_tags_each_rule(text: str, category: ScamCategory, rule_id: str) -> None:
    assert tag_indian_scams(text) == (ScamTag(category, (rule_id,)),)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  \t\n",
        "Scan our QR code to pay for your order.",
        "A video call with the police is scheduled for tomorrow morning.",
        "To update your KYC documents, visit your bank branch.",
        "Apply for our part-time job on the company careers page.",
        "Read about investing and the risk of losing money.",
        "Electricity service will be temporarily disconnected for planned maintenance.",
        "A loan offer is visible in the bank app.",
        "The lottery results are now available.",
        "Scan the QRTABLE to receive money.",
        "MYKYC expired, click the link.",
        "Crypto trading with guaranteed nonprofit gains discussions.",
        "You have won a prizewinner badge.",
    ],
)
def test_requires_complete_words_and_scam_cues(text: str) -> None:
    assert tag_indian_scams(text) == ()


def test_matches_case_whitespace_and_compatible_unicode() -> None:
    assert tag_indian_scams("SCAN\tthis\nＱＲ code to receive MONEY!") == (
        ScamTag(ScamCategory.UPI_QR, ("upi_qr.scan_to_receive",)),
    )


def test_returns_all_categories_and_rule_ids_in_stable_order() -> None:
    messages = (
        "You have won a lottery prize. Pay the processing fee. "
        "Approve the UPI collect request to get your prize. "
        "Scan this QR code to claim your prize. "
        "For KYC enter your OTP. Scan this QR code to claim your prize."
    )

    assert tag_indian_scams(messages) == (
        ScamTag(ScamCategory.UPI_QR, ("upi_qr.scan_to_receive", "upi_qr.authorize_to_receive")),
        ScamTag(ScamCategory.FAKE_KYC, ("fake_kyc.credential_request",)),
        ScamTag(ScamCategory.LOTTERY, ("lottery.prize_claim", "lottery.advance_fee")),
    )


def test_explanations_use_rule_ids_without_input_text() -> None:
    private_input = "user@example.test"
    text = f"SCAN this QR code to claim a refund for {private_input}."

    assert tag_indian_scams(text) == (
        ScamTag(ScamCategory.UPI_QR, ("upi_qr.scan_to_receive",)),
    )
    assert private_input not in repr(tag_indian_scams(text))
