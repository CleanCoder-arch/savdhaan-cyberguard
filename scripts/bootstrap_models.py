"""Train local baseline models on bundled, synthetic examples."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import sklearn

from core.models.artifacts import DEFAULT_MODEL_DIR, SMS_MODEL_FILENAME, URL_MODEL_FILENAME
from core.models.sms_detector import SMSDetector
from core.models.url_detector import URLDetector

_URL_EXAMPLES = (
    (0, "https://library.test/Agenda"),
    (0, "https://docs.garden.test/Help/Payments"),
    (0, "https://campus.test/admissions/status"),
    (0, "https://metro.test/travel/route-map"),
    (0, "https://store.test/orders/returns?language=en"),
    (0, "https://jobs.studio.test/careers"),
    (0, "https://support.credit-union.test/contact"),
    (0, "https://energy.test/my-account/bill-history"),
    (0, "https://civic.test/permits/renewal"),
    (0, "https://www.artwalk.test/events/tickets"),
    (0, "https://portal.bookshop.test/sign-in"),
    (0, "https://water.test/billing/payment-options"),
    (0, "https://updates.museum.test/news/collection"),
    (0, "https://shop.test/checkout/receipt?source=email"),
    (0, "https://school.test/student/verify-email"),
    (0, "https://cooperative.test/fraud-awareness/upi"),
    (1, "http://bank-kyc-update-login.test/secure/verify-account"),
    (1, "https://claim-quick-refund-now.test/upi/collect-prize"),
    (1, "http://wallet-reactivation.portal-case.test/Verify/Netbanking"),
    (1, "https://service-billing.example-update.test/immediate-disconnect/pay-now"),
    (1, "http://earn-task-commission.test/recharge/unlock-withdrawal"),
    (1, "https://reward-bonus-audit.test/wallet/claim-cashback?offer=DEMO"),
    (1, "http://cbi-video-investigation.case-portal.test/digital-arrest/verify"),
    (1, "https://kyc-reactivate.payments-helpdesk.test/session/update"),
    (1, "http://loan-approved-fast.test/processing-fee/unlock"),
    (1, "https://security-alert.account-review.test/confirm-pan/urgent"),
    (1, "https://bonus-lottery-winner.test/reward/claim-prize"),
    (1, "http://192.0.2.20:8080/secure/verify-wallet"),
    (1, "https://partner-bank.test.auth-notice.test/blocked/login-update"),
    (1, "http://investment-guarantee-return.test/vip/deposit"),
    (1, "https://account-suspended.test/%76erify/mobile-update"),
    (1, "https://payment-collect-refund.test/claim?campaign=EXAMPLE"),
)
_SMS_EXAMPLES = (
    (0, "Your library book is due on Friday. Renew through the library catalogue."),
    (0, "We received your application. You can view its status in the student portal."),
    (0, "An electricity supply outage is planned for maintenance tomorrow morning."),
    (0, "The UPI transfer was successful. Check your statement for the receipt."),
    (0, "Your appointment is confirmed. Please bring a copy of your identification."),
    (0, "Never disclose your OTP or UPI PIN to customer support."),
    (0, "Your order has shipped. You can see delivery updates in the store app."),
    (0, "View your current balance and account statement in the official banking app."),
    (0, "Read the investment factsheet and consider the risk of losing money."),
    (0, "Verify your email by opening the account settings page."),
    (0, "The company posted a part-time job on its careers page."),
    (0, "You can update your KYC information at the local branch."),
    (0, "The approved loan terms are available in your account. Review the repayment schedule."),
    (0, "The lottery results were published yesterday on the event noticeboard."),
    (0, "कल बैंक शाखा खुली रहेगी। सहायता के लिए शाखा से संपर्क करें।"),
    (0, "Your meter reading was recorded. The bill will appear next week."),
    (1, "Your KYC has expired. Click the link to avoid account suspension and enter your OTP."),
    (1, "Scan the UPI QR code to receive a refund. Approve the collect request now."),
    (1, "You are under digital arrest. Stay on the video call and transfer money to a secure account."),
    (1, "Part-time online job. Pay a registration fee now to unlock daily task commissions."),
    (1, "Get guaranteed weekly returns on crypto trading. Deposit into the VIP group today."),
    (1, "Electricity will be disconnected tonight. Contact this officer and pay immediately."),
    (1, "Instant loan approved. Pay the processing fee and share your contact list before disbursal."),
    (1, "You have won the lottery. Send the release fee to claim your cash prize."),
    (1, "Account suspended. Verify your PAN and share the security code to reactivate it."),
    (1, "Recharge your task balance before you can withdraw your earnings."),
    (1, "Claim guaranteed cashback. Enter your UPI PIN to receive the money."),
    (1, "Customs seized an illegal parcel. Remain on the video call until you transfer funds."),
    (1, "QR scan karo, paise milenge. Collect request abhi approve karo."),
    (1, "Confirm your online banking credentials before the account is frozen."),
    (1, "You have been selected for a prize. Deposit a refundable processing charge."),
    (1, "अपना खाता बंद होने से बचाएं। तुरंत अपना ओटीपी भेजें।"),
)


def bootstrap_models(output_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Fit both models and publish the artifacts and their provenance."""
    urls, url_labels = zip(*((text, label) for label, text in _URL_EXAMPLES), strict=True)
    messages, sms_labels = zip(*((text, label) for label, text in _SMS_EXAMPLES), strict=True)
    url_model = URLDetector().fit(urls, url_labels)
    sms_model = SMSDetector().fit(messages, sms_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = "manifest.json"
    training_data = json.dumps((_URL_EXAMPLES, _SMS_EXAMPLES), ensure_ascii=False).encode("utf-8")
    manifest = {
        "training_data": "synthetic_baseline",
        "corpus_sha256": hashlib.sha256(training_data).hexdigest(),
        "scikit_learn_version": sklearn.__version__,
        "url": {"file": URL_MODEL_FILENAME, "sample_count": len(urls)},
        "sms": {"file": SMS_MODEL_FILENAME, "sample_count": len(messages)},
    }
    with TemporaryDirectory(dir=output_dir) as staging:
        staged = Path(staging)
        url_model.save(staged / URL_MODEL_FILENAME)
        sms_model.save(staged / SMS_MODEL_FILENAME)
        (staged / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in (URL_MODEL_FILENAME, SMS_MODEL_FILENAME, manifest_name):
            os.replace(staged / filename, output_dir / filename)
    return output_dir / manifest_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    path = bootstrap_models(args.output_dir)
    print(f"Trained both offline baseline models. Manifest written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
