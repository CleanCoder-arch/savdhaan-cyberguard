import math
from pathlib import Path

import pytest

from core.models._prediction import ModelArtifactError
from core.models.sms_detector import SMSDetector, tokenize_sms
from core.models.url_detector import URLDetector


_MESSAGES = [
    "Library book renewal confirmed. Return your book at the counter.",
    "Running club practice is scheduled on Tuesday.",
    "Science class notes are ready to collect.",
    "Urgent KYC expiry. Verify your OTP to recover your blocked account.",
    "Your lottery prize expires. Pay a fee to unlock your money.",
    "Claim your reward by sharing an OTP and deposit a fee.",
]
_LABELS = [0, 0, 0, 1, 1, 1]


@pytest.fixture(scope="module")
def detector() -> SMSDetector:
    return SMSDetector().fit(_MESSAGES, _LABELS)


def test_tokenizer_uses_whole_words_with_unicode_marks() -> None:
    assert tokenize_sms("ОТП नहीं: पैसे, ＯＴＰ—code hotpot my_otp") == [
        "отп", "नहीं", "पैसे", "otp", "code", "hotpot", "my_otp"
    ]


def test_tokenizer_removes_recognized_pii_and_masks_numeric_tokens() -> None:
    tokens = tokenize_sms(
        "OTP: 123456. Contact sample@upi or https://example.test/?token=X. Order XZ44920"
    )
    assert "otp" in tokens
    assert "<number>" in tokens
    assert not {"123456", "sample", "upi", "example", "xz44920"} & set(tokens)


def test_predictions_use_word_features_with_signed_attribution(detector: SMSDetector) -> None:
    prediction = detector.predict(_MESSAGES[-1], top_k=None)
    assert detector.predict(_MESSAGES[0]).probability < prediction.probability < 1
    assert {feature.kind for feature in prediction.contributions} == {"token"}
    assert any(feature.feature == "otp" for feature in prediction.contributions)
    assert prediction.intercept + sum(feature.contribution for feature in prediction.contributions) == pytest.approx(
        prediction.log_odds
    )
    assert prediction.probability == pytest.approx(1 / (1 + math.exp(-prediction.log_odds)))
    assert all(
        feature.raw_value is None and feature.contribution == pytest.approx(feature.value * feature.coefficient)
        for feature in prediction.contributions
    )


def test_words_inside_other_tokens_do_not_contribute(detector: SMSDetector) -> None:
    for message in ("hotpot", "my_otp", "otps"):
        prediction = detector.predict(message, top_k=None)
        assert prediction.contributions == ()
        assert prediction.log_odds == prediction.intercept


def test_clean_multilingual_token_attribution() -> None:
    detector = SMSDetector().fit(["पुस्तक तैयार है", "तुरंत पैसे भेजो"], [0, 1])
    prediction = detector.predict("तुरंत पैसे भेजो", top_k=None)
    assert {feature.feature for feature in prediction.contributions} == {"तुरंत", "पैसे", "भेजो"}


def test_top_token_limit_and_neutral_input(detector: SMSDetector) -> None:
    full = detector.predict(_MESSAGES[-1], top_k=None)
    limited = detector.predict(_MESSAGES[-1], top_k=2)
    ranks = [abs(feature.contribution) for feature in full.contributions]
    assert ranks == sorted(ranks, reverse=True)
    assert limited.contributions == full.contributions[:2]
    shown = sum(item.contribution for item in limited.contributions)
    assert limited.intercept + shown + limited.unshown_contribution == pytest.approx(limited.log_odds)
    assert detector.predict("!? ₹", top_k=None).contributions == ()


@pytest.mark.parametrize(
    ("messages", "labels"),
    [
        ("text", [0, 1]),
        (["ok"], [0, 1]),
        (["hello", "hello again"], [0, 0]),
        (["hello", "hello again"], [0, "1"]),
        (["hello", "bad\x00input"], [0, 1]),
        (["...", "$!!"], [0, 1]),
    ],
)
def test_rejects_invalid_training_sets(messages, labels) -> None:
    with pytest.raises(ValueError):
        SMSDetector().fit(messages, labels)


def test_failed_fit_leaves_existing_detector_usable() -> None:
    detector = SMSDetector().fit(_MESSAGES, _LABELS)
    before = detector.predict(_MESSAGES[0])
    with pytest.raises(ValueError):
        detector.fit(["!!!", "..."], [0, 1])
    assert before == detector.predict(_MESSAGES[0])


def test_unfitted_and_malformed_inference(detector: SMSDetector, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Fit or load"):
        SMSDetector().predict("message")
    with pytest.raises(RuntimeError, match="Fit or load"):
        SMSDetector().save(tmp_path / "sms.joblib")
    with pytest.raises(ValueError):
        detector.predict(" \t ")
    with pytest.raises(ValueError, match="top_k"):
        detector.predict("message", top_k=-1)


def test_sms_detector_roundtrip_and_artifact_type_check(detector: SMSDetector, tmp_path: Path) -> None:
    path = tmp_path / "sms.joblib"
    detector.save(path)
    assert SMSDetector.load(path).predict(_MESSAGES[-1]) == detector.predict(_MESSAGES[-1])
    with pytest.raises(ModelArtifactError, match="different detector"):
        URLDetector.load(path)
