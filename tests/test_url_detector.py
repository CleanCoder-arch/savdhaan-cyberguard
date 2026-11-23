import math
from pathlib import Path

import joblib
import pytest
from sklearn import __version__ as sklearn_version

from core.models._prediction import ModelArtifactError
from core.models.url_detector import URLDetector, extract_structural_features


_URLS = [
    "https://library.example.test/Agenda",
    "https://museum.example.test/Exhibitions",
    "https://school.example.test/Calendar",
    "http://verify-account-b6021.example.test/security/recover?token=example",
    "http://payroll-access-a903.example.test/claim/wallet?auth=example&continue=verify",
    "http://192.0.2.80:8080/payment/update/wallet?session_id=example",
]
_LABELS = [0, 0, 0, 1, 1, 1]


@pytest.fixture(scope="module")
def detector() -> URLDetector:
    return URLDetector().fit(_URLS, _LABELS)


def test_structural_features_have_stable_count_and_meaning() -> None:
    url = (
        "https://www.a0-b.xn--bcher-kva.test:443/Docs//account_A1.apk?"
        "token=a&token=b&next=https%3A%2F%2Fexample.test#Part"
    )
    features = extract_structural_features(url)
    assert len(features) == 40
    assert tuple(features) == tuple(extract_structural_features("http://192.0.2.5"))
    assert all(math.isfinite(value) for value in features.values())
    assert features["url_length"] == len(url)
    assert features["host_has_www_prefix"] == 1
    assert features["host_label_count"] == 4
    assert features["host_punycode_label_count"] == 1
    assert features["uses_https"] == 1
    assert features["has_explicit_port"] == 1
    assert features["has_nondefault_port"] == 0
    assert features["path_depth"] == 2
    assert features["path_has_repeated_slash"] == 1
    assert features["path_has_download_extension"] == 1
    assert features["query_parameter_count"] == 3
    assert features["query_distinct_key_count"] == 2
    assert features["query_token_like_key_count"] == 2
    assert features["query_has_nested_url"] == 1
    assert features["percent_escape_count"] == 3
    assert features["has_fragment"] == 1


@pytest.mark.parametrize(
    ("url", "ipv6", "nondefault_port"),
    [("http://192.0.2.1:8080", 0, 1), ("https://[2001:db8::1]:443/", 1, 0)],
)
def test_structural_features_handle_ip_literals(url: str, ipv6: int, nondefault_port: int) -> None:
    features = extract_structural_features(url)
    assert features["host_is_ip_literal"] == 1
    assert features["host_is_ipv6_literal"] == ipv6
    assert features["host_label_count"] == 0
    assert features["has_nondefault_port"] == nondefault_port


def test_lexical_features_preserve_path_case() -> None:
    urls = ["https://portal.test/Portal", "https://portal.test/portal"]
    assert extract_structural_features(urls[0]) == extract_structural_features(urls[1])
    detector = URLDetector().fit(urls, [0, 1])
    assert detector.predict(urls[0]).probability < detector.predict(urls[1]).probability


def test_model_uses_both_feature_families_and_explains_signed_log_odds(detector: URLDetector) -> None:
    prediction = detector.predict(_URLS[-2], top_k=None)
    assert detector.predict(_URLS[0]).probability < prediction.probability < 1
    assert {item.kind for item in prediction.contributions} == {"character_ngram", "structure"}
    assert prediction.intercept + sum(item.contribution for item in prediction.contributions) == pytest.approx(
        prediction.log_odds
    )
    assert 1 / (1 + math.exp(-prediction.log_odds)) == pytest.approx(prediction.probability)
    for item in prediction.contributions:
        assert item.contribution == pytest.approx(item.value * item.coefficient)
        if item.kind == "structure":
            assert item.raw_value == extract_structural_features(_URLS[-2])[item.feature]
        else:
            assert item.raw_value is None


def test_top_k_is_ranked_and_preserves_unshown_attribution(detector: URLDetector) -> None:
    full = detector.predict(_URLS[-1], top_k=None)
    short = detector.predict(_URLS[-1], top_k=3)
    ranks = [abs(feature.contribution) for feature in full.contributions]
    assert ranks == sorted(ranks, reverse=True)
    assert short.contributions == full.contributions[:3]
    assert short.probability == full.probability
    shown = sum(item.contribution for item in short.contributions)
    assert short.intercept + shown + short.unshown_contribution == pytest.approx(short.log_odds)
    assert detector.predict(_URLS[-1], top_k=0).contributions == ()


def test_query_and_fragment_contents_do_not_enter_character_ngrams() -> None:
    first = "https://sample.test/Agenda?token=QxZ7w#AbC"
    second = "https://sample.test/Agenda?token=ZyB7a#DeF"
    detector = URLDetector().fit([first, "https://other.test/verify/token"], [0, 1])
    assert detector.predict(first, top_k=None) == detector.predict(second, top_k=None)


@pytest.mark.parametrize("top_k", [True, -1, 1.5])
def test_invalid_top_k_is_rejected(detector: URLDetector, top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        detector.predict(_URLS[0], top_k=top_k)


@pytest.mark.parametrize("labels", [[0], [0, 0], [0, True], [0, 1.0], [0, -1]])
def test_training_requires_valid_binary_labels(labels: list[int]) -> None:
    with pytest.raises(ValueError):
        URLDetector().fit(_URLS[:2], labels)


def test_url_is_validated_in_training_and_inference(detector: URLDetector) -> None:
    malformed = "https://private-credential@example.test/Agenda"
    with pytest.raises(ValueError) as error:
        detector.predict(malformed)
    assert malformed not in str(error.value)
    with pytest.raises(ValueError):
        URLDetector().fit([_URLS[0], malformed], [0, 1])


def test_url_detector_requires_fitted_state(tmp_path: Path) -> None:
    detector = URLDetector()
    with pytest.raises(RuntimeError, match="Fit or load"):
        detector.predict(_URLS[0])
    with pytest.raises(RuntimeError, match="Fit or load"):
        detector.save(tmp_path / "model.joblib")


def test_url_detector_roundtrip_is_deterministic(detector: URLDetector, tmp_path: Path) -> None:
    path = tmp_path / "url.joblib"
    detector.save(path)
    loaded = URLDetector.load(path)
    assert loaded.predict(_URLS[-1], top_k=None) == detector.predict(_URLS[-1], top_k=None)
    retrained = URLDetector().fit(_URLS, _LABELS)
    assert retrained.predict(_URLS[0]) == detector.predict(_URLS[0])


def test_missing_corrupt_and_incompatible_artifacts_are_distinct(tmp_path: Path) -> None:
    path = tmp_path / "url.joblib"
    with pytest.raises(FileNotFoundError):
        URLDetector.load(path)
    path.write_bytes(b"not an artifact")
    with pytest.raises(ModelArtifactError, match="read"):
        URLDetector.load(path)
    joblib.dump({"artifact_version": -1}, path)
    with pytest.raises(ModelArtifactError, match="format"):
        URLDetector.load(path)
    joblib.dump({"artifact_version": 1, "sklearn_version": "old"}, path)
    with pytest.raises(ModelArtifactError, match="scikit-learn"):
        URLDetector.load(path)
    joblib.dump({"artifact_version": 1, "sklearn_version": sklearn_version, "detector": URLDetector()}, path)
    with pytest.raises(ModelArtifactError, match="no fitted model"):
        URLDetector.load(path)


def test_failed_save_keeps_existing_artifact(detector: URLDetector, tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "url.joblib"
    detector.save(path)
    original = path.read_bytes()

    def fail_after_write(model, temporary_path, *, compress):
        temporary_path.write_bytes(b"partial")
        raise OSError("write failed")

    monkeypatch.setattr(joblib, "dump", fail_after_write)
    with pytest.raises(OSError, match="write failed"):
        detector.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
