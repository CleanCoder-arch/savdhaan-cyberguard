"""Offline phishing URL classification with character n-grams and structural features."""

import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Self
from urllib.parse import parse_qsl, urlsplit

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.guidance.redactor import redact_text
from core.models._prediction import (
    ModelArtifactError,
    explain_prediction,
    load_artifact,
    save_artifact,
    training_data,
)
from core.models.prediction import FeatureKind, ModelPrediction
from core.validators import validate_url

_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
_EXTENSION = re.compile(r"\.([A-Za-z0-9]{1,12})$")
_DOWNLOAD_EXTENSIONS = frozenset({"apk", "exe", "msi", "bat", "cmd", "ps1", "dmg"})
_TOKEN_KEYS = frozenset(
    {"access_token", "api_key", "auth", "code", "key", "nonce", "session", "session_id", "sig", "token"}
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _entropy(value: str) -> float:
    frequencies = (count / len(value) for count in Counter(value).values())
    return -math.fsum(frequency * math.log2(frequency) for frequency in frequencies)


def extract_structural_features(url: str) -> dict[str, float]:
    """Return 40 named, fixed-order lexical URL features without resolving the host."""
    parts = urlsplit(validate_url(url))
    hostname = (parts.hostname or "").removesuffix(".")
    host_labels = hostname.split(".")
    host_digits = sum(character.isdecimal() for character in hostname)
    path = parts.path
    segments = tuple(segment for segment in path.split("/") if segment)
    extension = _EXTENSION.search(segments[-1]) if segments else None
    parameters = parse_qsl(parts.query, keep_blank_values=True)
    https = parts.scheme == "https"
    try:
        address = ip_address(hostname)
    except ValueError:
        address = None

    return {
        "url_length": len(url),
        "host_length": len(hostname),
        "path_length": len(path),
        "query_length": len(parts.query),
        "fragment_length": len(parts.fragment),
        "host_label_count": len(host_labels) if address is None else 0,
        "host_dot_count": hostname.count("."),
        "host_hyphen_count": hostname.count("-"),
        "host_digit_count": host_digits,
        "host_digit_ratio": _ratio(host_digits, len(hostname)),
        "host_max_label_length": max(map(len, host_labels)) if address is None else 0,
        "host_mean_label_length": _ratio(sum(map(len, host_labels)), len(host_labels)) if address is None else 0.0,
        "host_has_non_ascii": int(not hostname.isascii()),
        "host_punycode_label_count": sum(label.startswith("xn--") for label in host_labels),
        "host_is_ip_literal": int(address is not None),
        "host_is_ipv6_literal": int(address is not None and address.version == 6),
        "host_has_www_prefix": int(hostname.startswith("www.")),
        "uses_https": int(https),
        "has_explicit_port": int(parts.port is not None),
        "has_nondefault_port": int(parts.port not in (None, 443 if https else 80)),
        "path_depth": len(segments),
        "path_max_segment_length": max(map(len, segments), default=0),
        "path_dot_count": path.count("."),
        "path_hyphen_count": path.count("-"),
        "path_underscore_count": path.count("_"),
        "path_digit_ratio": _ratio(sum(character.isdecimal() for character in path), len(path)),
        "path_has_repeated_slash": int("//" in path),
        "path_has_file_extension": int(extension is not None),
        "path_has_download_extension": int(
            extension is not None and extension[1].lower() in _DOWNLOAD_EXTENSIONS
        ),
        "percent_escape_count": len(_PERCENT_ESCAPE.findall(url)),
        "query_parameter_count": len(parameters),
        "query_distinct_key_count": len({key for key, _ in parameters}),
        "query_max_value_length": max((len(value) for _, value in parameters), default=0),
        "query_has_nested_url": int(any(value.lower().startswith(("http://", "https://")) for _, value in parameters)),
        "query_token_like_key_count": sum(key.casefold() in _TOKEN_KEYS for key, _ in parameters),
        "has_fragment": int(bool(parts.fragment)),
        "url_non_ascii_count": sum(not character.isascii() for character in url),
        "url_punctuation_ratio": _ratio(sum(not character.isalnum() for character in url), len(url)),
        "host_entropy": _entropy(hostname),
        "path_entropy": _entropy(path),
    }


def _lexical_url(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.lower()}{redact_text(parts.path)}"


@dataclass(frozen=True, slots=True)
class _URLModel:
    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    classifier: LogisticRegression
    features: tuple[tuple[FeatureKind, str], ...]


class URLDetector:
    """Hybrid TF-IDF and structural logistic regression for positive class 1."""

    def __init__(self) -> None:
        self._model: _URLModel | None = None

    def fit(self, urls: Iterable[str], labels: Iterable[int]) -> Self:
        records, targets = training_data(urls, labels, validate_url)
        indicators = [extract_structural_features(url) for url in records]
        structure_rows = [tuple(row.values()) for row in indicators]
        scaler = StandardScaler()
        structure = csr_matrix(scaler.fit_transform(structure_rows))
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, sublinear_tf=True)
        ngrams = vectorizer.fit_transform(_lexical_url(url) for url in records)
        matrix = hstack((structure, ngrams), format="csr")
        classifier = LogisticRegression(max_iter=1000, random_state=0).fit(matrix, targets)
        features = (
            *(("structure", name) for name in indicators[0]),
            *(("character_ngram", term) for term in vectorizer.get_feature_names_out()),
        )
        self._model = _URLModel(vectorizer, scaler, classifier, features)
        return self

    def predict(self, url: str, *, top_k: int | None = 12) -> ModelPrediction:
        """Score one validated URL. Set top_k=None to include every nonzero term."""
        model = self._require_model()
        indicators = tuple(extract_structural_features(url).values())
        structure = csr_matrix(model.scaler.transform([indicators]))
        ngrams = model.vectorizer.transform([_lexical_url(url)])
        row = hstack((structure, ngrams), format="csr")
        return explain_prediction(
            model.classifier, row, model.features, top_k, raw_structural_values=indicators
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        self._require_model()
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Load a locally generated joblib artifact of trusted origin."""
        detector = load_artifact(path, cls)
        if detector._model is None:
            raise ModelArtifactError("The URL detector artifact has no fitted model.")
        return detector

    def _require_model(self) -> _URLModel:
        if self._model is None:
            raise RuntimeError("Fit or load a URL detector before using it.")
        return self._model
