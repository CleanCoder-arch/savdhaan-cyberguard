"""Offline SMS classification with word-level TF-IDF and token attribution."""

import os
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby
from typing import Self

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from core.guidance.redactor import redact_text
from core.models._prediction import (
    ModelArtifactError,
    explain_prediction,
    load_artifact,
    save_artifact,
    training_data,
)
from core.models.prediction import FeatureKind, ModelPrediction
from core.validators import validate_text


def _is_word_character(character: str) -> bool:
    return character == "_" or unicodedata.category(character)[0] in {"L", "M", "N"}


def tokenize_sms(text: str) -> list[str]:
    """Segment normalized words, keeping combining marks attached to Indic text."""
    normalized = redact_text(text).casefold()
    tokens = []
    for is_word, characters in groupby(normalized, key=_is_word_character):
        if not is_word:
            continue
        token = "".join(characters)
        if any(character.isnumeric() for character in token):
            tokens.append("<number>")
        elif any(character.isalpha() for character in token):
            tokens.append(token)
    return tokens


@dataclass(frozen=True, slots=True)
class _SMSModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    features: tuple[tuple[FeatureKind, str], ...]


class SMSDetector:
    """Word-level logistic regression for positive class 1."""

    def __init__(self) -> None:
        self._model: _SMSModel | None = None

    def fit(self, messages: Iterable[str], labels: Iterable[int]) -> Self:
        records, targets = training_data(messages, labels, validate_text)
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_sms, token_pattern=None, lowercase=False, sublinear_tf=True
        )
        try:
            matrix = vectorizer.fit_transform(records)
        except ValueError:
            raise ValueError("Training messages must contain usable words or numbers.") from None
        classifier = LogisticRegression(max_iter=1000, random_state=0).fit(matrix, targets)
        features = tuple(("token", token) for token in vectorizer.get_feature_names_out())
        self._model = _SMSModel(vectorizer, classifier, features)
        return self

    def predict(self, message: str, *, top_k: int | None = 10) -> ModelPrediction:
        """Score one message and return its strongest contributing words."""
        model = self._require_model()
        row = model.vectorizer.transform([validate_text(message)])
        return explain_prediction(model.classifier, row, model.features, top_k)

    def save(self, path: str | os.PathLike[str]) -> None:
        self._require_model()
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Load a locally generated joblib artifact of trusted origin."""
        detector = load_artifact(path, cls)
        if detector._model is None:
            raise ModelArtifactError("The SMS detector artifact has no fitted model.")
        return detector

    def _require_model(self) -> _SMSModel:
        if self._model is None:
            raise RuntimeError("Fit or load an SMS detector before using it.")
        return self._model
