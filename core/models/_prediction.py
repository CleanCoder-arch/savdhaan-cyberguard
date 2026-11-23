"""Shared inputs, explanations, and trusted local artifact I/O for the detectors."""

import math
import os
import warnings
import zlib
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral
from pathlib import Path
from pickle import UnpicklingError
from tempfile import NamedTemporaryFile
from typing import TypeVar

import joblib
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression

from core.models.prediction import FeatureContribution, FeatureKind, ModelPrediction

_ARTIFACT_VERSION = 1


class ModelArtifactError(ValueError):
    """A local detector artifact is malformed or incompatible."""


def training_data(
    inputs: Iterable[str], labels: Iterable[int], validate: Callable[[str], str]
) -> tuple[list[str], list[int]]:
    if isinstance(inputs, (str, bytes)):
        raise ValueError("Provide an iterable of training records.")
    records = [validate(item) for item in inputs]
    targets = list(labels)
    if len(records) != len(targets):
        raise ValueError("Provide one label for each training record.")
    if any(
        not isinstance(label, Integral) or isinstance(label, bool) or label not in (0, 1)
        for label in targets
    ):
        raise ValueError("Labels must be integers, with 0 for benign and 1 for scam.")
    if set(targets) != {0, 1}:
        raise ValueError("Training data must contain both benign and scam records.")
    return records, targets


def explain_prediction(
    classifier: LogisticRegression,
    row: csr_matrix,
    features: Sequence[tuple[FeatureKind, str]],
    top_k: int | None,
    *,
    raw_structural_values: Sequence[float] = (),
) -> ModelPrediction:
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer or None.")

    weights = classifier.coef_[0]
    contributions = []
    for index, value in zip(row.indices, row.data, strict=True):
        coefficient = weights[index]
        contribution = float(value * coefficient)
        if contribution == 0:
            continue
        kind, feature = features[index]
        contributions.append(
            FeatureContribution(
                feature=feature,
                value=float(value),
                coefficient=float(coefficient),
                contribution=contribution,
                kind=kind,
                raw_value=raw_structural_values[index] if kind == "structure" else None,
            )
        )
    contributions.sort(key=lambda feature: (-abs(feature.contribution), feature.kind, feature.feature))
    selected = tuple(contributions[:top_k])
    return ModelPrediction(
        probability=float(classifier.predict_proba(row)[0, 1]),
        log_odds=float(classifier.decision_function(row)[0]),
        intercept=float(classifier.intercept_[0]),
        contributions=selected,
        unshown_contribution=math.fsum(
            feature.contribution for feature in contributions[len(selected):]
        ),
    )


def save_artifact(model: object, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    with NamedTemporaryFile(dir=destination.parent, suffix=".joblib.tmp", delete=False) as staging:
        temporary_path = Path(staging.name)
    try:
        joblib.dump(
            {"artifact_version": _ARTIFACT_VERSION, "sklearn_version": sklearn_version, "detector": model},
            temporary_path,
            compress=3,
        )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


_Model = TypeVar("_Model")


def load_artifact(path: str | os.PathLike[str], expected: type[_Model]) -> _Model:
    """Load a trusted artifact created by this installation."""
    with Path(path).open("rb") as source:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                artifact = joblib.load(source)
        except InconsistentVersionWarning:
            raise ModelArtifactError("Rebuild the artifact with this scikit-learn version.") from None
        except (
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            TypeError,
            UnpicklingError,
            ValueError,
            zlib.error,
        ):
            raise ModelArtifactError("The detector artifact could not be read.") from None
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != _ARTIFACT_VERSION:
        raise ModelArtifactError("The detector artifact format is incompatible.")
    if artifact.get("sklearn_version") != sklearn_version:
        raise ModelArtifactError("Rebuild the artifact with this scikit-learn version.")
    model = artifact.get("detector")
    if not isinstance(model, expected):
        raise ModelArtifactError("The artifact is for a different detector.")
    return model
