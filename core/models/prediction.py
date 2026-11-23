"""Inference results and signed attributions shared by both detectors."""

from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["structure", "character_ngram", "token"]


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One signed term in the model's decision log odds."""

    feature: str = field(repr=False)
    value: float
    coefficient: float
    contribution: float
    kind: FeatureKind
    raw_value: float | None = None


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Positive-class probability with a signed feature attribution."""

    probability: float
    log_odds: float
    intercept: float
    contributions: tuple[FeatureContribution, ...]
    unshown_contribution: float = 0.0
