"""Fit on training groups, then freeze the operating threshold on validation groups."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, Self

from core.models.prediction import ModelPrediction
from core.models.sms_detector import SMSDetector
from core.models.url_detector import URLDetector

from .split_builder import ModelKind, Partition

THRESHOLD_POLICY = "validation_f1_then_lower_fpr_then_higher_threshold"


class Detector(Protocol):
    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> Self: ...

    def predict(self, text: str, *, top_k: int | None = 0) -> ModelPrediction: ...


@dataclass(frozen=True, slots=True)
class Confusion:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0


def _probability(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A prediction or threshold must be a probability between zero and one.")
    return float(value)


def confusion_at(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Confusion:
    threshold = _probability(threshold)
    true_positive = false_positive = true_negative = false_negative = 0
    for score, label in zip(scores, labels, strict=True):
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Benchmark labels must be integers of 0 or 1.")
        if _probability(score) >= threshold:
            true_positive += label
            false_positive += 1 - label
        else:
            false_negative += label
            true_negative += 1 - label
    return Confusion(true_positive, false_positive, true_negative, false_negative)


def select_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Maximize validation F1, then prefer a lower FPR and a higher threshold."""
    if any(type(label) is not int for label in labels) or set(labels) != {0, 1}:
        raise ValueError("Threshold selection needs both validation classes.")
    ranked = sorted(
        ((_probability(score), label) for score, label in zip(scores, labels, strict=True)), reverse=True
    )
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positive = false_positive = 0
    best = (-1.0, 0.0, 1.0)
    next_index = 0
    while next_index < len(ranked):
        threshold = ranked[next_index][0]
        while next_index < len(ranked) and ranked[next_index][0] == threshold:
            label = ranked[next_index][1]
            true_positive += label
            false_positive += 1 - label
            next_index += 1
        confusion = Confusion(true_positive, false_positive, negatives - false_positive, positives - true_positive)
        best = max(best, (confusion.f1, -confusion.false_positive_rate, threshold))
    return best[2]


def partition_fingerprint(partition: Partition) -> str:
    """Hash a partition as a whole, without exposing per-record fingerprints."""
    rows = sorted(
        (example.text, example.label, group)
        for example, group in zip(partition.examples, partition.groups, strict=True)
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _lock_fingerprint(kind: ModelKind, threshold: float, train_sha256: str, validation_sha256: str) -> str:
    record = (kind, threshold, train_sha256, validation_sha256, THRESHOLD_POLICY)
    return hashlib.sha256(json.dumps(record, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LockedModel:
    kind: ModelKind
    detector: Detector = field(repr=False)
    threshold: float
    train_sha256: str
    validation_sha256: str
    source_groups: frozenset[str] = field(repr=False)
    lock_sha256: str

    def verify_for(self, test: Partition) -> None:
        expected = _lock_fingerprint(self.kind, self.threshold, self.train_sha256, self.validation_sha256)
        if self.lock_sha256 != expected:
            raise ValueError("The detector threshold lock does not match its training and validation provenance.")
        if not self.source_groups.isdisjoint(test.groups):
            raise ValueError("Held-out groups overlap training or validation groups.")


def fit_and_lock(kind: ModelKind, train: Partition, validation: Partition) -> LockedModel:
    if not set(train.groups).isdisjoint(validation.groups):
        raise ValueError("Training and validation groups overlap.")
    if {row.label for row in train.examples} != {0, 1} or {row.label for row in validation.examples} != {0, 1}:
        raise ValueError("Both training and validation partitions need both classes.")
    if kind == "url":
        detector = URLDetector()
    elif kind == "sms":
        detector = SMSDetector()
    else:
        raise ValueError("Select a URL or SMS detector.")
    detector.fit([row.text for row in train.examples], [row.label for row in train.examples])
    validation_scores = [detector.predict(row.text, top_k=0).probability for row in validation.examples]
    threshold = select_threshold(validation_scores, [row.label for row in validation.examples])
    train_sha256 = partition_fingerprint(train)
    validation_sha256 = partition_fingerprint(validation)
    return LockedModel(
        kind,
        detector,
        threshold,
        train_sha256,
        validation_sha256,
        frozenset((*train.groups, *validation.groups)),
        _lock_fingerprint(kind, threshold, train_sha256, validation_sha256),
    )
