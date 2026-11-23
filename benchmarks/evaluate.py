"""Evaluate the frozen detectors against held-out groups and emit metadata-only results."""

import csv
import hashlib
import json
import math
import platform
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile
from typing import Literal

import sklearn
import tldextract

from .split_builder import GroupedSplit, LabeledExample, ModelKind, Partition, SPLIT_FRACTIONS, build_split
from .train_and_lock import THRESHOLD_POLICY, LockedModel, confusion_at, fit_and_lock, partition_fingerprint

DatasetKind = Literal["provided", "synthetic_demo"]


def read_csv(path: Path) -> tuple[LabeledExample, ...]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if sorted(reader.fieldnames or []) != ["label", "text"]:
            raise ValueError("Benchmark CSV needs exactly the text and label headers.")
        examples = []
        for row in reader:
            label = row.get("label")
            text = row.get("text")
            if None in row or label is None or label.strip() not in {"0", "1"} or text is None:
                raise ValueError(f"Invalid benchmark CSV record at line {reader.line_num}.")
            examples.append(LabeledExample(text, int(label)))
        return tuple(examples)


def _partition_summary(partition: Partition) -> dict:
    positives = sum(example.label for example in partition.examples)
    return {
        "samples": len(partition.examples),
        "groups": len(set(partition.groups)),
        "positives": positives,
        "negatives": len(partition.examples) - positives,
        "sha256": partition_fingerprint(partition),
    }


def evaluate_locked(locked: LockedModel, splits: GroupedSplit) -> dict:
    """Use one locked operating point to score the untouched test partition."""
    if (
        partition_fingerprint(splits.train) != locked.train_sha256
        or partition_fingerprint(splits.validation) != locked.validation_sha256
    ):
        raise ValueError("The split does not match the locked training and validation inputs.")
    locked.verify_for(splits.test)
    labels = [example.label for example in splits.test.examples]
    if set(labels) != {0, 1}:
        raise ValueError("The held-out partition needs both classes.")
    scores = []
    latencies = []
    for example in splits.test.examples:
        start = time.perf_counter_ns()
        scores.append(locked.detector.predict(example.text, top_k=0).probability)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    confusion = confusion_at(scores, labels, locked.threshold)
    partitions = {
        name: _partition_summary(getattr(splits, name))
        for name in SPLIT_FRACTIONS
    }
    joined = Partition(
        (*splits.train.examples, *splits.validation.examples, *splits.test.examples),
        (*splits.train.groups, *splits.validation.groups, *splits.test.groups),
    )
    return {
        "threshold": locked.threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "lock_sha256": locked.lock_sha256,
        "dataset_sha256": partition_fingerprint(joined),
        "split_sha256": hashlib.sha256(
            "".join(summary["sha256"] for summary in partitions.values()).encode("ascii")
        ).hexdigest(),
        "splits": partitions,
        "metrics": {
            "recall": confusion.recall,
            "precision": confusion.precision,
            "false_positive_rate": confusion.false_positive_rate,
            "f1": confusion.f1,
            **asdict(confusion),
        },
        "latency_ms": {
            "mean": mean(latencies),
            "p50": median(latencies),
            "p95": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        },
    }


def run_benchmark(
    urls: Iterable[LabeledExample],
    messages: Iterable[LabeledExample],
    *,
    seed: int = 42,
    dataset_kind: DatasetKind = "provided",
) -> dict:
    if dataset_kind not in ("provided", "synthetic_demo"):
        raise ValueError("Unknown benchmark dataset kind.")
    datasets: dict[ModelKind, Iterable[LabeledExample]] = {"url": urls, "sms": messages}
    splits = {kind: build_split(kind, rows, seed=seed) for kind, rows in datasets.items()}
    locks = {kind: fit_and_lock(kind, split.train, split.validation) for kind, split in splits.items()}
    return {
        "schema_version": 1,
        "protocol": "grouped_holdout_v1",
        "dataset_kind": dataset_kind,
        "seed": seed,
        "split_fractions": SPLIT_FRACTIONS.copy(),
        "latency_scope": "one local detector inference per held-out example, without network or storage",
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "tldextract": tldextract.__version__,
        },
        "models": {kind: evaluate_locked(locks[kind], split) for kind, split in splits.items()},
    }


def write_results(output: Path, result: dict) -> None:
    """Replace the artifact after both held-out evaluations have finished."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output.parent, prefix=".benchmark-", suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.close()
            temporary.replace(output)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)
