import csv
import json
import sys
from dataclasses import replace
from fractions import Fraction
from random import Random
from types import SimpleNamespace

import pytest

from benchmarks import __main__, train_and_lock
from benchmarks.evaluate import evaluate_locked, read_csv, run_benchmark, write_results
from benchmarks.split_builder import (
    GroupedSplit,
    LabeledExample,
    Partition,
    _similar_pairs,
    build_split,
    registrable_domain,
    sms_template_groups,
)
from benchmarks.train_and_lock import fit_and_lock, select_threshold
from scripts.bootstrap_models import _SMS_EXAMPLES, _URL_EXAMPLES


def _examples(rows):
    return [LabeledExample(text, label) for label, text in rows]


def _partition(examples, groups):
    return Partition(tuple(_examples(examples)), tuple(groups))


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.current.BANK.CO.IN.:8443/private?token=secret#fragment", "bank.co.in"),
        ("https://s.biücher.de/customer-help", "xn--bicher-4ya.de"),
        ("https://app.tenant.github.io/home", "tenant.github.io"),
        ("http://portal.testbank.test/a", "testbank.test"),
        ("http://localhost/", "localhost"),
        ("http://192.0.2.17:8080/a", "192.0.2.17"),
        ("https://[::ffff:192.0.2.17]/", "192.0.2.17"),
        ("https://[2001:0db8:0::1234]/", "2001:db8::1234"),
    ],
)
def test_url_registration_boundaries(url, domain):
    assert registrable_domain(url) == domain


def test_url_split_never_separates_subdomains_of_a_registration():
    examples = [
        LabeledExample(f"https://{subdomain}.tenant{index}.co.in/{path}", label)
        for index in range(5)
        for subdomain, path, label in (("services", "order", 0), ("login", "support", 1))
    ]
    splits = build_split("url", examples)
    partitions = (splits.train, splits.validation, splits.test)
    assert sum(len(partition.examples) for partition in partitions) == 10
    groups = [set(partition.groups) for partition in partitions]
    assert groups[0].isdisjoint(groups[1] | groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert all({example.label for example in partition.examples} == {0, 1} for partition in partitions)
    assert build_split("url", reversed(examples)) == splits


def test_one_pure_group_per_class_per_partition():
    rows = [
        LabeledExample(f"https://class{label}-group{group}.test/only", label)
        for label in (0, 1)
        for group in range(3)
    ]
    splits = build_split("url", rows)
    partitions = (splits.train, splits.validation, splits.test)
    assert all({row.label for row in partition.examples} == {0, 1} for partition in partitions)
    assert len({group for part in partitions for group in part.groups}) == 6


def test_sms_groups_mask_tokens_and_join_near_duplicate_templates():
    first = sms_template_groups([
        "Verify your KYC immediately using https://private.test/first?secret=confidential",
        "Verify your KYC urgently using https://private.test/second",
        "VERIFY your kyc immediately using https://other.test/A.",
        "The blue delivery van is waiting at the warehouse.",
    ])
    assert len(set(first[:3])) == 1
    assert first[0] != first[3]


def test_sms_near_matches_are_clustered_transitively():
    a = "payment balance freeze notice verify account now"
    b = "payment balance freeze notice verify account today"
    c = "payment balance review notice verify account today"
    assert len(set(sms_template_groups([a, c]))) == 2
    assert len(set(sms_template_groups([a, b, c]))) == 1


def test_sms_near_duplicates_cannot_cross_partitions():
    examples = _examples(_SMS_EXAMPLES)
    variants = (
        LabeledExample("Verify your mobile security to recover account now via https://different.test/login", 1),
        LabeledExample("Verify your mobile safety to recover account now via https://another.test/confirm", 1),
    )
    splits = build_split("sms", (*examples, *variants))
    partitions = (splits.train, splits.validation, splits.test)
    locations = [
        slot for variant in variants
        for slot, partition in enumerate(partitions)
        if variant in partition.examples
    ]
    assert len(locations) == 2
    assert len(set(locations)) == 1
    groups = [set(partition.groups) for partition in partitions]
    assert sum(map(len, groups)) == len(set.union(*groups))


@pytest.mark.parametrize("threshold", [Fraction(2, 3), Fraction(4, 5)])
def test_prefix_index_finds_all_pairs_above_similarity_cutoff(threshold):
    random = Random(43)
    vocabulary = list("abcdefghijkl")
    signatures = [
        frozenset(random.sample(vocabulary, random.randrange(1, len(vocabulary))))
        for _ in range(150)
    ]
    brute = {
        (first, second)
        for first, signature in enumerate(signatures)
        for second, other in enumerate(signatures[:first])
        if Fraction(len(signature & other), len(signature | other)) >= threshold
    }
    assert set(_similar_pairs(signatures, threshold)) == brute


@pytest.mark.parametrize(
    ("kind", "examples", "message"),
    [
        ("url", [(0, "https://ab.co.in/a")], "separate groups"),
        ("sms", [(1, "The parcel has arrived.")], "separate groups"),
        ("sms", [(1, "🫢")], "usable word"),
        ("sms", [(1, "Urgent payment 500 due"), (0, "URGENT PAYMENT 900 DUE!")], "contradictory"),
        ("url", [(0, "https://foo.test/"), (1, "https://FOO.test:443#frag")], "contradictory"),
        ("url", [(True, "https://foo.test/")], "integer labels"),
        ("url", [(1, "http://@foo.test/path?token=private")], "credentials"),
    ],
)
def test_invalid_or_contradictory_benchmark_data_is_rejected(kind, examples, message):
    with pytest.raises(ValueError, match=message) as raised:
        build_split(kind, _examples(examples))
    assert "token=private" not in str(raised.value)


def test_threshold_selection_prefers_lower_false_positive_rate_on_f1_tie():
    assert select_threshold([0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1]) == 0.9
    assert select_threshold([0.42, 0.42], [0, 1]) == 0.42
    with pytest.raises(ValueError, match="probability"):
        select_threshold([0.3, float("nan")], [0, 1])
    with pytest.raises(ValueError, match="both"):
        select_threshold([0.2, 0.8], [1, 1])


def test_training_and_validation_lock_before_heldout_scoring(monkeypatch):
    calls = []

    class Detector:
        def fit(self, texts, labels):
            calls.append(("fit", tuple(texts), tuple(labels)))
            return self

        def predict(self, text, *, top_k):
            assert top_k == 0
            calls.append(("predict", text))
            return SimpleNamespace(probability={
                "validation scam": 0.7,
                "validation benign": 0.4,
                "test scam": 0.65,
                "test benign": 0.8,
            }[text])

    monkeypatch.setattr(train_and_lock, "SMSDetector", Detector)
    train = _partition([(1, "train scam"), (0, "train benign")], ["s1", "b1"])
    validation = _partition([(1, "validation scam"), (0, "validation benign")], ["s2", "b2"])
    test = _partition([(1, "test scam"), (0, "test benign")], ["s3", "b3"])
    locked = fit_and_lock("sms", train, validation)
    assert locked.threshold == 0.7
    pre_test_calls = [
        ("fit", ("train scam", "train benign"), (1, 0)),
        ("predict", "validation scam"),
        ("predict", "validation benign"),
    ]
    assert calls == pre_test_calls
    result = evaluate_locked(locked, GroupedSplit(train, validation, test))
    assert calls == [*pre_test_calls, ("predict", "test scam"), ("predict", "test benign")]
    assert result["threshold"] == 0.7
    assert result["metrics"]["recall"] == 0
    assert result["metrics"]["precision"] == 0
    assert result["metrics"]["false_positive_rate"] == 1
    assert result["metrics"]["false_positive"] == 1
    assert result["metrics"]["false_negative"] == 1


def test_evaluation_rejects_overlap_and_changed_lock():
    splits = build_split("url", _examples(_URL_EXAMPLES))
    locked = fit_and_lock("url", splits.train, splits.validation)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_locked(locked, replace(splits, test=splits.validation))
    with pytest.raises(ValueError, match="threshold lock"):
        evaluate_locked(replace(locked, threshold=1 - locked.threshold), splits)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_locked(locked, replace(splits, train=splits.test))


def test_benchmark_results_reproduce_and_omit_raw_inputs(tmp_path):
    urls = _examples(_URL_EXAMPLES) + [LabeledExample("https://extra.test/SecretPath?token=RawSecretValue", 0)]
    messages = _examples(_SMS_EXAMPLES) + [LabeledExample("Open the private outlet named UnlistedSender.", 0)]
    first = run_benchmark(urls, messages, seed=71)
    second = run_benchmark(reversed(urls), reversed(messages), seed=71)
    output = tmp_path / "reports" / "benchmark_results.json"
    write_results(output, first)
    raw = output.read_text()
    assert all(secret not in raw for secret in ("RawSecretValue", "SecretPath", "UnlistedSender"))
    loaded = json.loads(raw)
    assert loaded["protocol"] == "grouped_holdout_v1"
    assert loaded["dataset_kind"] == "provided"
    for kind in ("url", "sms"):
        measured = first["models"][kind].pop("latency_ms")
        second["models"][kind].pop("latency_ms")
        assert set(measured) == {"mean", "p50", "p95"}
        assert all(value > 0 for value in measured.values())
        assert len(loaded["models"][kind]["lock_sha256"]) == 64
        assert all(partition["groups"] > 0 for partition in loaded["models"][kind]["splits"].values())
    assert first == second


def test_results_write_is_atomic_on_serialization_failure(tmp_path):
    output = tmp_path / "benchmark_results.json"
    write_results(output, {"prior": "report"})
    with pytest.raises(ValueError):
        write_results(output, {"invalid": float("nan")})
    assert json.loads(output.read_text()) == {"prior": "report"}
    assert list(tmp_path.iterdir()) == [output]


def test_csv_input_and_cli_output(tmp_path, monkeypatch):
    url_csv = tmp_path / "urls.csv"
    sms_csv = tmp_path / "messages.csv"
    for path, rows in ((url_csv, _URL_EXAMPLES), (sms_csv, _SMS_EXAMPLES)):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("text", "label"))
            writer.writerows((text, label) for label, text in rows)
    output = tmp_path / "benchmark_results.json"
    monkeypatch.setattr(
        sys, "argv",
        ["benchmarks", "--url-data", str(url_csv), "--sms-data", str(sms_csv), "--output", str(output)],
    )
    __main__.main()
    assert json.loads(output.read_text())["dataset_kind"] == "provided"
    assert read_csv(url_csv) == tuple(_examples(_URL_EXAMPLES))


def test_csv_rejects_invalid_headers_and_labels_without_echoing_values(tmp_path):
    path = tmp_path / "messages.csv"
    path.write_text("message,label\nsecret,1\n")
    with pytest.raises(ValueError, match="headers"):
        read_csv(path)
    path.write_text("text,label\nsecret,other_secret\n")
    with pytest.raises(ValueError, match="line 2") as raised:
        read_csv(path)
    assert "secret" not in str(raised.value)
