"""Keep dependent URL hosts and SMS templates in the same partition."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from ipaddress import IPv4Address, ip_address
from random import Random
from typing import Literal
from urllib.parse import urlsplit

import httpx
from tldextract import TLDExtract

from core.models.sms_detector import tokenize_sms
from core.validators import validate_text, validate_url

ModelKind = Literal["url", "sms"]
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}
_PSL = TLDExtract(
    cache_dir=None,
    suffix_list_urls=(),
    include_psl_private_domains=True,
)
_WORD_SIMILARITY = Fraction(2, 3)
_CHARACTER_SIMILARITY = Fraction(4, 5)


@dataclass(frozen=True, slots=True)
class LabeledExample:
    text: str = field(repr=False)
    label: int


@dataclass(frozen=True, slots=True)
class Partition:
    examples: tuple[LabeledExample, ...] = field(repr=False)
    groups: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.examples) != len(self.groups):
            raise ValueError("Each example needs one group.")


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    train: Partition
    validation: Partition
    test: Partition


def _hostname(url: str) -> str:
    try:
        return httpx.URL(url).raw_host.decode("ascii").rstrip(".")
    except (httpx.InvalidURL, UnicodeError):
        raise ValueError("The URL hostname could not be normalized.") from None


def _domain_group(host: str) -> str:
    try:
        address = ip_address(host)
    except ValueError:
        domain = _PSL(host).top_domain_under_public_suffix
        # Unknown suffixes, including .test, use the last two labels.
        return domain or ".".join(host.split(".")[-2:])
    if not isinstance(address, IPv4Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(address)


def registrable_domain(url: str) -> str:
    """Return a DNS registration boundary or a normalized IP address."""
    return _domain_group(_hostname(validate_url(url)))


def _check_label(labels: dict[object, int], key: object, label: int) -> None:
    if labels.setdefault(key, label) != label:
        raise ValueError("Equivalent examples have contradictory labels.")


def _url_groups(examples: Sequence[LabeledExample]) -> list[str]:
    labels: dict[object, int] = {}
    groups = []
    for example in examples:
        value = validate_url(example.text)
        parts = urlsplit(value)
        host = _hostname(value)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        _check_label(labels, (parts.scheme, host, port, parts.path or "/", parts.query), example.label)
        groups.append(_domain_group(host))
    return groups


def _similar_pairs(signatures: Sequence[frozenset[str]], threshold: Fraction) -> Iterable[tuple[int, int]]:
    """Find Jaccard matches via frequency-ordered prefix indexing."""
    frequency = Counter(token for signature in signatures for token in signature)
    previous: dict[str, list[int]] = defaultdict(list)
    numerator, denominator = threshold.numerator, threshold.denominator
    for index, signature in enumerate(signatures):
        prefix_length = len(signature) - (len(signature) * numerator + denominator - 1) // denominator + 1
        prefix = sorted(signature, key=lambda token: (frequency[token], token))[:prefix_length]
        candidates = {candidate for token in prefix for candidate in previous[token]}
        for candidate in sorted(candidates):
            other = signatures[candidate]
            if min(len(signature), len(other)) * denominator < max(len(signature), len(other)) * numerator:
                continue
            intersection = len(signature & other)
            union = len(signature) + len(other) - intersection
            if intersection * denominator >= union * numerator:
                yield index, candidate
        for token in prefix:
            previous[token].append(index)


def _character_shingles(template: tuple[str, ...]) -> frozenset[str]:
    value = " " + " ".join(template) + " "
    return frozenset(value[index : index + 3] for index in range(len(value) - 2))


def _sms_template(message: str) -> tuple[str, ...]:
    template = tuple(tokenize_sms(validate_text(message)))
    if not template:
        raise ValueError("Benchmark messages need at least one usable word or number.")
    return template


def _template_components(templates: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    unique = sorted(set(templates))
    indices = {template: index for index, template in enumerate(unique)}
    parents = list(range(len(unique)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for signatures, threshold in (
        ([frozenset(template) for template in unique], _WORD_SIMILARITY),
        ([_character_shingles(template) for template in unique], _CHARACTER_SIMILARITY),
    ):
        for first, second in _similar_pairs(signatures, threshold):
            first, second = find(first), find(second)
            parents[max(first, second)] = min(first, second)

    return tuple(f"sms:{find(indices[template])}" for template in templates)


def sms_template_groups(messages: Iterable[str]) -> tuple[str, ...]:
    """Identify connected components of exact or near-duplicate masked templates."""
    return _template_components([_sms_template(message) for message in messages])


def _sms_groups(examples: Sequence[LabeledExample]) -> tuple[str, ...]:
    labels: dict[object, int] = {}
    templates = [_sms_template(example.text) for example in examples]
    for example, template in zip(examples, templates, strict=True):
        _check_label(labels, template, example.label)
    return _template_components(templates)


def _assign_groups(
    group_labels: dict[str, Counter[int]], seed: int,
) -> tuple[list[str], list[str], list[str]]:
    names = sorted(group_labels)
    Random(seed).shuffle(names)
    names.sort(key=lambda name: group_labels[name].total(), reverse=True)
    if any(sum(label in counts for counts in group_labels.values()) < 3 for label in (0, 1)):
        raise ValueError("Each class needs at least three separate groups for train, validation, and test.")
    mixed = iter(name for name in names if len(group_labels[name]) == 2)
    pure = tuple(iter([name for name in names if set(group_labels[name]) == {label}]) for label in (0, 1))
    assignments: tuple[list[str], list[str], list[str]] = ([], [], [])
    running: tuple[Counter[int], ...] = (Counter(), Counter(), Counter())
    assigned: set[str] = set()

    def assign(slot: int, name: str) -> None:
        assignments[slot].append(name)
        running[slot].update(group_labels[name])
        assigned.add(name)

    for slot in range(3):
        name = next(mixed, None)
        if name is not None:
            assign(slot, name)
        else:
            for candidates in pure:
                assign(slot, next(candidates))

    total = sum(group_labels.values(), Counter())
    targets = [{label: fraction * total[label] for label in (0, 1)} for fraction in SPLIT_FRACTIONS.values()]
    for name in names:
        if name in assigned:
            continue
        counts = group_labels[name]

        def imbalance_increase(slot: int) -> float:
            return sum(
                counts[label] * (2 * (running[slot][label] - target) + counts[label]) / target
                for label, target in targets[slot].items()
            )

        assign(min(range(3), key=imbalance_increase), name)
    return assignments


def build_split(kind: ModelKind, examples: Iterable[LabeledExample], *, seed: int = 42) -> GroupedSplit:
    """Create deterministic class-covered 60/20/20 splits at the group level."""
    if kind not in ("url", "sms") or type(seed) is not int:
        raise ValueError("Select URL or SMS data and an integer seed.")
    examples = tuple(examples)
    if not examples or any(
        not isinstance(example, LabeledExample)
        or not isinstance(example.text, str)
        or type(example.label) is not int
        or example.label not in (0, 1)
        for example in examples
    ):
        raise ValueError("Benchmark data needs text and integer labels of 0 or 1.")
    ordered = sorted(examples, key=lambda example: (example.text, example.label))
    groups = _url_groups(ordered) if kind == "url" else _sms_groups(ordered)
    group_labels: dict[str, Counter[int]] = defaultdict(Counter)
    for example, group in zip(ordered, groups, strict=True):
        group_labels[group][example.label] += 1
    assignments = _assign_groups(group_labels, seed)
    split_index = {group: slot for slot, subset in enumerate(assignments) for group in subset}
    rows: list[list[LabeledExample]] = [[], [], []]
    row_groups: list[list[str]] = [[], [], []]
    for example, group in zip(ordered, groups, strict=True):
        slot = split_index[group]
        rows[slot].append(example)
        row_groups[slot].append(group)
    return GroupedSplit(*(Partition(tuple(part), tuple(keys)) for part, keys in zip(rows, row_groups, strict=True)))
