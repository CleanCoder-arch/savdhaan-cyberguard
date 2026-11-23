from math import fsum

import pytest

from core.schemas import EvidenceSource, RiskLevel, ScanIssue, ScanStatus
from core.scoring.aggregator import ML_MAX_POINTS, URLHAUS_LISTED_POINTS, aggregate_risk
from core.scoring.rules_engine import RuleHit, evaluate_rules


def test_combines_models_rules_and_available_intel_with_signed_attribution() -> None:
    rules = evaluate_rules(text="Final warning. Update your details.")
    result = aggregate_risk(
        url_probability=0.49,
        sms_probability=0.2,
        rule_hits=rules,
        urlhaus_listed=True,
    )
    assert result.status == ScanStatus.COMPLETE
    assert result.risk_score == 97
    assert result.risk_level == RiskLevel.CRITICAL
    assert fsum(item.points for item in result.contributions) == pytest.approx(result.risk_score)
    assert [item.source for item in result.contributions] == [
        EvidenceSource.ML,
        EvidenceSource.ML,
        EvidenceSource.RULES,
        EvidenceSource.THREAT_INTEL,
        EvidenceSource.RULES,
    ]
    assert [item.code for item in result.evidence] == [
        "model.url",
        "model.sms",
        "message.urgency",
        "urlhaus.active_host_listing",
        "scoring.rounding",
    ]
    assert [item.points for item in result.contributions] == pytest.approx([34.3, 0, 8, 55, -0.3])


def test_larger_sms_model_contributes_when_both_are_available() -> None:
    result = aggregate_risk(url_probability=0.1, sms_probability=0.6)
    assert result.risk_score == 42
    assert [(item.code, item.points) for item in result.contributions] == [
        ("model.url", 0),
        ("model.sms", 42),
    ]


def test_equal_model_estimates_only_contribute_once_in_stable_order() -> None:
    result = aggregate_risk(url_probability=1, sms_probability=1)
    assert result.risk_score == ML_MAX_POINTS
    assert [item.points for item in result.contributions] == [ML_MAX_POINTS, 0]


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (0, RiskLevel.LOW),
        (24, RiskLevel.LOW),
        (25, RiskLevel.MEDIUM),
        (54, RiskLevel.MEDIUM),
        (55, RiskLevel.HIGH),
        (79, RiskLevel.HIGH),
        (80, RiskLevel.CRITICAL),
        (100, RiskLevel.CRITICAL),
    ],
)
def test_risk_thresholds_are_inclusive_at_each_boundary(score: int, level: RiskLevel) -> None:
    result = aggregate_risk(rule_hits=(RuleHit("example.points", score, "Synthetic score."),))
    assert result.risk_score == score
    assert result.risk_level == level


def test_rounds_half_up_and_explains_positive_adjustment() -> None:
    result = aggregate_risk(url_probability=0.25)
    assert result.risk_score == 18
    assert result.contributions[-1].code == "scoring.rounding"
    assert result.contributions[-1].points == 0.5


def test_clamps_above_100_and_explains_reduction() -> None:
    result = aggregate_risk(url_probability=1, urlhaus_listed=True)
    assert result.risk_score == 100
    assert result.contributions[-1].code == "scoring.clamp"
    assert result.contributions[-1].points == -25


def test_negative_rule_contributions_and_lower_clamp_are_explained() -> None:
    result = aggregate_risk(
        sms_probability=0.1, rule_hits=(RuleHit("example.safe_context", -20, "Evidence lowered risk."),)
    )
    assert result.risk_score == 0
    assert [item.points for item in result.contributions] == [7, -20, 13]
    assert result.contributions[-1].code == "scoring.clamp"


def test_active_urlhaus_listing_alone_is_host_level_evidence() -> None:
    result = aggregate_risk(urlhaus_listed=True)
    assert result.risk_score == URLHAUS_LISTED_POINTS
    assert result.risk_level == RiskLevel.HIGH
    assert "different URL on that host" in result.contributions[0].summary


def test_confirmed_no_listing_is_distinct_from_unavailable() -> None:
    no_listing = aggregate_risk(url_probability=0.5, urlhaus_listed=False)
    offline = aggregate_risk(
        url_probability=0.5,
        issues=(ScanIssue(component="urlhaus", code="timeout"),),
    )
    assert no_listing.risk_score == offline.risk_score == 35
    assert no_listing.status == ScanStatus.COMPLETE
    assert offline.status == ScanStatus.PARTIAL
    assert no_listing.contributions[-1].source == EvidenceSource.THREAT_INTEL
    assert no_listing.contributions[-1].points == 0
    assert all(item.source != EvidenceSource.THREAT_INTEL for item in offline.contributions)


def test_partial_result_retains_usable_rules_and_deduplicates_issues() -> None:
    timed_out = ScanIssue(component="whois", code="timeout")
    rate_limited = ScanIssue(component="urlhaus", code="rate_limited")
    hits = evaluate_rules(text="Final warning.")
    result = aggregate_risk(
        sms_probability=0.3,
        rule_hits=hits + hits,
        issues=(timed_out, rate_limited, timed_out),
    )
    assert result.risk_score == 29
    assert result.risk_level == RiskLevel.MEDIUM
    assert result.status == ScanStatus.PARTIAL
    assert result.issues == (timed_out, rate_limited)
    assert [item.code for item in result.evidence] == ["model.sms", "message.urgency"]


@pytest.mark.parametrize("listing", [None, False])
def test_no_usable_signal_has_an_explicit_partial_unassessed_result(listing: bool | None) -> None:
    issue = ScanIssue(component="url_model", code="unavailable")
    result = aggregate_risk(urlhaus_listed=listing, issues=(issue,))
    assert result.risk_score is None
    assert result.risk_level is None
    assert result.status == ScanStatus.PARTIAL
    assert result.issues == (issue, ScanIssue(component="scoring", code="no_usable_signals"))


def test_successful_zero_model_estimate_is_distinct_from_unavailable() -> None:
    result = aggregate_risk(sms_probability=0)
    assert result.risk_score == 0
    assert result.risk_level == RiskLevel.LOW
    assert result.status == ScanStatus.COMPLETE


@pytest.mark.parametrize("probability", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.25"])
@pytest.mark.parametrize("field", ["url_probability", "sms_probability"])
def test_invalid_probabilities_fail_without_changing_the_assessment(field: str, probability: object) -> None:
    with pytest.raises(ValueError, match="Model probability"):
        aggregate_risk(**{field: probability})


@pytest.mark.parametrize("listing", [0, 1, "false"])
def test_rejects_nonboolean_threat_verdicts(listing: object) -> None:
    with pytest.raises(ValueError, match="URLhaus listing status"):
        aggregate_risk(urlhaus_listed=listing)


def test_rejects_conflicting_points_for_the_same_rule() -> None:
    conflicting = (
        RuleHit("example.repeated", 8, "First rule."),
        RuleHit("example.repeated", 16, "Conflicting rule."),
    )
    with pytest.raises(ValueError, match="inconsistent scoring evidence"):
        aggregate_risk(rule_hits=conflicting)
