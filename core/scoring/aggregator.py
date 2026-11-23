"""Combine independent score signals with a signed, auditable contribution for each."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import floor, fsum
from numbers import Real

from core.schemas import EvidenceItem, EvidenceSource, RiskLevel, ScanIssue, ScanStatus
from core.scoring.rules_engine import RuleHit

ML_MAX_POINTS = 70
URLHAUS_LISTED_POINTS = 55
MEDIUM_THRESHOLD = 25
HIGH_THRESHOLD = 55
CRITICAL_THRESHOLD = 80


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    source: EvidenceSource
    code: str
    points: float
    summary: str


@dataclass(frozen=True, slots=True)
class RiskResult:
    risk_score: int | None
    risk_level: RiskLevel | None
    contributions: tuple[ScoreContribution, ...]
    issues: tuple[ScanIssue, ...]

    @property
    def status(self) -> ScanStatus:
        return ScanStatus.PARTIAL if self.issues else ScanStatus.COMPLETE

    @property
    def evidence(self) -> tuple[EvidenceItem, ...]:
        return tuple(
            EvidenceItem(
                source=item.source,
                code=item.code,
                summary=item.summary,
            )
            for item in self.contributions
        )


def _model_contributions(
    url_probability: float | None, sms_probability: float | None
) -> list[ScoreContribution]:
    estimates = []
    for name, value in (("url", url_probability), ("sms", sms_probability)):
        if value is None:
            continue
        if not isinstance(value, Real) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError("Model probability must be a finite number from 0 to 1.")
        estimates.append((name, float(value)))
    if not estimates:
        return []

    strongest, _ = max(estimates, key=lambda estimate: estimate[1])
    contributions = []
    for name, probability in estimates:
        if name == strongest:
            points = probability * ML_MAX_POINTS
            reason = "This is the strongest available model estimate."
        else:
            points = 0.0
            reason = "The score uses the strongest available model estimate once."
        contributions.append(
            ScoreContribution(
                EvidenceSource.ML,
                f"model.{name}",
                points,
                f"The {name.upper()} model estimated {probability:.1%} scam probability. {reason}",
            )
        )
    return contributions


def _risk_level(score: int) -> RiskLevel:
    if score >= CRITICAL_THRESHOLD:
        return RiskLevel.CRITICAL
    if score >= HIGH_THRESHOLD:
        return RiskLevel.HIGH
    if score >= MEDIUM_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def aggregate_risk(
    *,
    url_probability: float | None = None,
    sms_probability: float | None = None,
    rule_hits: Iterable[RuleHit] = (),
    urlhaus_listed: bool | None = None,
    issues: Iterable[ScanIssue] = (),
) -> RiskResult:
    """Apply the score policy to available signals and retain failed component issues.

    The strongest model estimate contributes up to 70 points. Rules contribute individually,
    and an active URLhaus match on a checked host contributes 55. Thresholds classify the
    bounded integer score. Callers provide issues for failed applicable checks; a None
    signal can also mean that the check was inapplicable to the input.
    """
    if urlhaus_listed is not None and type(urlhaus_listed) is not bool:
        raise ValueError("URLhaus listing status must be true, false, or unavailable.")

    contributions = _model_contributions(url_probability, sms_probability)
    unique_rules: dict[str, RuleHit] = {}
    for hit in rule_hits:
        previous = unique_rules.setdefault(hit.code, hit)
        if previous != hit:
            raise ValueError("The same rule identifier has inconsistent scoring evidence.")
    contributions.extend(
        ScoreContribution(EvidenceSource.RULES, hit.code, float(hit.points), hit.summary)
        for hit in unique_rules.values()
    )
    has_signal = bool(contributions) or urlhaus_listed is True
    if urlhaus_listed is not None:
        contributions.append(
            ScoreContribution(
                EvidenceSource.THREAT_INTEL,
                "urlhaus.active_host_listing",
                float(URLHAUS_LISTED_POINTS if urlhaus_listed else 0),
                (
                    "URLhaus reports an active listing for a URL on the checked host. "
                    "A host match can refer to a different URL on that host."
                    if urlhaus_listed
                    else "URLhaus found no active URL listing on the checked host."
                ),
            )
        )

    unique_issues = dict.fromkeys(issues)
    if not has_signal:
        unique_issues[ScanIssue(component="scoring", code="no_usable_signals")] = None
        return RiskResult(None, None, tuple(contributions), tuple(unique_issues))

    raw_score = fsum(item.points for item in contributions)
    score = min(100, max(0, floor(raw_score + 0.5)))
    if score != raw_score:
        clamped = not 0 <= raw_score <= 100
        contributions.append(
            ScoreContribution(
                EvidenceSource.RULES,
                "scoring.clamp" if clamped else "scoring.rounding",
                score - raw_score,
                "The risk score is bounded to 0–100." if clamped else "The risk score is rounded to a whole number.",
            )
        )
    return RiskResult(score, _risk_level(score), tuple(contributions), tuple(unique_issues))
