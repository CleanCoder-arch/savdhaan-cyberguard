"""Run local scan analysis and combine it with available threat intelligence."""

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import perf_counter

from core.extractors.link_extractor import extract_links
from core.extractors.qr_decoder import InvalidUPIPayloadError, QRCodeError, decode_qr, parse_upi_uri
from core.models._prediction import ModelArtifactError
from core.models.artifacts import DEFAULT_MODEL_DIR, SMS_MODEL_FILENAME, URL_MODEL_FILENAME
from core.models.indian_scam_tagger import tag_indian_scams
from core.models.prediction import ModelPrediction
from core.models.sms_detector import SMSDetector
from core.models.url_detector import URLDetector
from core.schemas import (
    EvidenceItem,
    LinkReport,
    ModelExplanation,
    ScanIssue,
    ScanReport,
    ScanRequest,
    ScanType,
    ScoreAttribution,
    UPIPayload,
)
from core.scoring.aggregator import RiskResult, aggregate_risk
from core.scoring.rules_engine import RuleHit, evaluate_rules
from core.threat_intel.feed_cache import FeedCache
from core.threat_intel.live_providers import IntelProvider
from core.threat_intel.orchestrator import ThreatIntelRunner

_MAX_LINKS_PER_SCAN = 8
_NO_SIGNAL = ScanIssue(component="scoring", code="no_usable_signals")


@dataclass(frozen=True, slots=True)
class _AssessedLink:
    report: LinkReport
    prediction: ModelPrediction | None
    rule_hits: tuple[RuleHit, ...]
    listed: bool | None
    freshness: datetime | None


def _score_attribution(risk: RiskResult) -> tuple[ScoreAttribution, ...]:
    return tuple(
        ScoreAttribution(source=item.source, code=item.code, points=item.points, summary=item.summary)
        for item in risk.contributions
    )


class ScanEngine:
    """Assess validated requests with reusable local models and an optional intel cache."""

    def __init__(
        self,
        *,
        url_detector: URLDetector | None = None,
        sms_detector: SMSDetector | None = None,
        intel_runner: ThreatIntelRunner | None = None,
    ) -> None:
        self._url_detector = url_detector
        self._sms_detector = sms_detector
        self._intel_runner = intel_runner if intel_runner is not None else ThreatIntelRunner()
        self._startup_issues: dict[str, ScanIssue] = {}

    @classmethod
    def from_artifacts(cls, model_dir: str | Path = DEFAULT_MODEL_DIR) -> "ScanEngine":
        """Load trusted local model files and use the cache beside them."""
        directory = Path(model_dir)
        issues: dict[str, ScanIssue] = {}
        try:
            url_detector = URLDetector.load(directory / URL_MODEL_FILENAME)
        except (ModelArtifactError, OSError) as error:
            url_detector = None
            code = "artifact_invalid" if isinstance(error, ModelArtifactError) else "artifact_unavailable"
            issues["url_model"] = ScanIssue(component="url_model", code=code)
        try:
            sms_detector = SMSDetector.load(directory / SMS_MODEL_FILENAME)
        except (ModelArtifactError, OSError) as error:
            sms_detector = None
            code = "artifact_invalid" if isinstance(error, ModelArtifactError) else "artifact_unavailable"
            issues["sms_model"] = ScanIssue(component="sms_model", code=code)
        try:
            cache = FeedCache(directory / "threat_intel.sqlite3")
        except (OSError, sqlite3.Error):
            cache = None
            issues["threat_intel"] = ScanIssue(component="threat_intel", code="cache_unavailable")
        engine = cls(
            url_detector=url_detector,
            sms_detector=sms_detector,
            intel_runner=ThreatIntelRunner(cache=cache),
        )
        engine._startup_issues = issues
        return engine

    def _predict(
        self,
        detector: URLDetector | SMSDetector | None,
        component: str,
        text: str,
    ) -> tuple[ModelPrediction | None, tuple[ScanIssue, ...]]:
        if detector is None:
            issue = self._startup_issues.get(component) or ScanIssue(component=component, code="unavailable")
            return None, (issue,)
        try:
            result = detector.predict(text)
            if (
                not isinstance(result, ModelPrediction)
                or isinstance(result.probability, bool)
                or not 0 <= result.probability <= 1
                or not all(isfinite(value) for value in (result.log_odds, result.intercept, result.unshown_contribution))
            ):
                raise ValueError("Invalid detector result.")
            return result, ()
        except Exception:
            # Detector errors leave rule and threat-intelligence evidence available.
            return None, (ScanIssue(component=component, code="prediction_failed"),)

    async def _assess_link(self, url: str, *, offline: bool) -> _AssessedLink:
        prediction, issues = self._predict(self._url_detector, "url_model", url)
        intel = await self._intel_runner.run(url, offline=offline)
        whois = next((result for result in intel.results if result.provider is IntelProvider.WHOIS), None)
        urlhaus = next((result for result in intel.results if result.provider is IntelProvider.URLHAUS), None)
        domain_age = whois.domain_age_days if whois is not None else None
        listed = urlhaus.listed if urlhaus is not None else None
        cache_issue = self._startup_issues.get("threat_intel")
        if cache_issue is not None:
            issues += (cache_issue,)
        rule_hits = evaluate_rules(urls=(url,), domain_age_days=domain_age)
        risk = aggregate_risk(
            url_probability=prediction.probability if prediction is not None else None,
            rule_hits=rule_hits,
            urlhaus_listed=listed,
            issues=issues + intel.issues,
        )
        evidence = risk.evidence
        if domain_age is not None:
            evidence += (
                EvidenceItem(
                    source="threat_intel",
                    code="domain.registration_age",
                    summary=f"The registration lookup estimated the domain age at {domain_age} days.",
                ),
            )
        explanations = (ModelExplanation(model="url", prediction=prediction),) if prediction is not None else ()
        return _AssessedLink(
            report=LinkReport(
                url=url,
                risk_score=risk.risk_score,
                risk_level=risk.risk_level,
                evidence=evidence,
                issues=risk.issues,
                score_attribution=_score_attribution(risk),
                model_explanations=explanations,
            ),
            prediction=prediction,
            rule_hits=rule_hits,
            listed=listed,
            freshness=intel.freshness,
        )

    async def scan(self, request: ScanRequest, *, offline: bool = False) -> ScanReport:
        """Analyze the input and report failures alongside available evidence."""
        started = perf_counter()
        text = request.text
        upi: UPIPayload | None = None
        issues: list[ScanIssue] = []
        evidence: list[EvidenceItem] = []
        if request.scan_type is ScanType.QR and request.image_bytes is not None:
            try:
                payload = await asyncio.to_thread(decode_qr, request.image_bytes)
            except QRCodeError:
                issues.append(ScanIssue(component="qr", code="decode_failed"))
            else:
                if payload.lower().startswith("upi:"):
                    try:
                        upi = parse_upi_uri(payload)
                    except InvalidUPIPayloadError:
                        issues.append(ScanIssue(component="upi", code="invalid_payload"))
                else:
                    text = payload
        elif request.scan_type is ScanType.SCREENSHOT and text is None:
            issues.append(ScanIssue(component="ocr", code="review_required"))

        urls = (request.url,) if request.url is not None else extract_links(text) if text is not None else ()
        if len(urls) > _MAX_LINKS_PER_SCAN:
            issues.append(ScanIssue(component="link_analysis", code="limit_reached"))
        links = await asyncio.gather(
            *(self._assess_link(url, offline=offline) for url in urls[:_MAX_LINKS_PER_SCAN])
        )
        for link in links:
            issues.extend(issue for issue in link.report.issues if issue != _NO_SIGNAL)

        sms_prediction = None
        explanations: list[ModelExplanation] = []
        if text is not None and request.scan_type in {ScanType.SMS, ScanType.SCREENSHOT}:
            sms_prediction, model_issues = self._predict(self._sms_detector, "sms_model", text)
            issues.extend(model_issues)
            if sms_prediction is not None:
                explanations.append(ModelExplanation(model="sms", prediction=sms_prediction))
        tags = tag_indian_scams(text) if text is not None else ()
        for tag in tags:
            evidence.extend(
                EvidenceItem(source="taxonomy", code=rule_id, summary="Matched an Indian scam pattern.")
                for rule_id in tag.rule_ids
            )

        common_rules = evaluate_rules(text=text, upi=upi)
        sms_probability = sms_prediction.probability if sms_prediction is not None else None
        candidates = enumerate(links) if links else ((None, None),)
        assessments = (
            (
                index,
                link,
                aggregate_risk(
                    url_probability=link.prediction.probability if link is not None and link.prediction else None,
                    sms_probability=sms_probability,
                    rule_hits=(*common_rules, *link.rule_hits) if link is not None else common_rules,
                    urlhaus_listed=link.listed if link is not None else None,
                    issues=issues,
                ),
            )
            for index, link in candidates
        )
        scoring_link_index, strongest, risk = max(
            assessments, key=lambda entry: entry[2].risk_score if entry[2].risk_score is not None else -1
        )
        if strongest is not None and strongest.prediction is not None:
            explanations.append(ModelExplanation(model="url", prediction=strongest.prediction))
        freshness = min((link.freshness for link in links if link.freshness is not None), default=None)
        return ScanReport(
            scan_type=request.scan_type,
            status=risk.status,
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            score_attribution=_score_attribution(risk),
            model_explanations=tuple(explanations),
            scam_categories=tuple(tag.category for tag in tags),
            evidence=(*risk.evidence, *evidence),
            links=tuple(link.report for link in links),
            scoring_link_index=scoring_link_index if risk.risk_score is not None else None,
            upi=upi,
            issues=risk.issues,
            latency_ms=(perf_counter() - started) * 1000,
            feed_freshness=freshness,
        )
