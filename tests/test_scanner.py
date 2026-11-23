import asyncio
from datetime import datetime, timedelta, timezone
from math import fsum, log

import pytest

from core.guidance.redactor import build_redacted_report
from core.models.artifacts import URL_MODEL_FILENAME
from core.models.prediction import ModelPrediction
from core.models.sms_detector import SMSDetector
from core.models.url_detector import URLDetector
from core.schemas import RiskLevel, ScanIssue, ScanReport, ScanRequest, ScanStatus
from core.scanner import ScanEngine
from core.threat_intel.feed_cache import FeedCache
from core.threat_intel.live_providers import IntelProvider, IntelResult, IntelStatus
from core.threat_intel.orchestrator import ThreatIntelRunner
from scripts.bootstrap_models import bootstrap_models


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scanner-models")
    bootstrap_models(directory)
    return directory


def test_sms_scan_includes_a_complete_score_and_reconcilable_model_attribution(artifacts) -> None:
    engine = ScanEngine.from_artifacts(artifacts)
    request = ScanRequest(scan_type="sms", text="Scan this QR code to receive your refund.")
    report = asyncio.run(engine.scan(request, offline=True))

    assert report.status is ScanStatus.COMPLETE
    assert report.issues == ()
    assert report.risk_score == pytest.approx(fsum(item.points for item in report.score_attribution))
    assert "upi_qr" in report.scam_categories
    explanation, = report.model_explanations
    assert explanation.model == "sms"
    prediction = explanation.prediction
    assert prediction.log_odds == pytest.approx(
        prediction.intercept + prediction.unshown_contribution
        + fsum(item.contribution for item in prediction.contributions)
    )
    assert ScanReport.model_validate_json(report.model_dump_json()) == report
    assert "model_explanations" not in build_redacted_report(report)


def test_offline_url_has_an_explanation_and_explicit_unavailable_intel(artifacts) -> None:
    engine = ScanEngine.from_artifacts(artifacts)
    url = "https://www.example.com/Agenda?Session=PrivateValue"
    report = asyncio.run(engine.scan(ScanRequest(scan_type="url", url=url), offline=True))

    assert report.status is ScanStatus.PARTIAL
    assert report.risk_score is not None
    assert report.links[0].url == url
    assert report.links[0].model_explanations[0].model == "url"
    assert report.issues == (ScanIssue(component="whois", code="offline"), ScanIssue(component="urlhaus", code="offline"))
    assert report.feed_freshness is None
    assert "PrivateValue" not in repr(report)
    assert "PrivateValue" not in repr(build_redacted_report(report))


def test_rate_limit_leaves_whois_rules_and_completes_from_cache(artifacts, tmp_path) -> None:
    cache = FeedCache(tmp_path / "lookup.db")
    checked_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    class Whois:
        async def lookup(self, target: str) -> IntelResult:
            assert target == "example.com"
            return IntelResult(IntelProvider.WHOIS, IntelStatus.OK, checked_at, domain_age_days=2)

    class Urlhaus:
        async def lookup(self, target: str) -> IntelResult:
            assert target == "pay.example.com"
            return IntelResult(IntelProvider.URLHAUS, IntelStatus.RATE_LIMITED, checked_at)

    engine = ScanEngine(
        url_detector=URLDetector.load(artifacts / URL_MODEL_FILENAME),
        intel_runner=ThreatIntelRunner(whois=Whois(), urlhaus=Urlhaus(), cache=cache),
    )
    request = ScanRequest(scan_type="url", url="https://pay.example.com/Privacy?token=private")
    partial = asyncio.run(engine.scan(request))
    assert partial.status is ScanStatus.PARTIAL
    assert partial.feed_freshness == checked_at
    assert partial.issues == (ScanIssue(component="urlhaus", code="rate_limited"),)
    assert any(item.code == "domain.recent_registration" for item in partial.score_attribution)

    cache.put("pay.example.com", IntelResult(IntelProvider.URLHAUS, IntelStatus.OK, checked_at, listed=True))
    cached = asyncio.run(engine.scan(request, offline=True))
    assert cached.status is ScanStatus.COMPLETE
    assert cached.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert cached.risk_score > partial.risk_score
    assert cached.feed_freshness == checked_at


def test_corrupt_model_preserves_independent_rule_evidence(tmp_path) -> None:
    (tmp_path / URL_MODEL_FILENAME).write_bytes(b"not a detector artifact")
    engine = ScanEngine.from_artifacts(tmp_path)
    report = asyncio.run(engine.scan(ScanRequest(scan_type="url", url="http://192.0.2.10/"), offline=True))

    assert report.status is ScanStatus.PARTIAL
    assert ScanIssue(component="url_model", code="artifact_invalid") in report.issues
    assert report.risk_score is not None
    assert report.score_attribution
    assert report.model_explanations == ()


def test_missing_model_and_no_usable_evidence_returns_unassessed_partial(tmp_path) -> None:
    engine = ScanEngine.from_artifacts(tmp_path)
    report = asyncio.run(engine.scan(ScanRequest(scan_type="sms", text="A quiet morning."), offline=True))

    assert report.status is ScanStatus.PARTIAL
    assert ScanIssue(component="sms_model", code="artifact_unavailable") in report.issues
    assert ScanIssue(component="scoring", code="no_usable_signals") in report.issues
    assert report.risk_score is None
    assert report.risk_level is None


def test_raw_screenshot_awaits_review_before_model_scoring() -> None:
    report = asyncio.run(ScanEngine().scan(ScanRequest(scan_type="screenshot", image_bytes=b"image")))
    assert report.status is ScanStatus.PARTIAL
    assert ScanIssue(component="ocr", code="review_required") in report.issues
    assert report.model_explanations == ()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_one_model_failure_does_not_drop_the_other_model_or_links(artifacts, monkeypatch, failure) -> None:
    def fail_prediction(*args, **kwargs):
        if failure == "exception":
            raise AttributeError("private internal diagnostic")
        return ModelPrediction(float("nan"), 0.0, 0.0, ())

    monkeypatch.setattr(SMSDetector, "predict", fail_prediction)
    engine = ScanEngine.from_artifacts(artifacts)
    report = asyncio.run(
        engine.scan(ScanRequest(scan_type="sms", text="Read the update at https://example.com/Agenda."), offline=True)
    )
    assert report.risk_score is not None
    assert ScanIssue(component="sms_model", code="prediction_failed") in report.issues
    assert [item.model for item in report.model_explanations] == ["url"]
    assert report.links[0].url.endswith("/Agenda")
    assert "private internal diagnostic" not in report.model_dump_json()


def test_combined_scan_preserves_independent_link_signals() -> None:
    def prediction(probability: float) -> ModelPrediction:
        score = log(probability / (1 - probability))
        return ModelPrediction(probability, score, score, ())

    class URLModel:
        def predict(self, url: str) -> ModelPrediction:
            return prediction(0.99 if "first.example.com" in url else 0.15)

    class SMSModel:
        def predict(self, message: str) -> ModelPrediction:
            return prediction(0.99)

    class Whois:
        async def lookup(self, target: str) -> IntelResult:
            return IntelResult(IntelProvider.WHOIS, IntelStatus.OK, datetime.now(timezone.utc), domain_age_days=400)

    class Urlhaus:
        async def lookup(self, target: str) -> IntelResult:
            return IntelResult(
                IntelProvider.URLHAUS, IntelStatus.OK, datetime.now(timezone.utc), listed=target == "second.example.org"
            )

    engine = ScanEngine(
        url_detector=URLModel(),
        sms_detector=SMSModel(),
        intel_runner=ThreatIntelRunner(whois=Whois(), urlhaus=Urlhaus()),
    )
    request = ScanRequest(scan_type="sms", text="See https://first.example.com/ and https://second.example.org/")
    report = asyncio.run(engine.scan(request))

    assert report.links[0].risk_score > report.links[1].risk_score
    assert report.scoring_link_index == 1
    assert report.risk_level is RiskLevel.CRITICAL
    assert report.risk_score == 100
    assert [item.prediction.probability for item in report.model_explanations] == [0.99, 0.15]
    assert any(item.code == "domain.registration_age" for item in report.links[1].evidence)
    assert any(item.code == "urlhaus.active_host_listing" and item.points > 0 for item in report.score_attribution)
