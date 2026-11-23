"""Run the local Savdhaan CyberGuard scanner and analytics dashboard."""

import asyncio
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from app.ui_helpers import (
    INDIA_TIME,
    SCAN_TYPE_LABELS,
    benchmark_rows,
    history_row,
    load_benchmark_results,
    make_scan_request,
    redacted_summary_bytes,
)
from core.extractors.ocr_engine import OCRError, OCREngineUnavailable, extract_ocr
from core.guidance.redactor import generate_alert_card, redact_text
from core.guidance.rbi_helplines import (
    BANK_HELPLINES,
    CYBER_FRAUD_HELPLINE,
    CYBERCRIME_PORTAL,
    get_bank_helpline,
    get_rbi_liability_guidance,
    get_scam_playbook,
)
from core.models.artifacts import SMS_MODEL_FILENAME, URL_MODEL_FILENAME, configured_model_dir
from core.scanner import ScanEngine
from core.schemas import RiskLevel, ScanReport, ScanStatus, ScanType
from core.storage.repository import ScanRepository, configured_db_path
from core.validators import MAX_IMAGE_BYTES, MAX_TEXT_LENGTH, MAX_URL_LENGTH

_IMAGE_TYPES = ("png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff")
_IMAGE_LIMIT_MIB = MAX_IMAGE_BYTES // (1024 * 1024)
_STORAGE_ERRORS = (OSError, sqlite3.Error)
_SAMPLES = {
    "sample_bank": ("Fake bank login", "https://sbi-kyc-update.example.test/Verify/Login"),
    "sample_prize": ("Fake prize link", "https://claim-lottery.example.test/ReleaseFee"),
    "sample_kyc": (
        "Fake KYC SMS",
        "Your KYC expires today. Click the link to keep the account active and enter your OTP.",
    ),
    "sample_arrest": (
        "Digital arrest SMS",
        "You are under digital arrest. Stay on the video call and transfer money to a secure account.",
    ),
    "sample_upi": ("UPI refund SMS", "Scan this UPI QR code and approve the collect request to receive a refund."),
}
_RISK_COLORS = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "orange",
    RiskLevel.HIGH: "red",
    RiskLevel.CRITICAL: "red",
}
_OCR_LANGUAGES = {"English": "eng", "English and Hindi": "eng+hin"}
_INVALID_INPUT = {
    ScanType.URL: "Enter a valid HTTP or HTTPS URL.",
    ScanType.SMS: "Enter a nonempty message within the character limit.",
    ScanType.QR: "Upload a supported QR image no larger than 10 MiB.",
    ScanType.SCREENSHOT: "Enter the reviewed screenshot text within the character limit.",
}


def _clear_ocr() -> None:
    for key in ("ocr_identity", "ocr_text", "ocr_confidence"):
        st.session_state.pop(key, None)


def _clear_scan() -> None:
    for key in ("url_input", "sms_input", "scan_report", "scan_saved"):
        st.session_state.pop(key, None)
    _clear_ocr()
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def _sample_into(key: str, sample: str) -> None:
    st.session_state[key] = _SAMPLES[sample][1]
    st.session_state.pop("scan_report", None)


def _render_samples(input_key: str, sample_keys: tuple[str, ...]) -> None:
    columns = st.columns(len(sample_keys))
    for column, key in zip(columns, sample_keys, strict=True):
        column.button(_SAMPLES[key][0], key=key, on_click=_sample_into, args=(input_key, key))


def _upload(label: str, kind: str) -> bytes | None:
    version = st.session_state.get("upload_version", 0)
    upload = st.file_uploader(
        label,
        type=_IMAGE_TYPES,
        max_upload_size=_IMAGE_LIMIT_MIB,
        key=f"{kind}_upload_{version}",
    )
    if upload is not None:
        if 0 < upload.size <= MAX_IMAGE_BYTES:
            return upload.getvalue()
        st.error("Choose a nonempty image no larger than 10 MiB.")
    return None


def _render_screenshot() -> tuple[str, bool] | None:
    image = _upload("Screenshot", "screenshot")
    language_name = st.selectbox("OCR language", tuple(_OCR_LANGUAGES))
    language = _OCR_LANGUAGES[language_name]
    identity = (hashlib.sha256(image).digest(), language) if image is not None else None
    if st.session_state.get("ocr_identity") != identity:
        _clear_ocr()
    if st.button("Extract text", key="extract_ocr", disabled=image is None) and image is not None:
        _clear_ocr()
        with st.spinner("Extracting text locally…"):
            try:
                result = extract_ocr(image, language=language)
            except OCREngineUnavailable:
                st.error("Tesseract is unavailable. Install it locally and retry.")
            except OCRError:
                st.error("Text extraction failed. Check the image and the selected Tesseract language data.")
            else:
                st.session_state["ocr_identity"] = identity
                st.session_state["ocr_confidence"] = result.confidence
                st.session_state["ocr_text"] = result.text

    if identity is None or st.session_state.get("ocr_identity") != identity:
        return None
    confidence = st.session_state["ocr_confidence"]
    if confidence is not None:
        st.caption(f"Mean OCR confidence {confidence:.1f}%")
    text = st.text_area("Review and edit extracted text", key="ocr_text", max_chars=MAX_TEXT_LENGTH, height=150)
    submit = st.button("Analyze reviewed text", key="scan_screenshot", type="primary", disabled=not text.strip())
    return text, submit


def _artifact_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner=False, max_entries=4)
def _load_engine(directory: Path, stamps: tuple[tuple[int, int] | None, ...]) -> ScanEngine:
    """Cache only shared services, keyed by the model files' current stamps."""
    return ScanEngine.from_artifacts(directory)


def _run_scan(
    repository: ScanRepository | None,
    scan_type: ScanType,
    value: str | bytes,
    *,
    offline: bool,
) -> None:
    st.session_state.pop("scan_report", None)
    st.session_state.pop("scan_saved", None)
    try:
        request = make_scan_request(scan_type, value, ocr_reviewed=scan_type is ScanType.SCREENSHOT)
    except ValidationError:
        st.error(_INVALID_INPUT[scan_type])
        return
    with st.spinner("Analyzing…"):
        try:
            directory = configured_model_dir().resolve()
            stamps = tuple(_artifact_stamp(directory / name) for name in (URL_MODEL_FILENAME, SMS_MODEL_FILENAME))
            report = asyncio.run(_load_engine(directory, stamps).scan(request, offline=offline))
        except Exception:
            st.error("The scan could not finish. Try again with a valid input.")
            return
    st.session_state["scan_report"] = report
    if repository is not None:
        try:
            repository.save_scan(report)
        except _STORAGE_ERRORS:
            st.warning("The result is available for this session. History is temporarily unavailable.")
        else:
            st.session_state["scan_saved"] = True


def _render_inputs(repository: ScanRepository | None) -> None:
    st.subheader("Check a suspicious item")
    scan_type = st.radio(
        "Scan type", tuple(ScanType), format_func=SCAN_TYPE_LABELS.__getitem__, horizontal=True, key="scan_type"
    )
    live = st.toggle(
        "Check live threat intelligence",
        help="Live checks send the domain or hostname to external providers.",
        key="live_intel",
    )
    value: str | bytes | None
    if scan_type is ScanType.SCREENSHOT:
        screenshot = _render_screenshot()
        value, submit = screenshot if screenshot is not None else (None, False)
    else:
        if scan_type is ScanType.URL:
            value = st.text_input(
                "URL", key="url_input", max_chars=MAX_URL_LENGTH, placeholder="https://example.test/path"
            )
            _render_samples("url_input", ("sample_bank", "sample_prize"))
        elif scan_type is ScanType.SMS:
            value = st.text_area("SMS message", key="sms_input", max_chars=MAX_TEXT_LENGTH, height=150)
            _render_samples("sms_input", ("sample_kyc", "sample_arrest", "sample_upi"))
        else:
            value = _upload("QR image", "qr")
        submit = st.button("Analyze", key="scan_input", type="primary", disabled=not value)
    st.button("Clear scan", on_click=_clear_scan)
    if submit and value is not None:
        _run_scan(repository, scan_type, value, offline=not live)
    st.caption("History saves the scan type, result, and timing. Scan content stays in this session.")


def _render_explanation(report: ScanReport) -> None:
    with st.expander("Why this result?", expanded=True):
        if report.score_attribution:
            st.dataframe(
                [
                    {
                        "Signal": item.source.value.replace("_", " ").title(),
                        "Points": f"{item.points:+.2f}",
                        "Explanation": redact_text(item.summary),
                    }
                    for item in report.score_attribution
                ],
                hide_index=True,
            )
        else:
            st.caption("A score is available when the scan finds usable signals.")
        for item in report.model_explanations:
            prediction = item.prediction
            st.markdown(f"**{item.model.upper()} model estimate {prediction.probability:.1%}**")
            if prediction.contributions:
                st.dataframe(
                    [
                        {
                            "Feature": redact_text(feature.feature),
                            "Type": feature.kind.replace("_", " ").title(),
                            "Log-odds": f"{feature.contribution:+.4f}",
                        }
                        for feature in prediction.contributions
                    ],
                    hide_index=True,
                )
            st.caption(
                f"Model intercept {prediction.intercept:+.4f} · Remaining terms {prediction.unshown_contribution:+.4f}"
            )
        if len(report.links) > 1:
            st.dataframe(
                [
                    {
                        "Link": index + 1,
                        "Risk": link.risk_level.value if link.risk_level is not None else "Unassessed",
                        "Score": link.risk_score,
                        "Used in overall score": index == report.scoring_link_index,
                    }
                    for index, link in enumerate(report.links)
                ],
                hide_index=True,
            )


def _render_guidance(report: ScanReport | None) -> None:
    with st.expander("India reporting and bank guidance", expanded=False):
        if report is not None:
            for category in report.scam_categories:
                playbook = get_scam_playbook(category)
                st.markdown(f"**{playbook.title}**")
                st.write(playbook.actions[0])
        for action in get_scam_playbook().actions:
            st.write(action)
        st.link_button("National Cyber Crime Reporting Portal", CYBERCRIME_PORTAL)
        bank_key = st.selectbox("Bank helpline", tuple(BANK_HELPLINES), format_func=lambda key: BANK_HELPLINES[key].bank_name)
        bank = get_bank_helpline(bank_key)
        if bank is not None:
            st.text(" or ".join(bank.phone_numbers))
            st.link_button("Official bank contact information", bank.source_url)
        today = datetime.now(INDIA_TIME).date()
        transaction_date = st.date_input("Transaction date for RBI guidance", value=today, max_value=today)
        if transaction_date is not None:
            guidance = get_rbi_liability_guidance(transaction_date)
            st.caption(guidance.scope)
            for rule in guidance.rules:
                st.write(rule)
            st.link_button("Official RBI directions", guidance.source_url)


def _render_feedback(repository: ScanRepository, report: ScanReport) -> None:
    st.markdown("**Was this result helpful?**")
    yes, no = st.columns(2)
    helpful = yes.button("Yes", key=f"helpful_yes_{report.scan_id}")
    unhelpful = no.button("No", key=f"helpful_no_{report.scan_id}")
    try:
        feedback = (
            repository.upsert_feedback(report.scan_id, helpful)
            if helpful or unhelpful
            else repository.get_feedback(report.scan_id)
        )
    except (_STORAGE_ERRORS + (LookupError,)):
        st.warning("Feedback is temporarily unavailable.")
    else:
        if feedback is not None:
            st.caption(f"Latest answer saved · {'Yes' if feedback.helpful else 'No'}")


def _render_result(repository: ScanRepository | None) -> None:
    st.subheader("Scan result")
    report: ScanReport | None = st.session_state.get("scan_report")
    if report is None:
        st.write("Analyze a URL, message, or image to see the risk and a breakdown of the signals.")
        st.markdown(f"**Lost money?** Contact your bank and call **{CYBER_FRAUD_HELPLINE}**.")
        _render_guidance(None)
        return
    st.caption(f"Latest {SCAN_TYPE_LABELS[report.scan_type]} scan")
    if report.risk_level is not None:
        st.badge(report.risk_level.value, color=_RISK_COLORS[report.risk_level])
    st.metric("Risk score", f"{report.risk_score}/100" if report.risk_score is not None else "Unassessed")
    if report.status is ScanStatus.PARTIAL:
        st.warning("Partial scan. Review the available evidence below.")
        st.text(
            "\n".join(
                f"{issue.component.upper().replace('_', ' ')} · {issue.code.replace('_', ' ')}"
                for issue in report.issues
            )
        )
    if report.scam_categories:
        st.caption(" · ".join(get_scam_playbook(category).title for category in report.scam_categories))
    _render_explanation(report)
    _render_guidance(report)
    summary, card = st.columns(2)
    summary.download_button(
        "Download redacted summary",
        redacted_summary_bytes(report),
        file_name="savdhaan-summary.json",
        mime="application/json",
        on_click="ignore",
    )
    card.download_button(
        "Download PNG alert card",
        generate_alert_card(report),
        file_name="savdhaan-scam-alert.png",
        mime="image/png",
        on_click="ignore",
    )
    if repository is not None and st.session_state.get("scan_saved"):
        _render_feedback(repository, report)


def _render_history(repository: ScanRepository | None) -> None:
    st.subheader("Recent scans")
    if repository is None:
        st.info("Scan history is unavailable.")
        return
    try:
        scans = repository.list_scans(limit=50)
    except _STORAGE_ERRORS:
        st.warning("Scan history is temporarily unavailable.")
        return
    if scans:
        st.dataframe([history_row(scan) for scan in scans], hide_index=True)
    else:
        st.info("Saved scans will appear here.")


def _render_activity(repository: ScanRepository | None) -> None:
    st.subheader("Scan activity")
    if repository is None:
        st.info("Scan activity is unavailable.")
        return
    try:
        activity = repository.activity_by_day(days=30, utc_offset_minutes=330)
    except _STORAGE_ERRORS:
        st.warning("Scan activity is temporarily unavailable.")
        return
    count, partial, helpful = st.columns(3)
    responses = sum(day.feedback_count for day in activity)
    helpful_count = sum(day.helpful_count for day in activity)
    count.metric("Scans in 30 days", sum(day.scan_count for day in activity))
    partial.metric("Partial scans", sum(day.partial_count for day in activity))
    helpful.metric("Helpful responses", f"{helpful_count}/{responses}" if responses else "—")
    if any(day.scan_count for day in activity):
        st.bar_chart([{"Date": day.day, "Scans": day.scan_count} for day in activity], x="Date", y="Scans")


def _render_benchmarks() -> None:
    st.subheader("Locked benchmark results")
    try:
        results = load_benchmark_results()
    except FileNotFoundError:
        st.info("Benchmark results appear here after a grouped evaluation is run.")
        return
    except (OSError, ValueError):
        st.warning("Benchmark results are unavailable. Check the local results file.")
        return
    if results.dataset_kind == "synthetic_demo":
        st.info("Synthetic demonstration results. Treat these numbers as a check of the evaluation workflow.")
    st.caption("Metrics use held-out groups. Thresholds were selected on validation data before evaluation.")
    st.dataframe(benchmark_rows(results), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Savdhaan CyberGuard", page_icon="🛡️", layout="wide")
    st.title("Savdhaan CyberGuard")
    st.caption("Scam and phishing checks for links, messages, and images.")
    try:
        repository = ScanRepository(configured_db_path())
    except _STORAGE_ERRORS:
        repository = None
        st.warning("History storage is unavailable. Scanner checks are still available.")
    scanner, history, analytics = st.tabs(("Scanner", "History", "Analytics"))
    with scanner:
        inputs, result = st.columns((1, 1.15), gap="large")
        with inputs, st.container(border=True):
            _render_inputs(repository)
        with result, st.container(border=True):
            _render_result(repository)
    with history:
        _render_history(repository)
    with analytics:
        _render_activity(repository)
        _render_benchmarks()


if __name__ == "__main__":
    main()
