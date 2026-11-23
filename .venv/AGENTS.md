# AGENTS.md

## Project Overview
An Indian AI Scam & Phishing Detection Platform combining CyberGuard's explainable ML/rule engine, reproducible benchmarks, and metadata-only SQLite storage with Savdhaan AI's Indian scam taxonomy, multi-channel FastAPI/WhatsApp interfaces, live threat intel, and RBI/1930 guidance.

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `pytest -q`
- Run FastAPI server: `uvicorn api.main:app --reload`
- Run Streamlit UI: `streamlit run app/streamlit_app.py`

## Non-Negotiable Rules
1. **No Fake Threat Intel:** Never hardcode mock threat-intel verdicts in production paths. Every external check (`WHOIS`, `URLhaus`, `Safe Browsing`) must use real requests with timeouts and degrade gracefully into partial results when offline or rate-limited.
2. **Privacy by Design (DPDP Act 2023):** Never store raw SMS messages, screenshots, decoded QR payloads, UPI IDs, phone numbers, or full URLs with query tokens in SQLite. Store only non-sensitive metadata (`scan_id`, `scan_type`, `timestamp`, `status`, `risk_level`, `risk_score`, `scam_category`, `latency_ms`, `feed_freshness`).
3. **Keep `core/` Decoupled:** Neither Streamlit nor FastAPI may contain scoring or ML logic. Both must import from `core/`.
4. **Test Gate:** Before finishing any step, run `pytest -q` and ensure every test passes.