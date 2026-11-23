# Savdhaan CyberGuard

Savdhaan checks suspicious links, messages, and images using local models and rules. FastAPI and Streamlit share the same scan engine and metadata history.

## Run locally

From the repository root, use a Python 3.11+ environment.

```powershell
python -m pip install -r requirements.txt
python scripts/bootstrap_models.py
python -m benchmarks --demo
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The bootstrap trains starter models on synthetic examples. The demo benchmark exercises validation and held-out evaluation on synthetic data, and Analytics labels it as a demo. Install Tesseract locally for screenshot OCR. The Scanner lets you correct the extracted text before analysis. Select Hindi OCR when Tesseract has the `hin` language data installed.

## FastAPI

Set an API key and start the service in a separate terminal.

```powershell
$env:SAVDHAAN_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
python -m uvicorn api.main:app --no-access-log
```

- `GET /api/v1/health` reports model, storage, authentication, and webhook readiness.
- `POST /api/v1/scan` accepts a JSON scan request with an `X-API-Key` header. The default sliding window allows ten requests per key per hour. Use `?stream=true` or `Accept: text/event-stream` for SSE progress and results, and `?offline=true` to restrict a request to local analysis and fresh cached intelligence.
- `GET /api/v1/webhooks/whatsapp` handles Meta subscription verification. Set `SAVDHAAN_WHATSAPP_VERIFY_TOKEN` to the matching token.
- `POST /api/v1/webhooks/whatsapp` verifies Meta signatures using `SAVDHAAN_WHATSAPP_APP_SECRET` and acknowledges processed inbound messages. Set `SAVDHAAN_WHATSAPP_ACCESS_TOKEN` to enable inbound image downloads for QR analysis.

FastAPI serves the request schemas at `/docs`. Set `SAVDHAAN_API_KEYS` for additional comma-separated keys, and `SAVDHAAN_RATE_LIMIT` and `SAVDHAAN_RATE_WINDOW_SECONDS` to configure the sliding window. `SAVDHAAN_OFFLINE=1` applies offline mode to all API and webhook scans. Live URLhaus checks use `URLHAUS_AUTH_KEY`.

## Storage and benchmarks

History defaults to `artifacts/scans.sqlite3`. API and Streamlit share `SAVDHAAN_DB_PATH` and `SAVDHAAN_MODEL_DIR`. History stores only scan metadata and a separate latest helpfulness answer per scan. The API uses `SAVDHAAN_API_DB_PATH` for its rate counters and webhook delivery digests.

To evaluate your own data, supply UTF-8 CSV files with `text,label` headers and integer labels `0` for benign or `1` for scam.

```powershell
python -m benchmarks --url-data urls.csv --sms-data messages.csv
pytest -q
```

The splitter assigns each URL registrable domain and each near-duplicate SMS template group to one partition. The benchmark selects thresholds on validation groups and applies them to held-out groups. It writes detector metrics and latency to `artifacts/benchmarks/benchmark_results.json`. Set `SAVDHAAN_BENCHMARK_RESULTS` to display a results file from a different location.
