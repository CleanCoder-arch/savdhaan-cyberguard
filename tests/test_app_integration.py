from pathlib import Path

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import APISettings, create_app
from core.storage.repository import ScanRepository
from scripts.bootstrap_models import bootstrap_models


def test_api_scan_appears_as_metadata_in_streamlit_history(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    bootstrap_models(models)
    scans = tmp_path / "scans.sqlite3"
    controls = tmp_path / "api-control.sqlite3"
    key = "fixture-key-for-the-integration-test"
    settings = APISettings(api_keys=(key,), db_path=scans, control_db_path=controls, model_dir=models, offline=True)
    request_content = "Your KYC has expired. Click the link at https://example.test/Agenda?ticket=Private987."
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        response = client.post(
            "/api/v1/scan", headers={"X-API-Key": key}, json={"scan_type": "sms", "text": request_content},
        )
        assert response.status_code == 200
        report = response.json()
        assert report["risk_score"] is not None
        assert report["score_attribution"]

    metadata = ScanRepository(scans).get_scan(report["scan_id"])
    assert metadata is not None
    assert metadata.risk_score == report["risk_score"]
    assert b"Private987" not in scans.read_bytes()
    assert request_content.encode() not in scans.read_bytes()
    assert key.encode() not in controls.read_bytes()

    monkeypatch.setenv("SAVDHAAN_DB_PATH", str(scans))
    monkeypatch.setenv("SAVDHAAN_MODEL_DIR", str(models))
    monkeypatch.setenv("SAVDHAAN_BENCHMARK_RESULTS", str(tmp_path / "no-benchmark.json"))
    file = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(file)).run(timeout=25)
    assert not app.exception
    assert any(
        report["scan_id"] in table.value["Scan ID"].tolist()
        for table in app.dataframe
        if "Scan ID" in table.value
    )
    assert all("Private987" not in str(table.value) for table in app.dataframe)
