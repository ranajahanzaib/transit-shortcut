from fastapi.testclient import TestClient

from transitshortcut.api import app
from transitshortcut.experiments import train

client = TestClient(app)


def test_setup_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSITSHORTCUT_HOME", str(tmp_path))
    assert client.get("/api/health").json()["dataset_prepared"] is False
    assert client.get("/api/experiments").json() == {"data": []}
    assert client.get("/api/dataset/summary").status_code == 404


def test_api_data_and_reports(cohort):
    run = train("multiclass")
    run_id = run["id"]
    assert client.get("/api/health").json()["models_trained"]
    assert client.get("/api/targets?limit=2").json()["total"] == 150
    assert len(client.get("/api/targets?limit=2").json()["items"]) == 2
    assert client.get("/api/targets?limit=-1").status_code == 422
    assert client.get("/api/targets/unknown").status_code == 404
    assert client.get("/api/experiments/unknown").status_code == 404
    flux = client.get("/api/targets/test-0/flux?view=flux_global").json()["data"]
    assert flux["unavailable"] and flux["annotation"]
    raw = client.get("/api/targets/test-0/flux?max_points=100").json()["data"]
    assert len(raw["x"]) <= 100 and raw["x_label"] == "Sample index"
    assert client.get("/api/targets/test-0/flux?view=bad").status_code == 422
    assert (
        client.get("/api/targets/train-0/predictions", params={"run_id": run_id}).status_code == 404
    )
    rows = client.get(f"/api/experiments/{run_id}/predictions?true_class=NO_SIGNAL&limit=2").json()
    assert rows["total"] == 10 and len(rows["items"]) == 2
    assert rows["metrics"]["signal"]["n"] == 10
    assert client.get("/api/targets/test-0").json()["data"]["metadata"]["period_days"] is None
    for format in ["json", "markdown", "csv"]:
        report = client.get(f"/api/reports/{run_id}?format={format}")
        assert report.status_code == 200 and run_id in report.headers["content-disposition"]


def test_prediction_discovery_and_global_sort(cohort):
    result = train("multiclass")
    prefix = f"/api/experiments/{result['id']}/predictions"
    assert client.get(prefix + "?mission=unknown").status_code == 422
    rows = client.get(prefix + "?sort_ablation=0&limit=200").json()["items"]
    deltas = [r["ablation_0_score_delta"] for r in rows]
    assert deltas == sorted(deltas, reverse=True)
    for discovery in ("shortcut-only", "signal-only", "both-wrong"):
        response = client.get(prefix, params={"discovery": discovery, "limit": 200}).json()
        for row in response["items"]:
            a = row["shortcut_prediction"] == row["disposition"]
            b = row["signal_prediction"] == row["disposition"]
            assert (
                (a and not b)
                if discovery == "shortcut-only"
                else ((not a and b) if discovery == "signal-only" else (not a and not b))
            )
        assert response["total"] == len(response["items"])
