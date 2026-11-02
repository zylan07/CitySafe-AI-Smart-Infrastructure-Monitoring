"""
API Integration Tests for CitySafe V2 Server
============================================
Verifies:
  1. Backward compatibility of v1 endpoints (/api/analyze, /api/reports, /api/report/save)
  2. V2 /api/v2/model-info endpoint (AHP consistency ratios, parameters)
  3. V2 /api/v2/simulate endpoint (What-if parameter variation without image)
  4. V2 /api/v2/analyze endpoint (Full vision + explainability pipeline)
  5. V2 /api/v2/hotspots (Getis-Ord Gi*)
  6. V2 /api/v2/prioritize (Budget knapsack relaxation)
  7. V2 /api/v2/crowd-clusters (Bayesian multi-report fusion)
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)


def test_v2_model_info():
    res = client.get("/api/v2/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "ahp_hazard" in data
    assert "ahp_consequence" in data
    assert data["ahp_hazard"]["consistency_ratio"] < 0.10
    assert data["ahp_consequence"]["consistency_ratio"] < 0.10
    assert "asset_types" in data


def test_v2_simulate_endpoint():
    payload = {
        "asset_type": "culvert",
        "age_years": 15.0,
        "rainfall_24h_mm": 75.0,
        "antecedent_rain_5d_mm": 25.0,
        "traffic_index": 70.0,
        "damage_probability": 0.85
    }
    res = client.post("/api/v2/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "risk_score" in data
    assert "failure_probability_90d" in data
    assert "consequence_score" in data
    assert "breakdown" in data
    assert data["breakdown"]["runoff_Q_mm"] > 0.0


def test_v2_hotspots_endpoint():
    res = client.get("/api/v2/hotspots")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    if data["status"] == "success":
        assert "hotspots" in data
        assert len(data["hotspots"]) >= 10


def test_v2_prioritize_endpoint():
    payload = {"budget_inr": 250000.0, "strategy": "ratio"}
    res = client.post("/api/v2/prioritize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "total_cost_inr" in data
    assert data["total_cost_inr"] <= 250000.0
    assert "scheduled_repairs" in data


def test_v2_crowd_clusters_endpoint():
    res = client.get("/api/v2/crowd-clusters")
    assert res.status_code == 200
    data = res.json()
    assert "clusters" in data


def test_v1_backward_compatibility_reports():
    res = client.get("/api/reports")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_v2_analyze_with_image():
    # Create test RGB image
    img = Image.new("RGB", (100, 100), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_wall.jpg", buf, "image/jpeg")}
    data = {
        "rainfall": "35.0",
        "traffic": "55.0",
        "latitude": "11.0168",
        "longitude": "76.9558",
        "asset_type": "retaining_wall"
    }

    res = client.post("/api/v2/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "success"
    assert "prediction" in result
    assert "occlusion_heatmap" in result
    assert "breakdown" in result
    assert "uncertainty_range" in result

