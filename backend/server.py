"""
CitySafe / ZylanZest V2 — Production Backend API Server
======================================================
FastAPI server serving:
  - Backward-compatible v1 endpoints: /api/analyze, /api/report/save, /api/reports, /api/notifications
  - Upgraded v2 endpoints:
      * POST /api/v2/analyze (Full pipeline: calibrated vision, occlusion heatmap, context, risk, uncertainty)
      * POST /api/v2/simulate (Controlled what-if parameter simulation without image requirement)
      * GET  /api/v2/model-info (Rigorous mathematical metrics, AHP matrices, CR, literature parameters)
      * GET  /api/v2/hotspots (Getis-Ord Gi* spatial statistics)
      * POST /api/v2/prioritize (Budget-constrained repair scheduling via greedy knapsack relaxation)
      * GET  /api/v2/crowd-clusters (Multi-report Bayesian evidence fusion)
"""

import os
import sys
import io
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from PIL import Image

from backend.config import (
    ASSET_TYPES,
    WEIBULL_SHAPE_K,
    SCORE_HIGH_THRESHOLD,
    SCORE_MODERATE_THRESHOLD,
    HAZARD_MULTIPLIER_B
)
from backend.risk_engine import (
    evaluate_infrastructure_risk,
    HAZARD_WEIGHTS,
    HAZARD_LAMBDA_MAX,
    HAZARD_CI,
    HAZARD_CR,
    CONSEQUENCE_WEIGHTS,
    CONSEQUENCE_LAMBDA_MAX,
    CONSEQUENCE_CI,
    CONSEQUENCE_CR,
    BETA_VECTOR
)
from backend.context import get_spatial_context
from backend.spatial import (
    calculate_getis_ord_gi_star,
    prioritize_repairs,
    fuse_crowd_evidence
)
from backend.vision_service import (
    predict_damage,
    generate_occlusion_heatmap,
    _model_source
)

# Persistence Paths
os.makedirs("backend/database", exist_ok=True)
os.makedirs("backend/model", exist_ok=True)
REPORTS_FILE = "backend/database/reports.json"
METRICS_FILE = "backend/model/vision_metrics.json"
OPTIMIZED_METRICS_FILE = "backend/model/vision_metrics_optimized.json"

if not os.path.exists(REPORTS_FILE):
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

app = FastAPI(
    title="CitySafe Infrastructure Risk Platform",
    description="Mathematical & AI-Powered Vulnerability Assessment of Urban Infrastructure",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: Load reports with backward-compatible migration
def load_reports() -> List[Dict[str, Any]]:
    if not os.path.exists(REPORTS_FILE):
        return []
    try:
        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
            reports = json.load(f)

        # Migrate legacy records if missing required fields
        for r in reports:
            if "status" not in r:
                r["status"] = "Pending"
            if "action_log" not in r:
                r["action_log"] = [{"action": "Created", "timestamp": r.get("timestamp", datetime.now().isoformat())}]
            if "asset_type" not in r:
                r["asset_type"] = r.get("breakdown", {}).get("asset_type", "road")

        return reports
    except Exception as e:
        print(f"Error loading reports: {e}")
        return []


def save_reports_to_disk(reports: List[Dict[str, Any]]) -> None:
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)


# ==============================================================================
# 1. BACKWARD-COMPATIBLE V1 ENDPOINTS
# ==============================================================================

@app.post("/api/analyze")
async def legacy_analyze_image(
    file: UploadFile = File(...),
    rainfall: float = Form(...),
    traffic: float = Form(...)
):
    """
    Backward-compatible v1 analyze endpoint.
    Maintains original interface while removing unseeded pseudo-randomness in favor of
    the deterministic mathematical risk pipeline.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image format"}

    # Calibrated vision prediction
    pred, damage_p, conf, source = predict_damage(image)

    # Evaluate using deterministic risk engine
    report_id = str(uuid.uuid4())
    risk_eval = evaluate_infrastructure_risk(
        report_id=report_id,
        asset_type="road",
        damage_probability=damage_p,
        rainfall_forecast_24h_mm=rainfall,
        antecedent_rain_5d_mm=rainfall * 0.3,
        traffic_index=traffic,
        asset_age_years=8.0,
        context_source="legacy_v1_endpoint"
    )

    return {
        "prediction": risk_eval["prediction"],
        "confidence": round(conf, 4),
        "risk_score": risk_eval["risk_score"],
        "priority": risk_eval["priority"],
        "failure_probability": risk_eval["failure_probability"],
        "estimated_failure_days": risk_eval["estimated_failure_days"],
        "explanation": f"Evaluated via {source}. Environmental risk assessed under {rainfall:.1f}mm rainfall.",
        "recommended_action": risk_eval["recommended_action"],
        "impact_score": f"{risk_eval['priority']} (Consequence: {risk_eval['consequence_score']:.2f})",
        "breakdown": risk_eval["breakdown"]
    }


class ReportSave(BaseModel):
    image_name: str
    prediction: str
    risk_score: int
    latitude: float
    longitude: float
    failure_probability: int
    estimated_failure_days: int
    explanation: str
    recommended_action: str = "Monitor condition"
    impact_score: str = "Low"
    asset_type: str = "road"
    breakdown: Optional[Dict[str, Any]] = None


@app.post("/api/report/save")
async def save_report(report: ReportSave):
    try:
        reports = load_reports()
        new_report = {
            "id": str(uuid.uuid4()),
            "image_name": report.image_name,
            "asset_type": report.asset_type,
            "prediction": report.prediction,
            "risk_score": report.risk_score,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "failure_probability": report.failure_probability,
            "estimated_failure_days": report.estimated_failure_days,
            "explanation": report.explanation,
            "recommended_action": report.recommended_action,
            "impact_score": report.impact_score,
            "timestamp": datetime.now().isoformat(),
            "status": "Pending",
            "action_log": [{"action": "Created", "timestamp": datetime.now().isoformat()}],
            "breakdown": report.breakdown
        }
        reports.append(new_report)
        save_reports_to_disk(reports)
        return {"status": "success", "report": new_report}
    except Exception as e:
        return {"error": f"Failed to save report: {str(e)}"}


@app.get("/api/reports")
async def get_reports():
    return load_reports()


class UpdateStatus(BaseModel):
    status: str


@app.put("/api/report/update/{id}")
async def update_report(id: str, update: UpdateStatus):
    reports = load_reports()
    for report in reports:
        if report["id"] == id:
            report["status"] = update.status
            if "action_log" not in report:
                report["action_log"] = [{"action": "Created", "timestamp": report.get("timestamp", datetime.now().isoformat())}]
            report["action_log"].append({"action": f"Marked as {update.status}", "timestamp": datetime.now().isoformat()})
            save_reports_to_disk(reports)
            return {"status": "success", "report": report}
    return {"error": "Report not found"}


@app.put("/api/report/log_view/{id}")
async def log_view(id: str):
    reports = load_reports()
    for report in reports:
        if report["id"] == id:
            if "action_log" not in report:
                report["action_log"] = [{"action": "Created", "timestamp": report.get("timestamp", datetime.now().isoformat())}]
            if not report["action_log"] or report["action_log"][-1]["action"] != "Viewed":
                report["action_log"].append({"action": "Viewed", "timestamp": datetime.now().isoformat()})
                save_reports_to_disk(reports)
            return {"status": "success"}
    return {"error": "Report not found"}


@app.get("/api/notifications")
async def get_notifications():
    reports = load_reports()
    return [r for r in reports if r.get("risk_score", 0) >= SCORE_HIGH_THRESHOLD]


# ==============================================================================
# 2. UPGRADED V2 RESEARCH ENDPOINTS
# ==============================================================================

@app.post("/api/v2/analyze")
async def v2_analyze_infrastructure(
    file: UploadFile = File(...),
    rainfall: float = Form(25.0),
    traffic: float = Form(50.0),
    latitude: float = Form(11.0168),
    longitude: float = Form(76.9558),
    asset_type: str = Form("road"),
    asset_age: Optional[float] = Form(None),
    antecedent_rain: Optional[float] = Form(None)
):
    """
    V2 End-to-End Pipeline:
      Image -> Vision Inference -> Temperature Scaling -> Occlusion Heatmap ->
      Overpass Context Enrichment -> Hydrology SCS-CN -> Weibull PH Model ->
      Consequence -> Risk -> Monte Carlo Uncertainty -> Complete Breakdown.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file format"}

    # 1. Vision Inference with Temperature Calibration
    pred, damage_p, conf, model_source = predict_damage(image)

    # 2. Occlusion Sensitivity Heatmap (5x5 grid)
    try:
        heatmap_uri = generate_occlusion_heatmap(image, grid_size=5)
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        heatmap_uri = None

    # 3. Geospatial Context via Overpass
    context, context_source = get_spatial_context(latitude, longitude)

    # 4. Evaluate Mathematical Risk Engine
    report_id = str(uuid.uuid4())
    antecedent = antecedent_rain if antecedent_rain is not None else (rainfall * 0.35)

    risk_eval = evaluate_infrastructure_risk(
        report_id=report_id,
        asset_type=asset_type,
        damage_probability=damage_p,
        rainfall_forecast_24h_mm=rainfall,
        antecedent_rain_5d_mm=antecedent,
        traffic_index=traffic,
        asset_age_years=asset_age,
        distance_to_facility_m=context.get("distance_to_facility_m"),
        buildings_within_200m=context.get("buildings_within_200m", 15),
        road_class=context.get("road_class", "secondary"),
        context_source=context_source
    )

    return {
        "status": "success",
        "prediction": risk_eval["prediction"],
        "damage_probability": risk_eval["damage_probability"],
        "confidence": round(conf, 4),
        "model_source": model_source,
        "risk_score": risk_eval["risk_score"],
        "priority": risk_eval["priority"],
        "failure_probability_90d": risk_eval["failure_probability_90d"],
        "failure_probability_30d": risk_eval["failure_probability_30d"],
        "estimated_failure_days": risk_eval["estimated_failure_days"],
        "consequence_score": risk_eval["consequence_score"],
        "risk_index": risk_eval["risk_index"],
        "uncertainty_range": risk_eval["uncertainty_range"],
        "occlusion_heatmap": heatmap_uri,
        "spatial_context": context,
        "breakdown": risk_eval["breakdown"]
    }


class SimulationInput(BaseModel):
    asset_type: str = "road"
    age_years: float = 8.0
    rainfall_24h_mm: float = 40.0
    antecedent_rain_5d_mm: float = 15.0
    traffic_index: float = 50.0
    damage_probability: float = 0.70
    distance_to_facility_m: Optional[float] = 120.0
    buildings_within_200m: int = 45
    road_class: str = "secondary"
    custom_cn: Optional[float] = None


@app.post("/api/v2/simulate")
async def v2_simulate_risk(sim: SimulationInput):
    """
    Simulation Endpoint (No Image Required).
    Allows interactive what-if exploration across rainfall, aging, traffic, and damage levels.
    Essential for the live examiner demonstration.
    """
    report_id = f"sim_{sim.asset_type}_{int(sim.rainfall_24h_mm)}_{int(sim.age_years)}"
    risk_eval = evaluate_infrastructure_risk(
        report_id=report_id,
        asset_type=sim.asset_type,
        damage_probability=sim.damage_probability,
        rainfall_forecast_24h_mm=sim.rainfall_24h_mm,
        antecedent_rain_5d_mm=sim.antecedent_rain_5d_mm,
        traffic_index=sim.traffic_index,
        asset_age_years=sim.age_years,
        distance_to_facility_m=sim.distance_to_facility_m,
        buildings_within_200m=sim.buildings_within_200m,
        road_class=sim.road_class,
        custom_cn=sim.custom_cn,
        context_source="user_interactive_simulation"
    )
    return risk_eval


@app.get("/api/v2/model-info")
async def get_model_info():
    """Returns complete mathematical parameters, AHP matrices, and test metrics."""
    vision_meta = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                vision_meta = json.load(f)
        except Exception:
            pass

    opt_metrics_path = OPTIMIZED_METRICS_FILE
    vision_opt_meta = {}
    if os.path.exists(opt_metrics_path):
        try:
            with open(opt_metrics_path, "r", encoding="utf-8") as f:
                vision_opt_meta = json.load(f)
        except Exception:
            pass

    return {
        "platform": "CitySafe / ZylanZest V2",
        "institution": "KPR Institute of Engineering and Technology, Dept. of CSE",
        "active_vision_model": _model_source,
        "vision_metrics": vision_meta,
        "vision_metrics_optimized": vision_opt_meta,
        "ahp_hazard": {
            "weights": {
                "w_severity": round(float(HAZARD_WEIGHTS[0]), 4),
                "w_rainfall": round(float(HAZARD_WEIGHTS[1]), 4),
                "w_traffic": round(float(HAZARD_WEIGHTS[2]), 4)
            },
            "lambda_max": round(HAZARD_LAMBDA_MAX, 4),
            "consistency_index": round(HAZARD_CI, 4),
            "consistency_ratio": round(HAZARD_CR, 4),
            "is_consistent": bool(HAZARD_CR < 0.10)
        },
        "ahp_consequence": {
            "weights": {
                "w_facility": round(float(CONSEQUENCE_WEIGHTS[0]), 4),
                "w_density": round(float(CONSEQUENCE_WEIGHTS[1]), 4),
                "w_road_class": round(float(CONSEQUENCE_WEIGHTS[2]), 4),
                "w_criticality": round(float(CONSEQUENCE_WEIGHTS[3]), 4)
            },
            "lambda_max": round(CONSEQUENCE_LAMBDA_MAX, 4),
            "consistency_index": round(CONSEQUENCE_CI, 4),
            "consistency_ratio": round(CONSEQUENCE_CR, 4),
            "is_consistent": bool(CONSEQUENCE_CR < 0.10)
        },
        "asset_types": ASSET_TYPES,
        "parameters": {
            "weibull_shape_k": WEIBULL_SHAPE_K,
            "hazard_multiplier_B": round(HAZARD_MULTIPLIER_B, 4),
            "q_ref_mm": 100.0,
            "high_risk_threshold": SCORE_HIGH_THRESHOLD,
            "moderate_risk_threshold": SCORE_MODERATE_THRESHOLD
        }
    }


@app.get("/api/v2/hotspots")
async def get_hotspots():
    """Executes Getis-Ord Gi* spatial hotspot analysis across current reports."""
    reports = load_reports()
    return calculate_getis_ord_gi_star(reports, distance_threshold_m=500.0)


class PrioritizeRequest(BaseModel):
    budget_inr: float = 300000.0
    strategy: str = "ratio"  # 'ratio' (R / cost) or 'risk' (R only)


@app.post("/api/v2/prioritize")
async def post_prioritize(req: PrioritizeRequest):
    """Executes budget-constrained repair scheduling via greedy fractional knapsack relaxation."""
    reports = load_reports()
    return prioritize_repairs(reports, budget_inr=req.budget_inr, strategy=req.strategy)


@app.get("/api/v2/crowd-clusters")
async def get_crowd_clusters():
    """Groups reports within 50m of same asset type and applies Bayesian evidence fusion."""
    reports = load_reports()
    clusters = fuse_crowd_evidence(reports)
    return {"total_clusters": len(clusters), "clusters": clusters}


# Mount frontend static directory if needed
if os.path.exists("frontend"):
    app.mount("/portal", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
