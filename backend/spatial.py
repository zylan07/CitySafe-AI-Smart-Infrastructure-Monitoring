"""
CitySafe / ZylanZest V2 — Spatial Statistics, Bayesian Fusion & Optimization
============================================================================
Implements:
  1. Multi-Report Bayesian Evidence Fusion (50m spatial cluster)
  2. Spatial Point Pattern Clustering & Getis-Ord Gi* Hotspot Detection
  3. Budget-Constrained Repair Prioritization (Greedy R / Cost Fractional Knapsack)
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from backend.config import (
    ASSET_TYPES,
    FUSION_DISTANCE_THRESHOLD_M,
    BAYESIAN_PRIOR_DAMAGE,
    DEFAULT_VISION_SENSITIVITY,
    DEFAULT_VISION_SPECIFICITY,
    HOTSPOT_DISTANCE_THRESHOLD_M,
    MIN_OBSERVATIONS_FOR_HOTSPOT,
    Z_SCORE_95_CONFIDENCE,
    Z_SCORE_99_CONFIDENCE
)
from backend.context import haversine_distance_meters


# ==============================================================================
# 1. MULTI-REPORT SPATIAL AGGREGATION & BAYESIAN EVIDENCE FUSION
# ==============================================================================

def fuse_crowd_evidence(
    reports: List[Dict[str, Any]],
    sensitivity: float = DEFAULT_VISION_SENSITIVITY,
    specificity: float = DEFAULT_VISION_SPECIFICITY,
    distance_threshold_m: float = FUSION_DISTANCE_THRESHOLD_M,
    prior_pi: float = BAYESIAN_PRIOR_DAMAGE
) -> List[Dict[str, Any]]:
    """
    Groups reports within distance_threshold_m of the same asset_type and applies
    Bayesian log-odds updating using the classifier's measured sensitivity and specificity.
    Returns clustered asset entities with fused damage confidence.
    """
    if not reports:
        return []

    # Safeguard likelihood ratio bounds
    se = min(max(sensitivity, 0.55), 0.99)
    sp = min(max(specificity, 0.55), 0.99)
    lr_damaged = se / (1.0 - sp)
    lr_safe = (1.0 - se) / sp

    log_lr_damaged = math.log(lr_damaged)
    log_lr_safe = math.log(lr_safe)
    prior_logit = math.log(prior_pi / (1.0 - prior_pi))

    # Cluster reports greedily by distance and asset type
    clusters: List[Dict[str, Any]] = []

    for r in reports:
        lat = r.get("latitude", 0.0)
        lon = r.get("longitude", 0.0)
        asset_type = r.get("breakdown", {}).get("asset_type") or r.get("asset_type", "road")

        matched_cluster = None
        for c in clusters:
            if c["asset_type"] == asset_type:
                dist = haversine_distance_meters(lat, lon, c["center_lat"], c["center_lon"])
                if dist <= distance_threshold_m:
                    matched_cluster = c
                    break

        if matched_cluster is not None:
            matched_cluster["reports"].append(r)
            # Update running centroid
            n = len(matched_cluster["reports"])
            matched_cluster["center_lat"] = ((n - 1) * matched_cluster["center_lat"] + lat) / n
            matched_cluster["center_lon"] = ((n - 1) * matched_cluster["center_lon"] + lon) / n
        else:
            clusters.append({
                "asset_id": f"cluster_{asset_type}_{len(clusters) + 1}",
                "asset_type": asset_type,
                "center_lat": lat,
                "center_lon": lon,
                "reports": [r]
            })

    # Apply Bayesian logit update per cluster
    fused_entities = []
    for c in clusters:
        obs_log_lr_sum = 0.0
        rep_ids = []
        highest_risk = 0
        latest_report = None

        for rep in c["reports"]:
            rep_ids.append(rep.get("id"))
            pred = rep.get("prediction", "Damaged")
            risk = rep.get("risk_score", 0)
            if risk > highest_risk:
                highest_risk = risk
                latest_report = rep

            if pred == "Damaged":
                obs_log_lr_sum += log_lr_damaged
            else:
                obs_log_lr_sum += log_lr_safe

        # Fused posterior probability
        fused_logit = prior_logit + obs_log_lr_sum
        fused_logit_clamped = min(max(fused_logit, -15.0), 15.0)
        fused_prob = 1.0 / (1.0 + math.exp(-fused_logit_clamped))

        fused_entities.append({
            "asset_id": c["asset_id"],
            "asset_type": c["asset_type"],
            "latitude": round(c["center_lat"], 5),
            "longitude": round(c["center_lon"], 5),
            "report_count": len(c["reports"]),
            "report_ids": rep_ids,
            "fused_damage_confidence": round(fused_prob, 4),
            "fused_damage_percent": int(round(fused_prob * 100)),
            "peak_risk_score": highest_risk,
            "representative_report": latest_report
        })

    return fused_entities


# ==============================================================================
# 2. GETIS-ORD GI* SPATIAL HOTSPOT ANALYSIS
# ==============================================================================

def calculate_getis_ord_gi_star(
    reports: List[Dict[str, Any]],
    distance_threshold_m: float = HOTSPOT_DISTANCE_THRESHOLD_M
) -> Dict[str, Any]:
    """
    Computes the Getis-Ord Gi* statistic for spatial hotspot detection:
      Gi* = (sum_j w_ij x_j - X_bar sum_j w_ij) / (S * sqrt((n sum_j w_ij^2 - (sum_j w_ij)^2)/(n - 1)))
    Requires at least MIN_OBSERVATIONS_FOR_HOTSPOT (10) reports to conduct valid inference.
    """
    n = len(reports)
    if n < MIN_OBSERVATIONS_FOR_HOTSPOT:
        return {
            "status": "insufficient_data",
            "message": f"Insufficient observations for hotspot analysis (found {n}, minimum {MIN_OBSERVATIONS_FOR_HOTSPOT} required).",
            "hotspots": []
        }

    # Extract coordinates and risk scores
    coords = []
    x = []
    for r in reports:
        lat = float(r.get("latitude", 0.0))
        lon = float(r.get("longitude", 0.0))
        score = float(r.get("risk_score", 0.0))
        coords.append((lat, lon))
        x.append(score)

    x = np.array(x, dtype=np.float64)
    x_bar = float(np.mean(x))
    sum_x2 = float(np.sum(x ** 2))
    variance = (sum_x2 / n) - (x_bar ** 2)
    s = math.sqrt(max(1e-9, variance))

    # Binary spatial weight matrix: w_ij = 1 if dist <= threshold, else 0 (including i=j for Gi*)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = haversine_distance_meters(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d <= distance_threshold_m:
                W[i, j] = 1.0

    z_scores = []
    results = []

    for i in range(n):
        w_i = W[i, :]
        sum_w = float(np.sum(w_i))
        sum_w2 = float(np.sum(w_i ** 2))

        numerator = float(np.sum(w_i * x)) - (x_bar * sum_w)
        denom_inner = (n * sum_w2 - (sum_w ** 2)) / (n - 1)
        denom = s * math.sqrt(max(0.0, denom_inner))

        z_val = numerator / denom if denom > 1e-9 else 0.0
        z_scores.append(z_val)

        # Determine statistical significance
        if z_val >= Z_SCORE_99_CONFIDENCE:
            sig = "Hotspot (99% Confidence)"
        elif z_val >= Z_SCORE_95_CONFIDENCE:
            sig = "Hotspot (95% Confidence)"
        elif z_val <= -Z_SCORE_99_CONFIDENCE:
            sig = "Coldspot (99% Confidence)"
        elif z_val <= -Z_SCORE_95_CONFIDENCE:
            sig = "Coldspot (95% Confidence)"
        else:
            sig = "Not Significant"

        results.append({
            "report_id": reports[i].get("id"),
            "latitude": coords[i][0],
            "longitude": coords[i][1],
            "risk_score": x[i],
            "gi_star_z": round(z_val, 3),
            "significance": sig,
            "is_hotspot": bool(z_val >= Z_SCORE_95_CONFIDENCE)
        })

    return {
        "status": "success",
        "sample_size": n,
        "mean_risk_score": round(x_bar, 2),
        "std_dev_risk_score": round(s, 2),
        "hotspot_count": sum(1 for r in results if r["is_hotspot"]),
        "hotspots": results
    }


# ==============================================================================
# 3. REPAIR PRIORITIZATION & KNAPSACK RELAXATION
# ==============================================================================

def prioritize_repairs(
    reports: List[Dict[str, Any]],
    budget_inr: float = 250000.0,
    strategy: str = "ratio"  # 'ratio' (R / cost) or 'risk' (R only)
) -> Dict[str, Any]:
    """
    Applies greedy knapsack relaxation scheduling to prioritize infrastructure repair
    interventions under a fixed municipal capital budget constraint.
    """
    candidates = []

    for r in reports:
        # Extract risk index R and asset type
        asset_type = r.get("breakdown", {}).get("asset_type") or r.get("asset_type", "road")
        norm_asset = asset_type.lower() if asset_type in ASSET_TYPES else "road"
        cost = float(ASSET_TYPES[norm_asset]["repair_cost_inr"])

        # Risk metric
        r_val = float(r.get("breakdown", {}).get("risk_index_R", r.get("risk_score", 0) / 100.0))
        ratio = r_val / cost if cost > 0 else 0.0

        candidates.append({
            "id": r.get("id"),
            "image_name": r.get("image_name", "report.jpg"),
            "asset_type": norm_asset,
            "asset_name": ASSET_TYPES[norm_asset]["name"],
            "latitude": r.get("latitude", 0.0),
            "longitude": r.get("longitude", 0.0),
            "risk_score": r.get("risk_score", 0),
            "risk_index": round(r_val, 5),
            "cost_inr": cost,
            "efficiency_ratio": round(ratio * 1e6, 4)  # scaled for readability
        })

    # Sort based on selected strategy
    if strategy == "risk":
        candidates.sort(key=lambda x: x["risk_index"], reverse=True)
    else:  # 'ratio' (greedy knapsack relaxation)
        candidates.sort(key=lambda x: x["efficiency_ratio"], reverse=True)

    # Greedily allocate budget
    allocated = []
    remaining_budget = float(budget_inr)
    total_cost = 0.0
    total_risk_removed = 0.0

    for item in candidates:
        if item["cost_inr"] <= remaining_budget:
            allocated.append(item)
            remaining_budget -= item["cost_inr"]
            total_cost += item["cost_inr"]
            total_risk_removed += item["risk_index"]

    return {
        "budget_total_inr": budget_inr,
        "total_cost_inr": total_cost,
        "remaining_budget_inr": remaining_budget,
        "total_risk_removed": round(total_risk_removed, 5),
        "selected_count": len(allocated),
        "total_candidate_count": len(candidates),
        "strategy": strategy,
        "scheduled_repairs": allocated
    }

