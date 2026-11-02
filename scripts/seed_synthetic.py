"""
CitySafe / ZylanZest V2 — Synthetic Infrastructure Report Seeder
================================================================
Generates 60 synthetic demonstration reports clustered across Coimbatore, Tamil Nadu.
Coordinates centered at Lat: 11.0168, Lon: 76.9558.

Clusters designed for spatial hotspot validation:
  - Cluster 1: Gandhipuram / Cross-Cut Road (Dense commercial, high runoff, aging drains & culverts)
  - Cluster 2: Peelamedu / Avinashi Road (Major arterial corridor, high traffic stress)
  - Cluster 3: Singanallur / Trichy Road (Industrial, flood-prone retaining structures)
  - Cluster 4: RS Puram / Thadagam Road (Residential, moderate risk)
  - Cluster 5: Vadavalli Foothills (Periphery, low risk footpaths)

Usage:
  python scripts/seed_synthetic.py          # Appends/Seeds synthetic demonstration data
  python scripts/seed_synthetic.py --reset  # Resets database to original state
"""

import os
import sys
import json
import uuid
import random
import math
from datetime import datetime, timedelta

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import ASSET_TYPES
from backend.risk_engine import evaluate_infrastructure_risk

REPORTS_FILE = "backend/database/reports.json"
BACKUP_FILE = "backend/database/reports_original_backup.json"
os.makedirs("backend/database", exist_ok=True)


def backup_original():
    if os.path.exists(REPORTS_FILE) and not os.path.exists(BACKUP_FILE):
        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Backed up original reports ({len(data)} items) to {BACKUP_FILE}")


def reset_database():
    if os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(REPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Database successfully restored to original backup ({len(data)} items).")
    else:
        print("No backup file found to restore from.")


def generate_synthetic_reports(count=60):
    backup_original()

    clusters = [
        {
            "name": "Gandhipuram Commercial Hub",
            "lat": 11.0176, "lon": 76.9673,
            "severity_bias": 0.94, "rain_bias": 105.0, "traffic_bias": 92.0,
            "asset_types": ["drain", "culvert", "road"],
            "density": 150, "road_class": "primary",
            "facility_dist_range": (10, 45),
            "age_fraction": (0.75, 0.85),
            "cluster_size": 16
        },
        {
            "name": "Peelamedu Arterial Corridor",
            "lat": 11.0285, "lon": 77.0018,
            "severity_bias": 0.78, "rain_bias": 65.0, "traffic_bias": 95.0,
            "asset_types": ["road", "culvert"],
            "density": 110, "road_class": "trunk",
            "facility_dist_range": (40, 150),
            "age_fraction": (0.60, 0.75),
            "cluster_size": 14
        },
        {
            "name": "Singanallur Industrial Drain Corridor",
            "lat": 10.9982, "lon": 77.0195,
            "severity_bias": 0.70, "rain_bias": 80.0, "traffic_bias": 70.0,
            "asset_types": ["drain", "retaining_wall"],
            "density": 70, "road_class": "secondary",
            "facility_dist_range": (80, 250),
            "age_fraction": (0.50, 0.65),
            "cluster_size": 12
        },
        {
            "name": "RS Puram Residential Sector",
            "lat": 11.0089, "lon": 76.9458,
            "severity_bias": 0.40, "rain_bias": 30.0, "traffic_bias": 45.0,
            "asset_types": ["footpath", "drain"],
            "density": 85, "road_class": "tertiary",
            "facility_dist_range": (150, 400),
            "age_fraction": (0.35, 0.50),
            "cluster_size": 10
        },
        {
            "name": "Vadavalli Suburban Fringe",
            "lat": 11.0255, "lon": 76.9085,
            "severity_bias": 0.20, "rain_bias": 15.0, "traffic_bias": 25.0,
            "asset_types": ["footpath", "road"],
            "density": 35, "road_class": "residential",
            "facility_dist_range": (300, 800),
            "age_fraction": (0.20, 0.35),
            "cluster_size": 8
        }
    ]

    # Deterministic generation seed
    rng = random.Random(42)
    reports = []
    base_time = datetime.now() - timedelta(days=14)

    for cluster in clusters:
        for i in range(cluster["cluster_size"]):
            report_id = str(uuid.uuid4())

            # Coordinates jittered within ~350m radius (1 deg lat ~ 111 km => 0.003 deg ~ 330m)
            jitter_lat = rng.gauss(0, 0.0018)
            jitter_lon = rng.gauss(0, 0.0018)
            lat = cluster["lat"] + jitter_lat
            lon = cluster["lon"] + jitter_lon

            asset_type = rng.choice(cluster["asset_types"])
            eta = ASSET_TYPES[asset_type]["eta_years"]
            age_frac = rng.uniform(cluster["age_fraction"][0], cluster["age_fraction"][1])
            age = max(1.0, round(eta * age_frac, 1))

            damage_p = min(max(rng.gauss(cluster["severity_bias"], 0.08), 0.05), 0.99)
            rainfall = max(0.0, rng.gauss(cluster["rain_bias"], 15.0))
            antecedent = max(0.0, rng.gauss(cluster["rain_bias"] * 0.4, 8.0))
            traffic = min(max(rng.gauss(cluster["traffic_bias"], 10.0), 10.0), 100.0)
            facility_d = rng.uniform(cluster["facility_dist_range"][0], cluster["facility_dist_range"][1])

            # Evaluate through pure risk engine
            risk_eval = evaluate_infrastructure_risk(
                report_id=report_id,
                asset_type=asset_type,
                damage_probability=damage_p,
                rainfall_forecast_24h_mm=rainfall,
                antecedent_rain_5d_mm=antecedent,
                traffic_index=traffic,
                asset_age_years=age,
                distance_to_facility_m=facility_d,
                buildings_within_200m=cluster["density"] + rng.randint(-10, 10),
                road_class=cluster["road_class"],
                context_source="synthetic_cluster_prior"
            )

            timestamp = (base_time + timedelta(hours=rng.randint(1, 336))).isoformat()
            status = "Pending" if risk_eval["priority"] == "High" else rng.choice(["Pending", "Resolved"])

            report = {
                "id": report_id,
                "image_name": f"synthetic_{asset_type}_{i+1:03d}.jpg",
                "asset_type": asset_type,
                "prediction": risk_eval["prediction"],
                "risk_score": risk_eval["risk_score"],
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "failure_probability": risk_eval["failure_probability"],
                "estimated_failure_days": risk_eval["estimated_failure_days"],
                "explanation": f"Synthetic simulation node in {cluster['name']} with {risk_eval['priority']} vulnerability.",
                "recommended_action": risk_eval["recommended_action"],
                "impact_score": f"{risk_eval['priority']} (Consequence: {risk_eval['consequence_score']:.2f})",
                "timestamp": timestamp,
                "status": status,
                "is_synthetic": True,
                "cluster_zone": cluster["name"],
                "breakdown": risk_eval["breakdown"]
            }
            reports.append(report)

    # Persist to database
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)

    high_n = sum(1 for r in reports if r["risk_score"] >= 60)
    mod_n = sum(1 for r in reports if 30 <= r["risk_score"] < 60)
    low_n = sum(1 for r in reports if r["risk_score"] < 30)

    print("=" * 65)
    print("SYNTHETIC INFRASTRUCTURE DATA SEEDED SUCCESSFULLY")
    print("=" * 65)
    print(f"Total reports generated : {len(reports)}")
    print(f"High risk (>= 60)       : {high_n}")
    print(f"Moderate risk (30 - 59) : {mod_n}")
    print(f"Low risk (< 30)         : {low_n}")
    print(f"Target location         : Coimbatore (Lat: 11.0168, Lon: 76.9558)")
    print("NOTE: These data are strictly synthetic demonstration records.")
    print("=" * 65)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_database()
    else:
        generate_synthetic_reports(60)
