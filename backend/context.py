"""
CitySafe / ZylanZest V2 — Geospatial Context Enrichment Module
==============================================================
Queries OpenStreetMap / Overpass API to extract environmental exposure features:
  1. Distance to nearest critical facility (School, Hospital, Clinic)
  2. Building density (count of building footprints within 200m radius)
  3. Road hierarchy class (motorway, primary, secondary, tertiary, residential)

Features:
  - Strict HTTP timeout <= 4 seconds
  - Local JSON disk cache with rounded coordinate keys (~100m grid)
  - Resilient fallback to documented neutral defaults without crashing
  - Transparent provenance tracking ('overpass_live', 'overpass_cache', 'default')
"""

import os
import json
import math
import requests
from typing import Dict, Any, Optional, Tuple

CACHE_DIR = "backend/database/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "overpass_cache.json")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT_SEC = 3.5

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to persist overpass cache: {e}")


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


def get_spatial_context(
    lat: float,
    lon: float
) -> Tuple[Dict[str, Any], str]:
    """
    Enriches coordinate with Overpass spatial features.
    Returns:
      context_dict: {
          "distance_to_facility_m": float or None,
          "buildings_within_200m": int,
          "road_class": str,
          "nearest_facility_type": str
      }
      source: 'overpass_live', 'overpass_cache', or 'default'
    """
    # Grid coordinate key to ~100m resolution (3 decimal places)
    cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
    cache = _load_cache()

    if cache_key in cache:
        return cache[cache_key], "overpass_cache"

    # Overpass QL Query: Search 1000m for facilities, 200m for buildings, 150m for roads
    query = f"""
    [out:json][timeout:4];
    (
      node["amenity"~"hospital|clinic|school|college"](around:1000,{lat},{lon});
      way["amenity"~"hospital|clinic|school|college"](around:1000,{lat},{lon});
      way["building"](around:200,{lat},{lon});
      way["highway"](around:150,{lat},{lon});
    );
    out center;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=REQUEST_TIMEOUT_SEC,
            headers={"User-Agent": "CitySafe-UrbanMonitoring/2.0 (Academic Research Prototype)"}
        )
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])

            nearest_facility_dist = None
            nearest_facility_type = "None detected within 1km"
            building_count = 0
            detected_roads = []

            for el in elements:
                tags = el.get("tags", {})
                
                # Check for critical facility
                if "amenity" in tags and tags["amenity"] in ["hospital", "clinic", "school", "college"]:
                    el_lat = el.get("lat") or el.get("center", {}).get("lat")
                    el_lon = el.get("lon") or el.get("center", {}).get("lon")
                    if el_lat and el_lon:
                        d = haversine_distance_meters(lat, lon, el_lat, el_lon)
                        if nearest_facility_dist is None or d < nearest_facility_dist:
                            nearest_facility_dist = d
                            nearest_facility_type = f"{tags['amenity'].capitalize()} ({tags.get('name', 'Facility')})"

                # Check for building
                if "building" in tags:
                    building_count += 1

                # Check for road
                if "highway" in tags:
                    detected_roads.append(tags["highway"].lower())

            # Determine dominant road hierarchy
            road_hierarchy = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "path"]
            road_class = "residential"
            for r in road_hierarchy:
                if r in detected_roads:
                    road_class = r
                    break

            context_result = {
                "distance_to_facility_m": round(nearest_facility_dist, 1) if nearest_facility_dist is not None else None,
                "buildings_within_200m": building_count,
                "road_class": road_class,
                "nearest_facility_type": nearest_facility_type
            }

            # Cache the successful live result
            cache[cache_key] = context_result
            _save_cache(cache)
            return context_result, "overpass_live"

    except Exception as e:
        # Graceful fallback without crashing
        pass

    # Neutral Documented Fallback
    fallback_context = {
        "distance_to_facility_m": None,
        "buildings_within_200m": 18,        # Typical average suburban cluster
        "road_class": "secondary",          # Standard municipal collector road
        "nearest_facility_type": "Neutral prior (Overpass offline/fallback)"
    }
    return fallback_context, "default"

