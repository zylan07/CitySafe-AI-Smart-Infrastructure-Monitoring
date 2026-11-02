"""
CitySafe / ZylanZest V2 — Core Configuration & Modeling Assumptions
===================================================================
Single source of truth for all mathematical modeling parameters, literature priors,
AHP pairwise comparison matrices, environmental thresholds, and asset attributes.

Every parameter is annotated with:
  - Meaning & Mathematical Usage
  - Classification:
      * MATHEMATICAL IDENTITY: Analytical exact definition
      * EMPIRICAL LITERATURE: Established published civil/hydrology findings
      * MODELLING CHOICE: Conscious architecture selection with physical rationale
      * EXPERT PRIOR: Engineering baseline prior before empirical field fitting
      * FUTURE LEARNED: To be estimated from municipal maintenance records
"""

import math
from typing import Dict, Any

# ==============================================================================
# 1. ASSET CLASS PRIORS & ATTRIBUTES
# ==============================================================================
# CLASSIFICATION: EXPERT PRIOR / FUTURE LEARNED
# Meaning: Initial baseline Weibull scale (eta, years) and asset-specific attributes.
# Rationale: In absence of complete Coimbatore municipal asset failure records,
# baseline design lifespans are derived from standard civil engineering infrastructure
# asset management manuals (e.g. ASCE / IRC guidelines for urban drainage & pavement).
# Future Work: Fit eta and k using survival analysis (Weibull MLE) on municipal work orders.

ASSET_TYPES: Dict[str, Dict[str, Any]] = {
    "culvert": {
        "name": "Box/Pipe Culvert",
        "eta_years": 45.0,        # Prior characteristic life (years)
        "criticality": 0.85,      # Consequence weight factor [0, 1]
        "default_cn": 75,         # SCS Curve Number for surrounding soil/impervious mix
        "repair_cost_inr": 85000, # Estimated average local repair cost (INR)
        "description": "Cross-drainage structure under roadways susceptible to blockage and scour"
    },
    "drain": {
        "name": "Stormwater Drain",
        "eta_years": 30.0,
        "criticality": 0.70,
        "default_cn": 80,
        "repair_cost_inr": 45000,
        "description": "Open or covered urban stormwater conveyance channels"
    },
    "retaining_wall": {
        "name": "Retaining Wall",
        "eta_years": 40.0,
        "criticality": 0.90,
        "default_cn": 70,
        "repair_cost_inr": 120000,
        "description": "Earth retaining structures along road cuts or drainage channels"
    },
    "footpath": {
        "name": "Pedestrian Footpath / Walkway",
        "eta_years": 20.0,
        "criticality": 0.40,
        "default_cn": 85,
        "repair_cost_inr": 15000,
        "description": "Pedestrian paved corridors and sidewalk slabs"
    },
    "road": {
        "name": "Urban Asphalt Road / Pavement",
        "eta_years": 15.0,
        "criticality": 0.80,
        "default_cn": 90,
        "repair_cost_inr": 95000,
        "description": "Bituminous wearing course and sub-base infrastructure"
    }
}

# CLASSIFICATION: EXPERT PRIOR
# Meaning: Weibull shape parameter k for structural wear-out aging.
# Range in literature: 1.5 - 3.0 for mechanical & structural wear-out degradation.
# k > 1 implies wear-out degradation (increasing hazard rate over time).
# k = 1 would reduce to memoryless exponential failure (inapplicable to aging concrete).
WEIBULL_SHAPE_K: float = 2.0
WEIBULL_K_LOWER_BOUND: float = 1.1
WEIBULL_K_UPPER_BOUND: float = 3.5

# Default evaluation horizons
HORIZON_30D_YEARS: float = 30.0 / 365.0
HORIZON_90D_YEARS: float = 90.0 / 365.0
DEFAULT_ASSET_AGE_YEARS: float = 8.0  # Assumed default age if unrecorded

# Horizon calculation target probability (10% conditional failure risk)
REMAINING_LIFE_P_TARGET: float = 0.10
MAX_DISPLAY_DAYS: int = 3650  # Cap UI display at 10 years

# ==============================================================================
# 2. HYDROLOGICAL MODEL (SCS-CN) ASSUMPTIONS
# ==============================================================================
# CLASSIFICATION: EMPIRICAL LITERATURE (NRCS / USDA National Engineering Handbook, Section 4)
# S = 25400 / CN - 254 (in mm)
# Ia = 0.2 * S (Initial abstraction)
INITIAL_ABSTRACTION_RATIO: float = 0.20

# CLASSIFICATION: MODELLING CHOICE
# Reference runoff scale Q_ref for normalising rainfall hazard covariate:
# x_rain = 1 - exp(-Q / Q_ref)
# Q_ref = 100 mm corresponds to a severe urban rainfall runoff event (~120-150mm precipitation)
Q_REF_MM: float = 100.0

# Antecedent Moisture Condition (AMC) 5-Day Rainfall Limits (NRCS standard):
# Dormant vs Growing season average for tropical/subtropical urban environments:
AMC_I_UPPER_LIMIT_MM: float = 15.0   # Total 5-day antecedent rain < 15mm => AMC I (Dry)
AMC_II_UPPER_LIMIT_MM: float = 40.0  # 15mm <= rain <= 40mm => AMC II (Normal)
                                     # rain > 40mm => AMC III (Wet/Saturated)

# ==============================================================================
# 3. AHP HAZARD COVARIATES (Severity, Rainfall, Traffic)
# ==============================================================================
# CLASSIFICATION: EXPERT PRIOR / SAATY AHP METHODOLOGY
# Pairwise comparison matrix for hazard factors [x_sev, x_rain, x_traf]:
# Severity (structural damage) is strongly prioritized over rainfall and traffic.
# Rainfall is moderately prioritized over traffic.
#
# Matrix A:
#           Severity  Rainfall  Traffic
# Severity    1.0       2.0       3.0
# Rainfall    0.5       1.0       2.0
# Traffic     0.3333    0.5       1.0
AHP_HAZARD_MATRIX = [
    [1.0,     2.0, 3.0],
    [0.5,     1.0, 2.0],
    [1.0/3.0, 0.5, 1.0]
]

# Random Index for n=3 (Saaty, 1980)
RI_N3: float = 0.58

# Maximum hazard multiplier scale B = ln(10) ≈ 2.302585
# Such that max combined covariate effect exp(beta · [1, 1, 1]) = exp(B) = 10.0
HAZARD_MULTIPLIER_B: float = math.log(10.0)

# ==============================================================================
# 4. AHP CONSEQUENCE MODEL
# ==============================================================================
# CLASSIFICATION: EXPERT PRIOR / SAATY AHP METHODOLOGY
# Pairwise comparison matrix for consequence factors:
# 1. Facility Proximity (Schools, Hospitals)
# 2. Building Density (High concentration of population)
# 3. Road Hierarchy / Class
# 4. Asset Criticality (Intrinsic structural dependence)
#
# Matrix C:
#              Facility  Density  RoadClass  Criticality
# Facility       1.0       2.0      3.0         3.0
# Density        0.5       1.0      2.0         2.0
# RoadClass      0.3333    0.5      1.0         1.0
# Criticality    0.3333    0.5      1.0         1.0
AHP_CONSEQUENCE_MATRIX = [
    [1.0,     2.0, 3.0, 3.0],
    [0.5,     1.0, 2.0, 2.0],
    [1.0/3.0, 0.5, 1.0, 1.0],
    [1.0/3.0, 0.5, 1.0, 1.0]
]

# Random Index for n=4 (Saaty, 1980)
RI_N4: float = 0.90

# Environmental Consequence Decay Constants
FACILITY_DECAY_DISTANCE_M: float = 150.0  # f_facility = exp(-d / 150)
FACILITY_MAX_SEARCH_RADIUS_M: float = 1000.0
DENSITY_REFERENCE_BUILDINGS: float = 150.0 # f_density = min(1.0, count_200m / 150)

ROAD_CLASS_WEIGHTS: Dict[str, float] = {
    "motorway": 1.0,
    "trunk": 1.0,
    "primary": 1.0,
    "secondary": 0.8,
    "tertiary": 0.6,
    "residential": 0.4,
    "service": 0.2,
    "path": 0.2,
    "default": 0.4
}

# ==============================================================================
# 5. RISK TIERS & REFERENCE CALIBRATION
# ==============================================================================
# CLASSIFICATION: MODELLING CHOICE
# Worst realistic scenario used to calibrate reference risk index R_ref:
# x_sev = 1.0, x_rain = 1.0, x_traf = 1.0, a = 0.8 * eta, C = 1.0
WORST_CASE_AGE_FRACTION: float = 0.80

# Configurable Risk Display Tiers
SCORE_HIGH_THRESHOLD: float = 60.0
SCORE_MODERATE_THRESHOLD: float = 30.0

# ==============================================================================
# 6. UNCERTAINTY QUANTIFICATION (MONTE CARLO)
# ==============================================================================
# CLASSIFICATION: MODELLING CHOICE
# Deterministic Monte Carlo seed derived from report ID: seed = hash(report_id)
# Number of samples:
MONTE_CARLO_SAMPLES: int = 500
ETA_PERTURBATION_STD: float = 0.25      # LogNormal(0, 0.25)
K_PERTURBATION_STD: float = 0.20        # Normal(k, 0.20), clamped >= 1.1
BETA_PERTURBATION_STD: float = 0.20     # LogNormal(0, 0.20) for hazard scaling

# ==============================================================================
# 7. MULTI-REPORT BAYESIAN FUSION & SPATIAL CLUSTERING
# ==============================================================================
# CLASSIFICATION: MODELLING CHOICE & STATISTICAL THEORY
FUSION_DISTANCE_THRESHOLD_M: float = 50.0 # Same physical asset radius
BAYESIAN_PRIOR_DAMAGE: float = 0.20       # Base rate prior probability of damaged asset

# Measured validation performance of vision model (updated dynamically upon calibration)
# Baseline literature defaults:
DEFAULT_VISION_SENSITIVITY: float = 0.88
DEFAULT_VISION_SPECIFICITY: float = 0.86

# ==============================================================================
# 8. SPATIAL HOTSPOT ANALYSIS (GETIS-ORD GI*)
# ==============================================================================
# CLASSIFICATION: STATISTICAL INFERENCE (Ord & Getis, 1995)
HOTSPOT_DISTANCE_THRESHOLD_M: float = 500.0
MIN_OBSERVATIONS_FOR_HOTSPOT: int = 10
Z_SCORE_95_CONFIDENCE: float = 1.96
Z_SCORE_99_CONFIDENCE: float = 2.58

