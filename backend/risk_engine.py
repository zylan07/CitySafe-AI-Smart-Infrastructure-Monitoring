"""
CitySafe / ZylanZest V2 — Mathematical Risk Engine
==================================================
Deterministic, explainable, and uncertainty-aware infrastructure risk calculation.
Implements:
  1. SCS Curve Number (SCS-CN) Runoff & Rainfall Hazard Normalization
  2. Saaty Analytic Hierarchy Process (AHP) for Covariate & Consequence Weighting
  3. Weibull Proportional Hazards Model with Age-Dependent Conditional Failure Probability
  4. Horizon to 10% Conditional Failure Probability (Remaining Life Indicator)
  5. Consequence Modeling with Spatial & Structural Factors
  6. Expected Risk Quantification R = P_fail × C
  7. Deterministic Seeded Monte Carlo Uncertainty Estimation (p5, p50, p95)
"""

import math
import hashlib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from backend.config import (
    ASSET_TYPES,
    WEIBULL_SHAPE_K,
    WEIBULL_K_LOWER_BOUND,
    WEIBULL_K_UPPER_BOUND,
    HORIZON_30D_YEARS,
    HORIZON_90D_YEARS,
    DEFAULT_ASSET_AGE_YEARS,
    REMAINING_LIFE_P_TARGET,
    MAX_DISPLAY_DAYS,
    INITIAL_ABSTRACTION_RATIO,
    Q_REF_MM,
    AMC_I_UPPER_LIMIT_MM,
    AMC_II_UPPER_LIMIT_MM,
    AHP_HAZARD_MATRIX,
    RI_N3,
    HAZARD_MULTIPLIER_B,
    AHP_CONSEQUENCE_MATRIX,
    RI_N4,
    FACILITY_DECAY_DISTANCE_M,
    DENSITY_REFERENCE_BUILDINGS,
    ROAD_CLASS_WEIGHTS,
    WORST_CASE_AGE_FRACTION,
    SCORE_HIGH_THRESHOLD,
    SCORE_MODERATE_THRESHOLD,
    MONTE_CARLO_SAMPLES,
    ETA_PERTURBATION_STD,
    K_PERTURBATION_STD,
    BETA_PERTURBATION_STD,
)


# ==============================================================================
# 1. AHP MATRIX SOLVER & CONSISTENCY CHECK
# ==============================================================================

def solve_ahp(matrix: List[List[float]], ri: float) -> Tuple[np.ndarray, float, float, float]:
    """
    Solves an AHP pairwise comparison matrix using the principal eigenvector method.
    Returns:
        weights: Normalized priority vector sum(w) = 1
        lambda_max: Principal eigenvalue
        ci: Consistency Index = (lambda_max - n) / (n - 1)
        cr: Consistency Ratio = CI / RI
    """
    A = np.array(matrix, dtype=np.float64)
    n = A.shape[0]

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_idx = np.argmax(np.real(eigenvalues))
    lambda_max = float(np.real(eigenvalues[max_idx]))
    principal_vector = np.real(eigenvectors[:, max_idx])

    # Ensure all components are positive
    if np.all(principal_vector < 0):
        principal_vector = -principal_vector
    principal_vector = np.maximum(principal_vector, 1e-9)

    # Normalize weights
    weights = principal_vector / np.sum(principal_vector)

    # Consistency Index & Ratio
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    cr = ci / ri if ri > 0 else 0.0

    return weights, lambda_max, ci, cr


# Pre-compute and validate AHP weights at module load time
HAZARD_WEIGHTS, HAZARD_LAMBDA_MAX, HAZARD_CI, HAZARD_CR = solve_ahp(AHP_HAZARD_MATRIX, RI_N3)
CONSEQUENCE_WEIGHTS, CONSEQUENCE_LAMBDA_MAX, CONSEQUENCE_CI, CONSEQUENCE_CR = solve_ahp(AHP_CONSEQUENCE_MATRIX, RI_N4)

# Beta vector for hazard covariates: beta_j = B * w_j where B = ln(10)
BETA_VECTOR = HAZARD_MULTIPLIER_B * HAZARD_WEIGHTS


# ==============================================================================
# 2. HYDROLOGICAL MODEL: SCS CURVE NUMBER (SCS-CN)
# ==============================================================================

def determine_amc(antecedent_rain_5d_mm: float) -> str:
    """
    Determines Antecedent Moisture Condition (AMC) class from 5-day antecedent rainfall.
    AMC I: Dry, AMC II: Average, AMC III: Saturated/Wet.
    """
    if antecedent_rain_5d_mm < AMC_I_UPPER_LIMIT_MM:
        return "I"
    elif antecedent_rain_5d_mm <= AMC_II_UPPER_LIMIT_MM:
        return "II"
    else:
        return "III"


def adjust_cn_for_amc(cn_ii: float, amc: str) -> float:
    """
    Adjusts standard AMC II Curve Number to AMC I or AMC III conditions.
    Formulas (NRCS):
      CN_I = 4.2 * CN / (10 - 0.058 * CN)
      CN_III = 23 * CN / (10 + 0.13 * CN)
    """
    cn_clamped = min(max(float(cn_ii), 30.0), 98.0)
    if amc == "I":
        return (4.2 * cn_clamped) / (10.0 - 0.058 * cn_clamped)
    elif amc == "III":
        return (23.0 * cn_clamped) / (10.0 + 0.13 * cn_clamped)
    return cn_clamped


def calculate_scs_runoff(rainfall_p_mm: float, cn: float) -> Tuple[float, float, float]:
    """
    Calculates estimated storm runoff Q using the SCS Curve Number method.
    Returns:
        Q: Runoff in mm
        S: Potential maximum retention in mm (S = 25400 / CN - 254)
        Ia: Initial abstraction in mm (Ia = 0.2 * S)
    """
    p = max(0.0, float(rainfall_p_mm))
    cn_clamped = min(max(float(cn), 25.0), 99.5)

    # Maximum potential retention
    S = (25400.0 / cn_clamped) - 254.0
    Ia = INITIAL_ABSTRACTION_RATIO * S

    if p <= Ia:
        Q = 0.0
    else:
        u = p - Ia
        Q = (u ** 2) / (u + S)

    return Q, S, Ia


def normalize_rainfall_hazard(runoff_q_mm: float, q_ref: float = Q_REF_MM) -> float:
    """
    Transforms surface runoff Q (mm) into a dimensionless bounded hazard covariate:
      x_rain = 1 - exp(-Q / Q_ref)
    Properties:
      - x_rain in [0, 1)
      - x_rain monotonically increasing with Q
    """
    q = max(0.0, float(runoff_q_mm))
    raw_val = 1.0 - math.exp(-q / q_ref)
    # Guarantee mathematical half-open interval [0, 1) against IEEE 754 float saturation
    return float(min(raw_val, 1.0 - 1e-12))


# ==============================================================================
# 3. WEIBULL PROPORTIONAL HAZARDS CORE
# ==============================================================================

def weibull_cumulative_hazard(t: float, eta: float, k: float, beta_dot_x: float) -> float:
    """
    Calculates the Weibull PH cumulative hazard function:
      H(t|x) = (t / eta)^k * exp(beta · x)
    """
    if t <= 0.0 or eta <= 0.0:
        return 0.0
    return float(((t / eta) ** k) * math.exp(beta_dot_x))


def weibull_survival(t: float, eta: float, k: float, beta_dot_x: float) -> float:
    """
    Calculates the Weibull PH survival function:
      S(t|x) = exp(-H(t|x))
    """
    H = weibull_cumulative_hazard(t, eta, k, beta_dot_x)
    return float(math.exp(-H))


def conditional_failure_probability(
    asset_age_years: float,
    delta_years: float,
    eta_years: float,
    k_shape: float,
    beta_dot_x: float
) -> float:
    """
    Calculates the conditional failure probability over interval [a, a + Delta]
    given survival up to age a:
      P_fail = 1 - S(a + Delta | x) / S(a | x)
             = 1 - exp(- [((a + Delta)/eta)^k - (a/eta)^k] * exp(beta · x))
    """
    a = max(0.0, float(asset_age_years))
    delta = max(0.0, float(delta_years))
    eta = max(1.0, float(eta_years))
    k = max(WEIBULL_K_LOWER_BOUND, float(k_shape))

    if delta == 0.0:
        return 0.0

    delta_baseline_hazard = ((a + delta) / eta) ** k - (a / eta) ** k
    exponent = -delta_baseline_hazard * math.exp(beta_dot_x)

    # Protect against floating point underflow
    if exponent < -50.0:
        return 1.0
    return float(1.0 - math.exp(exponent))


def horizon_to_target_failure_prob(
    asset_age_years: float,
    eta_years: float,
    k_shape: float,
    beta_dot_x: float,
    target_p: float = REMAINING_LIFE_P_TARGET
) -> Tuple[float, float]:
    """
    Computes the estimated horizon Delta* (in years and days) until conditional
    failure probability reaches target_p (default 10%):
      DeltaH = -ln(1 - target_p) / exp(beta · x)
      Delta* = (a^k + eta^k * DeltaH)^(1/k) - a
    """
    a = max(0.0, float(asset_age_years))
    eta = max(1.0, float(eta_years))
    k = max(WEIBULL_K_LOWER_BOUND, float(k_shape))

    delta_h = -math.log(1.0 - target_p) / math.exp(beta_dot_x)
    term = (a ** k) + (eta ** k) * delta_h
    delta_star_years = (term ** (1.0 / k)) - a
    delta_star_days = delta_star_years * 365.0

    return max(0.0, delta_star_years), max(0.0, delta_star_days)


# ==============================================================================
# 4. CONSEQUENCE & EXPOSURE MODELING
# ==============================================================================

def calculate_consequence_factors(
    distance_to_facility_m: Optional[float],
    buildings_within_200m: int,
    road_class: str,
    asset_criticality: float
) -> Tuple[Dict[str, float], float]:
    """
    Computes individual consequence factors in [0, 1] and aggregates them via AHP weights:
      f_facility: exp(-d / 150)
      f_density: min(1, count / 150)
      f_road: mapped from highway hierarchy
      f_criticality: asset intrinsic importance
    """
    # 1. Facility Proximity
    if distance_to_facility_m is not None and distance_to_facility_m >= 0:
        f_facility = float(math.exp(-distance_to_facility_m / FACILITY_DECAY_DISTANCE_M))
    else:
        f_facility = 0.20  # Neutral background exposure if no critical facility within 1km

    # 2. Building Density
    f_density = float(min(1.0, max(0, buildings_within_200m) / DENSITY_REFERENCE_BUILDINGS))

    # 3. Road Hierarchy
    norm_road = road_class.lower().strip() if road_class else "default"
    f_road = float(ROAD_CLASS_WEIGHTS.get(norm_road, ROAD_CLASS_WEIGHTS["default"]))

    # 4. Asset Criticality
    f_criticality = float(min(max(asset_criticality, 0.0), 1.0))

    factors = {
        "facility_proximity": f_facility,
        "building_density": f_density,
        "road_class": f_road,
        "asset_criticality": f_criticality
    }

    # Aggregate using AHP weights
    weights = CONSEQUENCE_WEIGHTS
    C = float(
        weights[0] * f_facility +
        weights[1] * f_density +
        weights[2] * f_road +
        weights[3] * f_criticality
    )
    C = min(max(C, 0.0), 1.0)

    return factors, C


# ==============================================================================
# 5. EXPECTED RISK & WORST-CASE NORMALIZATION
# ==============================================================================

def compute_reference_worst_risk(asset_type: str) -> float:
    """
    Calculates reference worst-realistic risk R_ref for the asset class:
      x_sev = 1.0, x_rain = 1.0, x_traf = 1.0
      a = 0.8 * eta
      C = 1.0
    Used to normalize display score onto [0, 100].
    """
    asset_cfg = ASSET_TYPES.get(asset_type, ASSET_TYPES["road"])
    eta = asset_cfg["eta_years"]
    k = WEIBULL_SHAPE_K
    a_worst = WORST_CASE_AGE_FRACTION * eta

    # Covariate dot product with x = [1, 1, 1]
    beta_dot_x_worst = float(np.sum(BETA_VECTOR))

    p_fail_worst_90d = conditional_failure_probability(
        asset_age_years=a_worst,
        delta_years=HORIZON_90D_YEARS,
        eta_years=eta,
        k_shape=k,
        beta_dot_x=beta_dot_x_worst
    )
    # With C = 1.0:
    R_ref = p_fail_worst_90d * 1.0
    return max(R_ref, 1e-4)


def determine_risk_tier(score: float) -> str:
    """Classifies normalized risk score into High, Moderate, or Low."""
    if score >= SCORE_HIGH_THRESHOLD:
        return "High"
    elif score >= SCORE_MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Low"


# ==============================================================================
# 6. DETERMINISTIC MONTE CARLO UNCERTAINTY QUANTIFICATION
# ==============================================================================

def quantify_uncertainty(
    report_id: str,
    asset_age_years: float,
    eta_years: float,
    k_shape: float,
    beta_dot_x: float,
    consequence_c: float,
    r_ref: float,
    samples: int = MONTE_CARLO_SAMPLES
) -> Dict[str, Any]:
    """
    Performs seeded deterministic Monte Carlo uncertainty estimation.
    Perturbations:
      eta ~ LogNormal(0, 0.25) -> eta_sample = eta * exp(N(0, 0.25))
      k ~ Normal(k, 0.20), clamped >= 1.1
      hazard_multiplier_scale ~ LogNormal(0, 0.20)
    Returns p5, p50, p95 for 90-day P_fail and display risk score.
    """
    # Deterministic seed from report_id
    hash_digest = hashlib.md5(report_id.encode("utf-8")).hexdigest()
    seed = int(hash_digest, 16) % (2**31 - 1)
    rng = np.random.RandomState(seed)

    # Generate perturbations
    eta_noise = rng.normal(0.0, ETA_PERTURBATION_STD, samples)
    eta_samples = eta_years * np.exp(eta_noise)

    k_samples = rng.normal(k_shape, K_PERTURBATION_STD, samples)
    k_samples = np.maximum(k_samples, WEIBULL_K_LOWER_BOUND)

    beta_scale_noise = rng.normal(0.0, BETA_PERTURBATION_STD, samples)
    hazard_multiplier_samples = np.exp(beta_dot_x) * np.exp(beta_scale_noise)

    # Vectorized conditional failure probability
    a = max(0.0, float(asset_age_years))
    delta = HORIZON_90D_YEARS

    baseline_diff = ((a + delta) / eta_samples) ** k_samples - (a / eta_samples) ** k_samples
    exponents = -baseline_diff * hazard_multiplier_samples
    p_fail_samples = 1.0 - np.exp(np.clip(exponents, -50.0, 0.0))

    risk_samples = p_fail_samples * consequence_c
    score_samples = np.clip(100.0 * risk_samples / r_ref, 0.0, 100.0)

    # Percentiles
    p_fail_p5, p_fail_p50, p_fail_p95 = np.percentile(p_fail_samples, [5, 50, 95])
    score_p5, score_p50, score_p95 = np.percentile(score_samples, [5, 50, 95])

    return {
        "p_fail_90d": {
            "p5": float(round(p_fail_p5, 4)),
            "p50": float(round(p_fail_p50, 4)),
            "p95": float(round(p_fail_p95, 4))
        },
        "score": {
            "p5": float(round(score_p5, 1)),
            "p50": float(round(score_p50, 1)),
            "p95": float(round(score_p95, 1))
        },
        "monte_carlo_samples": samples,
        "seed": seed
    }


# ==============================================================================
# 7. UNIFIED PIPELINE EVALUATION & EXPLANATION BUILDER
# ==============================================================================

def evaluate_infrastructure_risk(
    report_id: str,
    asset_type: str,
    damage_probability: float,
    rainfall_forecast_24h_mm: float,
    antecedent_rain_5d_mm: float,
    traffic_index: float,
    asset_age_years: Optional[float] = None,
    distance_to_facility_m: Optional[float] = None,
    buildings_within_200m: int = 15,
    road_class: str = "secondary",
    custom_cn: Optional[float] = None,
    context_source: str = "overpass"
) -> Dict[str, Any]:
    """
    Executes the full end-to-end mathematical risk engine pipeline and constructs
    the complete explainable breakdown object.
    """
    # 1. Asset Configuration
    norm_asset = asset_type.lower().strip() if asset_type else "road"
    if norm_asset not in ASSET_TYPES:
        norm_asset = "road"
    asset_cfg = ASSET_TYPES[norm_asset]

    eta = float(asset_cfg["eta_years"])
    k = float(WEIBULL_SHAPE_K)
    age = float(asset_age_years if asset_age_years is not None and asset_age_years >= 0 else DEFAULT_ASSET_AGE_YEARS)

    # 2. Vision Hazard Covariate (x_sev ∈ [0, 1])
    x_sev = float(min(max(damage_probability, 0.0), 1.0))

    # 3. Hydrology (SCS-CN)
    base_cn = float(custom_cn if custom_cn is not None else asset_cfg["default_cn"])
    amc = determine_amc(antecedent_rain_5d_mm)
    adjusted_cn = adjust_cn_for_amc(base_cn, amc)
    runoff_q, S, Ia = calculate_scs_runoff(rainfall_forecast_24h_mm, adjusted_cn)
    x_rain = normalize_rainfall_hazard(runoff_q, Q_REF_MM)

    # 4. Traffic Hazard Covariate (x_traf ∈ [0, 1])
    # Normalize input traffic index (which is conventionally 0-100 or 0-1)
    if traffic_index > 1.0:
        x_traf = float(min(max(traffic_index / 100.0, 0.0), 1.0))
    else:
        x_traf = float(min(max(traffic_index, 0.0), 1.0))

    # 5. AHP Hazard Weighting
    x_vector = np.array([x_sev, x_rain, x_traf], dtype=np.float64)
    beta_dot_x = float(np.dot(BETA_VECTOR, x_vector))
    hazard_multiplier = float(math.exp(beta_dot_x))

    # 6. Weibull Conditional Failure Probabilities
    p_fail_30d = conditional_failure_probability(age, HORIZON_30D_YEARS, eta, k, beta_dot_x)
    p_fail_90d = conditional_failure_probability(age, HORIZON_90D_YEARS, eta, k, beta_dot_x)

    # 7. Horizon to 10% Conditional Failure Probability
    _, horizon_days = horizon_to_target_failure_prob(age, eta, k, beta_dot_x, REMAINING_LIFE_P_TARGET)
    display_horizon_days = int(min(horizon_days, MAX_DISPLAY_DAYS))

    # 8. Consequence Model
    consequence_factors, C = calculate_consequence_factors(
        distance_to_facility_m=distance_to_facility_m,
        buildings_within_200m=buildings_within_200m,
        road_class=road_class,
        asset_criticality=asset_cfg["criticality"]
    )

    # 9. Expected Risk & Worst-Realistic Normalization
    R = float(p_fail_90d * C)
    R_ref = compute_reference_worst_risk(norm_asset)
    raw_score = float(100.0 * (R / R_ref))
    score = float(min(100.0, max(0.0, raw_score)))
    tier = determine_risk_tier(score)

    # 10. Monte Carlo Uncertainty Quantification
    uncertainty = quantify_uncertainty(
        report_id=report_id,
        asset_age_years=age,
        eta_years=eta,
        k_shape=k,
        beta_dot_x=beta_dot_x,
        consequence_c=C,
        r_ref=R_ref
    )

    # 11. DeltaH for horizon
    delta_h = float(-math.log(1.0 - REMAINING_LIFE_P_TARGET) / hazard_multiplier)

    # Recommended action based on score and tier
    if tier == "High":
        recommended_action = "Immediate structural shoring & municipal priority dispatch"
    elif tier == "Moderate":
        recommended_action = "Schedule preventive maintenance inspection within 14 days"
    else:
        recommended_action = "Routine sensor & visual monitoring cycle"

    # 12. Complete Explainability Breakdown Object
    breakdown = {
        # Environmental / Hydrological Chain
        "rainfall_p_mm": round(rainfall_forecast_24h_mm, 2),
        "antecedent_rain_5d_mm": round(antecedent_rain_5d_mm, 2),
        "amc": amc,
        "base_cn": round(base_cn, 1),
        "adjusted_cn": round(adjusted_cn, 1),
        "retention_S_mm": round(S, 2),
        "initial_abstraction_Ia_mm": round(Ia, 2),
        "runoff_Q_mm": round(runoff_q, 2),
        "x_rain": round(x_rain, 4),

        # Structural & Operational Covariates
        "x_sev": round(x_sev, 4),
        "x_traf": round(x_traf, 4),
        "ahp_hazard_weights": {
            "w_sev": round(float(HAZARD_WEIGHTS[0]), 4),
            "w_rain": round(float(HAZARD_WEIGHTS[1]), 4),
            "w_traf": round(float(HAZARD_WEIGHTS[2]), 4)
        },
        "beta_weights": [round(float(b), 4) for b in BETA_VECTOR],
        "hazard_multiplier": round(hazard_multiplier, 3),

        # Reliability & Asset Characteristics
        "asset_type": norm_asset,
        "asset_name": asset_cfg["name"],
        "eta_years": eta,
        "k_shape": k,
        "asset_age_years": age,
        "delta_30d_years": round(HORIZON_30D_YEARS, 4),
        "delta_90d_years": round(HORIZON_90D_YEARS, 4),
        "delta_H": round(delta_h, 5),
        "p_fail_30d": round(p_fail_30d, 4),
        "p_fail_90d": round(p_fail_90d, 4),
        "days_to_10pct": display_horizon_days,

        # Consequence Chain
        "consequence_factors": {k: round(v, 4) for k, v in consequence_factors.items()},
        "ahp_consequence_weights": {
            "facility_proximity": round(float(CONSEQUENCE_WEIGHTS[0]), 4),
            "building_density": round(float(CONSEQUENCE_WEIGHTS[1]), 4),
            "road_class": round(float(CONSEQUENCE_WEIGHTS[2]), 4),
            "asset_criticality": round(float(CONSEQUENCE_WEIGHTS[3]), 4)
        },
        "consequence_C": round(C, 4),

        # Risk & Scoring
        "risk_index_R": round(R, 5),
        "reference_risk_R_ref": round(R_ref, 5),
        "score": round(score, 1),
        "tier": tier,

        # Uncertainty bounds
        "uncertainty": uncertainty,

        # Metadata & Provenance
        "assumptions": [
            "Weibull shape k=2.0 indicates progressive structural wear-out degradation prior.",
            "Characteristic life eta derived from ASCE / IRC urban asset engineering manuals.",
            "SCS-CN runoff follows USDA NRCS National Engineering Handbook guidelines.",
            "Maximum hazard scaling bounded to exp(ln(10)) = 10x multiplier.",
            "Monte Carlo uncertainty computed deterministically from report ID hash."
        ],
        "context_source": context_source,
        "recommended_action": recommended_action
    }

    return {
        "prediction": "Damaged" if x_sev >= 0.50 else "Safe",
        "damage_probability": round(x_sev, 4),
        "risk_score": int(round(score)),
        "priority": tier,
        "failure_probability": int(round(p_fail_90d * 100)),
        "failure_probability_30d": round(p_fail_30d, 4),
        "failure_probability_90d": round(p_fail_90d, 4),
        "estimated_failure_days": display_horizon_days,
        "consequence_score": round(C, 4),
        "risk_index": round(R, 5),
        "recommended_action": recommended_action,
        "uncertainty_range": f"{int(round(uncertainty['score']['p5']))} – {int(round(uncertainty['score']['p95']))}",
        "breakdown": breakdown
    }
