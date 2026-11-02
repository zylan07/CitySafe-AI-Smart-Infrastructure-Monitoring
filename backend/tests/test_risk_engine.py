"""
Unit and Property Tests for CitySafe Risk Engine
================================================
Verifies 19 formal mathematical properties:
  1. Q >= 0
  2. Q monotonically increases with P
  3. Q < P - Ia for P > Ia
  4. Q = 0 when P <= Ia
  5. x_rain in [0, 1)
  6. x_rain monotonically increases with Q
  7. P_fail in [0, 1]
  8. P_fail increases with severity
  9. P_fail increases with rainfall
  10. P_fail increases with traffic
  11. P_fail increases with Delta
  12. P_fail = 0 when Delta = 0
  13. Conditional failure probability identity: 1 - S(a+Delta)/S(a)
  14. Weibull PH closure property: eta_x = eta * exp(-beta·x / k)
  15. Delta* round-trip reaches target probability (10%)
  16. AHP lambda_max >= n (Perron-Frobenius theorem)
  17. Configured AHP CR < 0.10 for both Hazard and Consequence matrices
  18. Deterministic output: exact identical outputs across repeated calls
  19. Risk ranking invariance under monotonic score normalization
"""

import math
import numpy as np
import pytest

from backend.config import (
    ASSET_TYPES,
    WEIBULL_SHAPE_K,
    HORIZON_30D_YEARS,
    HORIZON_90D_YEARS,
    AHP_HAZARD_MATRIX,
    RI_N3,
    AHP_CONSEQUENCE_MATRIX,
    RI_N4,
    REMAINING_LIFE_P_TARGET
)
from backend.risk_engine import (
    solve_ahp,
    calculate_scs_runoff,
    normalize_rainfall_hazard,
    weibull_cumulative_hazard,
    weibull_survival,
    conditional_failure_probability,
    horizon_to_target_failure_prob,
    evaluate_infrastructure_risk,
    HAZARD_WEIGHTS,
    HAZARD_CR,
    CONSEQUENCE_WEIGHTS,
    CONSEQUENCE_CR,
    BETA_VECTOR
)


# ==============================================================================
# SCS-CN HYDROLOGICAL PROPERTIES (Tests 1 - 4)
# ==============================================================================

def test_1_runoff_non_negative():
    """Property 1: Estimated surface runoff Q >= 0 for all rainfall amounts."""
    for p in [0.0, 5.0, 15.0, 30.0, 100.0, 250.0]:
        for cn in [50, 75, 90, 98]:
            Q, S, Ia = calculate_scs_runoff(p, cn)
            assert Q >= 0.0, f"Runoff Q must be >= 0, got {Q} for P={p}, CN={cn}"


def test_2_runoff_monotonic_in_rainfall():
    """Property 2: Runoff Q monotonically increases with precipitation P."""
    cn = 80
    p_values = np.linspace(0, 250, 100)
    q_values = [calculate_scs_runoff(p, cn)[0] for p in p_values]
    for i in range(len(q_values) - 1):
        assert q_values[i + 1] >= q_values[i], f"Monotonicity violated at P={p_values[i]}"


def test_3_runoff_bounded_by_effective_rainfall():
    """Property 3: Runoff Q cannot exceed effective rainfall (P - Ia) when P > Ia."""
    cn = 75
    for p in [30.0, 60.0, 120.0, 250.0]:
        Q, S, Ia = calculate_scs_runoff(p, cn)
        if p > Ia:
            assert Q < (p - Ia), f"Q={Q} must be strictly less than effective rain {p - Ia}"


def test_4_runoff_zero_below_initial_abstraction():
    """Property 4: Q = 0 when P <= Ia."""
    for cn in [60, 75, 90]:
        _, S, Ia = calculate_scs_runoff(0, cn)
        p_below = Ia * 0.95
        Q, _, _ = calculate_scs_runoff(p_below, cn)
        assert Q == 0.0, f"Expected Q=0 when P <= Ia, got {Q}"


# ==============================================================================
# RAINFALL HAZARD NORMALIZATION PROPERTIES (Tests 5 - 6)
# ==============================================================================

def test_5_x_rain_bounded():
    """Property 5: x_rain ∈ [0, 1) for all non-negative runoff values."""
    for q in [0.0, 0.1, 10.0, 50.0, 100.0, 500.0, 10000.0]:
        x_rain = normalize_rainfall_hazard(q)
        assert 0.0 <= x_rain < 1.0, f"x_rain out of bounds: {x_rain} for Q={q}"


def test_6_x_rain_monotonic_in_q():
    """Property 6: x_rain monotonically increases with runoff Q."""
    q_values = np.linspace(0, 300, 50)
    x_values = [normalize_rainfall_hazard(q) for q in q_values]
    for i in range(len(x_values) - 1):
        assert x_values[i + 1] >= x_values[i]


# ==============================================================================
# CONDITIONAL FAILURE PROBABILITY PROPERTIES (Tests 7 - 12)
# ==============================================================================

def test_7_p_fail_bounded():
    """Property 7: Conditional failure probability P_fail ∈ [0, 1]."""
    for age in [0.0, 5.0, 15.0, 30.0]:
        for delta in [HORIZON_30D_YEARS, HORIZON_90D_YEARS, 1.0, 5.0]:
            p = conditional_failure_probability(age, delta, 30.0, 2.0, 1.5)
            assert 0.0 <= p <= 1.0, f"P_fail out of bounds: {p}"


def test_8_p_fail_increases_with_severity():
    """Property 8: P_fail monotonically increases with damage severity x_sev."""
    p_low = evaluate_infrastructure_risk("r1", "road", 0.10, 20.0, 10.0, 50.0)["failure_probability_90d"]
    p_med = evaluate_infrastructure_risk("r1", "road", 0.50, 20.0, 10.0, 50.0)["failure_probability_90d"]
    p_high = evaluate_infrastructure_risk("r1", "road", 0.90, 20.0, 10.0, 50.0)["failure_probability_90d"]
    assert p_low < p_med < p_high


def test_9_p_fail_increases_with_rainfall():
    """Property 9: P_fail monotonically increases with forecast rainfall."""
    p_dry = evaluate_infrastructure_risk("r1", "drain", 0.60, 0.0, 5.0, 40.0)["failure_probability_90d"]
    p_mod = evaluate_infrastructure_risk("r1", "drain", 0.60, 40.0, 5.0, 40.0)["failure_probability_90d"]
    p_heavy = evaluate_infrastructure_risk("r1", "drain", 0.60, 150.0, 5.0, 40.0)["failure_probability_90d"]
    assert p_dry <= p_mod < p_heavy


def test_10_p_fail_increases_with_traffic():
    """Property 10: P_fail increases with traffic loading."""
    p_light = evaluate_infrastructure_risk("r1", "road", 0.50, 25.0, 10.0, 10.0)["failure_probability_90d"]
    p_heavy = evaluate_infrastructure_risk("r1", "road", 0.50, 25.0, 10.0, 95.0)["failure_probability_90d"]
    assert p_heavy > p_light


def test_11_p_fail_increases_with_horizon_delta():
    """Property 11: P_fail over 90 days must be strictly greater than over 30 days."""
    res = evaluate_infrastructure_risk("r1", "culvert", 0.50, 30.0, 15.0, 50.0)
    p_30 = res["failure_probability_30d"]
    p_90 = res["failure_probability_90d"]
    assert p_90 > p_30


def test_12_p_fail_zero_at_delta_zero():
    """Property 12: Conditional failure probability is 0 when evaluation horizon Delta = 0."""
    p = conditional_failure_probability(10.0, 0.0, 30.0, 2.0, 1.2)
    assert p == 0.0


# ==============================================================================
# MATHEMATICAL IDENTITIES & THEOREMS (Tests 13 - 15)
# ==============================================================================

def test_13_conditional_probability_identity():
    """
    Property 13: Verifies analytical identity:
      P(a < T <= a + Delta | T > a) = 1 - S(a + Delta) / S(a)
    Calculated independently via survival function.
    """
    age = 12.0
    delta = 0.5
    eta = 35.0
    k = 2.2
    beta_dot_x = 1.4

    # 1. Survival ratio identity
    S_a = weibull_survival(age, eta, k, beta_dot_x)
    S_a_plus_delta = weibull_survival(age + delta, eta, k, beta_dot_x)
    p_expected = 1.0 - (S_a_plus_delta / S_a)

    # 2. Risk engine implementation
    p_actual = conditional_failure_probability(age, delta, eta, k, beta_dot_x)

    assert math.isclose(p_expected, p_actual, rel_tol=1e-7, abs_tol=1e-7)


def test_14_weibull_ph_closure():
    """
    Property 14: Verifies Weibull PH closure property:
      H(t|x) = (t / eta)^k * exp(beta · x) == (t / eta_x)^k
      where eta_x = eta * exp(-beta · x / k)
    """
    t = 14.5
    eta = 40.0
    k = 2.3
    beta_dot_x = 1.85

    # Original PH representation
    H_ph = weibull_cumulative_hazard(t, eta, k, beta_dot_x)

    # Scale-accelerated representation
    eta_x = eta * math.exp(-beta_dot_x / k)
    H_scale = (t / eta_x) ** k

    assert math.isclose(H_ph, H_scale, rel_tol=1e-7, abs_tol=1e-7)


def test_15_delta_star_round_trip():
    """
    Property 15: Verifies that solving for Delta* at target_p (10%) and evaluating
    conditional failure probability over [a, a + Delta*] returns exactly 10%.
    """
    age = 9.0
    eta = 25.0
    k = 2.0
    beta_dot_x = 1.15
    target = REMAINING_LIFE_P_TARGET

    delta_star_years, _ = horizon_to_target_failure_prob(age, eta, k, beta_dot_x, target)
    p_at_delta_star = conditional_failure_probability(age, delta_star_years, eta, k, beta_dot_x)

    assert math.isclose(p_at_delta_star, target, rel_tol=1e-5, abs_tol=1e-5)


# ==============================================================================
# AHP THEOREMS & CONSISTENCY (Tests 16 - 17)
# ==============================================================================

def test_16_ahp_perron_frobenius():
    """
    Property 16: Perron-Frobenius theorem asserts lambda_max >= n
    for any reciprocal positive pairwise comparison matrix.
    """
    _, lambda_h, _, _ = solve_ahp(AHP_HAZARD_MATRIX, RI_N3)
    assert lambda_h >= 3.0, f"Hazard lambda_max={lambda_h} must be >= 3"

    _, lambda_c, _, _ = solve_ahp(AHP_CONSEQUENCE_MATRIX, RI_N4)
    assert lambda_c >= 4.0, f"Consequence lambda_max={lambda_c} must be >= 4"


def test_17_ahp_consistency_ratio():
    """Property 17: Configured AHP matrices must have Consistency Ratio CR < 0.10."""
    assert HAZARD_CR < 0.10, f"Hazard CR {HAZARD_CR} exceeds Saaty consistency threshold 0.10"
    assert CONSEQUENCE_CR < 0.10, f"Consequence CR {CONSEQUENCE_CR} exceeds Saaty consistency threshold 0.10"


# ==============================================================================
# DETERMINISM & RANKING INVARIANCE (Tests 18 - 19)
# ==============================================================================

def test_18_deterministic_output():
    """Property 18: Risk engine must be strictly deterministic across calls with identical inputs."""
    res1 = evaluate_infrastructure_risk("fixed-report-id-123", "culvert", 0.75, 45.0, 20.0, 65.0)
    res2 = evaluate_infrastructure_risk("fixed-report-id-123", "culvert", 0.75, 45.0, 20.0, 65.0)

    assert res1["risk_score"] == res2["risk_score"]
    assert res1["failure_probability"] == res2["failure_probability"]
    assert res1["consequence_score"] == res2["consequence_score"]
    assert res1["uncertainty_range"] == res2["uncertainty_range"]
    assert res1["breakdown"]["uncertainty"]["score"] == res2["breakdown"]["uncertainty"]["score"]


def test_19_risk_ranking_invariance():
    """
    Property 19: For any two assets of the same type, if R_A > R_B,
    then score_A >= score_B under monotonic score normalization.
    """
    res_A = evaluate_infrastructure_risk("rep_A", "road", 0.85, 80.0, 25.0, 80.0)
    res_B = evaluate_infrastructure_risk("rep_B", "road", 0.20, 10.0, 5.0, 20.0)

    r_A = res_A["breakdown"]["risk_index_R"]
    r_B = res_B["breakdown"]["risk_index_R"]

    score_A = res_A["breakdown"]["score"]
    score_B = res_B["breakdown"]["score"]

    assert r_A > r_B
    assert score_A > score_B

