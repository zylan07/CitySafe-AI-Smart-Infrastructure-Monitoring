# 🚀 CitySafe / ZylanZest V2 — AI-Powered Framework for Real-Time Vulnerability Assessment of Unmapped Urban Infrastructure

> *"Transforming unmapped urban infrastructure from reactive failure management into an explainable, uncertainty-aware reliability intelligence platform."*

**Institution:** KPR Institute of Engineering and Technology, Department of Computer Science and Engineering  
**Project Track:** Proto Studio Advanced Upgrade (Design Studio II Capstone)  
**Authors:**
- **Sathya R V** — 24CS192
- **Shayden S** — 24CS195
- **Sudharsan S M** — 24CS218
- **Tajmal Hussain S** — 25CSL20

---

## 1. Executive Summary & Objective

Urban public works authorities face a systemic challenge: municipal infrastructure assets (cross-drainage culverts, stormwater drains, roadside retaining walls, footpaths, and secondary road pavements) often degrade silently. Traditional municipal systems either rely on reactive citizen complaints (basic CRUD dashboards) or apply arbitrary heuristic scores with unseeded random noise.

**CitySafe V2** bridges the gap between vision AI and civil reliability engineering by introducing:
1. **Calibrated Computer Vision:** MobileNetV3-style depthwise-separable CNN with post-hoc temperature scaling ($T = 0.8695$) and 5×5 occlusion sensitivity explainability heatmaps (with legacy KNN fallback).
2. **Deterministic Hydrological Modeling:** USDA SCS Curve Number (SCS-CN) runoff formulation accounting for soil retention ($S$), initial abstraction ($I_a$), and 5-day Antecedent Moisture Conditions (AMC I, II, III).
3. **Analytical Hierarchy Process (AHP):** Mathematically consistent pairwise criteria weights for hazard covariates and consequence exposure, rigorously satisfying Saaty's consistency test ($\text{CR} < 0.10$).
4. **Weibull Proportional Hazards:** Continuous failure modeling conditioned on asset operational age $a$, calculating 30-day and 90-day conditional failure probabilities and the estimated horizon to 10% failure threshold.
5. **Expected Risk Index & Deterministic Uncertainty:** Formulated from statistical decision theory ($R = P_{\text{fail}} \times C$) and bounded by seeded Monte Carlo simulation ($N=500$) returning $p5 - p95$ risk intervals.
6. **Spatial Intelligence & Optimization:** Getis-Ord $G_i^*$ spatial hotspot detection ($z \ge 1.96$) and greedy fractional knapsack repair scheduling under fixed municipal budgets.

---

## 2. System Architecture & Causal Pipeline

```
Citizen Inspection Image + GPS Telemetry
                  │
                  ▼
   Vision Classification & Temperature Scaling
     (CitySafe MobileNet CNN / Legacy KNN)
                  │
                  ▼
Calibrated Damage Probability x_sev ∈ [0, 1]
                  │
                  ├── 5-day Antecedent Rain (Open-Meteo) ──► AMC Class (I, II, III)
                  ├── 24h Rainfall Forecast P (mm) ───────► SCS Curve Number Runoff Q (mm)
                  ▼                                           │
  Rainfall Hazard Covariate ◄─────────────────────────────────┘
     x_rain = 1 - exp(-Q / 100) ∈ [0, 1)
                  │
                  ├── Deterministic Peak-Hour Traffic Index x_traf ∈ [0, 1]
                  │
                  ▼
     AHP Hazard Covariate Multiplier
     β = ln(10) · w,  exp(β · x)  [CR = 0.0079 < 0.10]
                  │
                  ├── Asset Prior Lifespan η (years) & Shape k = 2.0 (Wear-out)
                  ├── Asset Age a (years) & Horizon Δ (30d, 90d)
                  ▼
     Weibull Proportional Hazards Model
     P_fail(Δ|a, x) = 1 - exp(-[((a+Δ)/η)^k - (a/η)^k] · exp(β · x))
     Horizon to 10% Failure: Δ* = (a^k + η^k · ΔH)^(1/k) - a
                  │
                  ├── OpenStreetMap / Overpass Context (Facility dist, Building density, Road class)
                  ├── AHP Consequence Weighting [CR = 0.0076 < 0.10]
                  ▼
     Multi-Criteria Consequence C ∈ [0, 1]
                  │
                  ▼
     Expected Loss Risk Index R = P_fail_90d × C
                  │
                  ├── Reference Worst-Case Normalization: 100 · R / R_ref
                  ├── Deterministic Seeded Monte Carlo Uncertainty (p5, p50, p95)
                  │
                  ├── Spatial Getis-Ord Gi* Hotspot Analysis (500m binary weights)
                  ├── Bayesian Multi-Report Evidence Fusion (50m radius)
                  └── Budget-Constrained Repair Optimization (Greedy R / Cost)
```

---

## 3. Mathematical Foundations & Summary of Proofs

Full analytical derivations and proofs are detailed in [`docs/MATH.md`](file:///docs/MATH.md).

### 3.1 SCS-CN Runoff Monotonicity & Conservation of Mass
- **Monotonicity:** For effective rainfall $u = P - I_a > 0$ and maximum retention $S > 0$:
  $$\frac{dQ}{du} = \frac{u(u + 2S)}{(u + S)^2} > 0$$
  Runoff strictly increases with storm rainfall volume.
- **Mass Boundedness:** 
  $$u - Q = \frac{uS}{u + S} > 0 \implies Q < P - I_a < P$$
  Runoff volume can never exceed effective precipitation.

### 3.2 Age-Conditioned Failure Probability Identity
Given survival up to age $a$, probability of failure within forward horizon $[a, a + \Delta]$:
$$P(a < T \le a + \Delta \mid T > a) = 1 - \frac{S(a + \Delta \mid x)}{S(a \mid x)} = 1 - \exp\left(-\left[\left(\frac{a + \Delta}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x)\right)$$

### 3.3 Weibull PH Accelerated Life Closure
Rewriting cumulative hazard shows that proportional hazards covariate multiplication directly scales characteristic lifespan:
$$H(t \mid x) = \left(\frac{t}{\eta}\right)^k \exp(\beta \cdot x) = \left(\frac{t}{\eta \exp(-\beta \cdot x / k)}\right)^k = \left(\frac{t}{\eta(x)}\right)^k$$

### 3.4 Expected Loss Risk Formulation
$$\mathbb{E}[\text{Loss}] = P(\text{Failure}) \cdot \mathbb{E}[\text{Loss} \mid \text{Failure}] \implies R = P_{\text{fail, 90d}} \times C$$

---

## 4. Key Upgrades in Version 2.0

| Feature | Legacy V1 (Design Studio I/II) | Upgraded V2 (Proto Studio) |
| :--- | :--- | :--- |
| **Scoring Determinism** | Random jitter `random.randint(0, 10)` | **100% Deterministic** pure mathematical engine |
| **Vision Model** | Flattened 20×20 grayscale 7-NN | **MobileNet CNN** + Temperature Scaling + KNN Fallback |
| **Explainability** | Generic static text | **5×5 Occlusion Heatmap** + Step-by-Step Mathematical Breakdown |
| **Environmental Stress** | Arbitrary linear multiplier | **SCS Curve Number Runoff** + AMC Saturation Dynamics |
| **Aging Dynamics** | Ageless assumption | **Weibull Wear-Out ($k=2.0$)** conditioned on operational age |
| **Uncertainty** | Single point estimate | **Deterministic Monte Carlo ($N=500$)** returning $p5 - p95$ |
| **Spatial Analytics** | Standard marker map | **Getis-Ord $G_i^*$ Hotspot Detection** ($p < 0.05, p < 0.01$) |
| **Crowd Aggregation** | Simple duplicate alert | **Bayesian Log-Odds Evidence Fusion** for clustered reports |
| **Decision Support** | Simple queue listing | **Budget-Constrained Repair Optimization** ($R / \text{Cost}$) |
| **Simulation** | None | **Live What-If Rainfall Slider** (0–250mm) calling `/api/v2/simulate` |

---

## 5. Measured Experimental Results

All reported values come directly from executed benchmark runs on the held-out test partition (74 images: 37 Positive, 37 Negative) and simulation validation (1,000 synthetic assets):

### Vision & Calibration Metrics (70/15/15 Stratified Split on 500 Images)

> **Critical Model Evaluation Finding:** The implemented CNN (MobileNetV3-Small style depthwise-separable architecture) is **experimental**. On the current 500-image dataset, the **Legacy KNN baseline ($k=7$) remains the significantly stronger classifier**, achieving much higher accuracy (0.8108 vs. 0.6081) and substantially higher recall on damaged infrastructure (0.6216 vs. 0.2162). The CNN should NOT be presented as superior to KNN. KNN remains the stronger baseline for production inspection until a larger multi-thousand image dataset or threshold-tuning strategy is implemented.

- **Experimental MobileNet-Style CNN:**
  - Architecture: Lightweight Depthwise-Separable CNN (MobileNetV3-Small style)
  - Accuracy: **0.6081**
  - Precision: **1.0000**
  - Recall: **0.2162**
  - F1 Score: **0.3556**
  - ROC-AUC: **0.9671**
  - PR-AUC: **0.9745**
  - Optimal Temperature ($T$): **0.8695** (Fitted via validation NLL optimization)
  - ECE Before Calibration: **0.1315**
  - ECE After Calibration: **0.1406** *(Note: Temperature scaling optimized validation NLL, but did not decrease test ECE on this small partition)*

- **Legacy KNN Baseline ($k=7$ on Flattened Grayscale) — [Stronger Baseline]:**
  - Accuracy: **0.8108**
  - Precision: **1.0000**
  - Recall: **0.6216**
  - F1 Score: **0.7667**
  - ROC-AUC: **0.8755**
  - PR-AUC: **0.8785**

### Reliability & Simulation-Based Validation (Synthetic Simulation Only)
> **Disclaimer:** The following validation metrics are evaluated on **1,000 synthetic simulation assets** with known Weibull PH ground-truth lifetimes. They validate the mathematical consistency of the pipeline and are **not fitted to real municipal failure records**.

- **Brier Score Loss (Simulation):** **0.0156**
- **Kendall's $\tau$ Ranking Stability (Simulation):** **0.2432**
- **Top-20% Risk Recall (Simulation):** **37.5%** (Captured 6 out of 16 actual failures in top 20% highest risk queue)
- **Property Test Suite:** **26 passed** out of 26 tests in 25.40 seconds.

---

## 6. Project Directory Structure

```
CitySafe-Dashboard/
├── backend/
│   ├── config.py              # Single source of truth for priors & AHP matrices
│   ├── risk_engine.py         # Pure Python/NumPy/SciPy mathematical engine
│   ├── context.py             # Overpass OSM exposure queries with disk cache
│   ├── spatial.py             # Getis-Ord Gi*, Bayesian fusion & knapsack prioritization
│   ├── vision_service.py      # Inference, temperature calibration & occlusion heatmaps
│   ├── train_vision.py        # PyTorch CNN training, calibration & plotting script
│   ├── train_knn.py           # Legacy KNN baseline training script
│   ├── server.py              # FastAPI server (v1 backward compatible + v2 research API)
│   ├── model/
│   │   ├── citysafe_cnn.pt    # Trained PyTorch CNN weights
│   │   ├── concrete_knn_model.pkl # Legacy KNN model artifact
│   │   └── vision_metrics.json # Stored benchmark metrics & temperature T
│   ├── database/
│   │   └── reports.json       # JSON persistence store
│   └── tests/
│       ├── test_risk_engine.py # 19 mathematical property tests
│       └── test_api.py        # 7 integration tests for v1 & v2 endpoints
├── dataset/
│   ├── Positive/              # 250 Damaged/Cracked concrete images (227x227)
│   └── Negative/              # 250 Sound/Undamaged concrete images (227x227)
├── frontend/
│   ├── index.html             # Landing portal
│   ├── style.css              # Main portal styling
│   ├── user/                  # Citizen Portal (Submission, Geolocation, Occlusion Map, Explain Score)
│   ├── admin/                 # Command Center (KPIs, Live Map, What-If Slider, Hotspots, Prioritization)
│   └── model/
│       └── index.html         # KaTeX Model & Mathematical Foundations page
├── docs/
│   ├── IMPLEMENTATION_PLAN.md # Audit findings & architecture roadmap
│   ├── MATH.md                # Complete mathematical proofs & epistemic taxonomy
│   ├── DEMO_SCRIPT.md         # 5-minute timed presentation script & 20 Viva Q&As
│   └── figures/               # Generated publication-quality evaluation plots
│       ├── reliability_diagram.png
│       ├── roc_pr_curves.png
│       ├── baseline_comparison.png
│       ├── robustness_stress.png
│       ├── calibration_plot.png
│       ├── tornado_plot.png
│       └── ranking_stability_plot.png
└── scripts/
    ├── seed_synthetic.py      # Realistic 60-node demonstration cluster generator
    └── validate_engine.py     # Ground-truth Weibull PH simulation validator
```

---

## 7. Execution Commands

### 1. Run All Tests
```bash
python -m pytest backend/tests/ -v
```

### 2. Start Backend Server
```bash
python backend/server.py
# Server runs on http://localhost:8000
```

### 3. Open Web Dashboards
- **Landing Portal:** `http://localhost:8000/portal/index.html` (or open `frontend/index.html` in browser)
- **Citizen Portal:** `http://localhost:8000/portal/user/index.html`
- **Command Center:** `http://localhost:8000/portal/admin/index.html`
- **Model & Mathematical Foundations:** `http://localhost:8000/portal/model/index.html`

### 4. Re-Seed Synthetic Demonstration Data
```bash
python scripts/seed_synthetic.py
# Use --reset to restore initial clean state
python scripts/seed_synthetic.py --reset
```

### 5. Re-Run Validation Simulation & Generate Figures
```bash
python scripts/validate_engine.py
```

### 6. Re-Train Vision Model
```bash
python backend/train_vision.py
```

---

## 8. Epistemic Assumptions & Transparent Limitations

In accordance with strict academic engineering ethics:

1. **Lifespan Priors vs. Municipal Data:**
   The current prototype uses literature-informed and expert-elicited priors ($\eta = 15-45\text{ years}$, $k = 2.0$) and validates the mathematical risk pipeline using controlled synthetic simulations. **The model has not yet learned failure rates from real municipal work orders**, because Coimbatore municipal failure records are currently unmapped and un-digitized.
2. **Vision Scope:**
   The CNN is trained to detect surface fissures and cracks on concrete. It does not inspect sub-surface voids or subterranean pipe collapse.
3. **Overpass Latency:**
   Live OpenStreetMap Overpass queries require 1–3 seconds; our local grid caching layer ensures real-time sub-second responses on repeated coordinates.

---

## 9. Academic & Technical References

1. **USDA Natural Resources Conservation Service (NRCS):** *National Engineering Handbook, Section 4: Hydrology (NEH-4)*. United States Department of Agriculture.
2. **Intergovernmental Panel on Climate Change (IPCC):** *Climate Change 2014: Impacts, Adaptation, and Vulnerability. Part A: Global and Sectoral Aspects (AR5 WGII)*. Cambridge University Press.
3. **Saaty, T. L. (1980):** *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill International.
4. **Cox, D. R. (1972):** "Regression Models and Life-Tables." *Journal of the Royal Statistical Society: Series B (Methodological)*, 34(2), 187–202.
5. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017):** "On Calibration of Modern Neural Networks." *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1321–1330.
6. **Ord, J. K., & Getis, A. (1995):** "Local Spatial Autocorrelation Statistics: Distributional Issues and an Application." *Geographical Analysis*, 27(4), 286–306.
7. **Özgenel, Ç. F. (2019):** "Concrete Crack Images for Classification." *Mendeley Data*, v2.
8. **Dorafshan, S., Thomas, R. J., & Maguire, M. (2018):** "SDNET2018: An annotated image dataset for training, validation, and benchmarking of artificial neural networks for detection of cracks in concrete." *Data in Brief*, 21, 1664–1668.
