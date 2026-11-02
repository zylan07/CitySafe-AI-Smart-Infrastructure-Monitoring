# CitySafe / ZylanZest V2 — Implementation & Upgrade Plan
**Institution:** KPR Institute of Engineering and Technology, Dept. of CSE  
**Project:** CitySafe — AI-Powered Framework for Real-Time Vulnerability Assessment of Unmapped Urban Infrastructure  
**Authors:** Sathya R V (24CS192), Shayden S (24CS195), Sudharsan S M (24CS218), Tajmal Hussain S (25CSL20)

---

## 1. Baseline Audit Findings

1. **Python Environment:** Python 3.14.2 (64-bit Windows). PyTorch 2.11.0, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0, Pillow 12.1.1, FastAPI 0.136.0, Uvicorn 0.44.0 are active.
2. **Dataset:**
   - `dataset/Positive/`: 250 images (cracked/damaged concrete surfaces)
   - `dataset/Negative/`: 250 images (sound/undamaged concrete surfaces)
   - Total: 500 images, all 227×227 RGB.
3. **Existing Model:**
   - 7-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=7)`) trained on 20×20 grayscale flattened images (400 dimensions).
   - Saved as `concrete_knn_model.pkl`.
4. **Existing API Endpoints:**
   - `POST /api/analyze`
   - `POST /api/report/save`
   - `GET /api/reports`
   - `PUT /api/report/update/{id}`
   - `PUT /api/report/log_view/{id}`
   - `GET /api/notifications`
5. **Existing Frontend Pages:**
   - `frontend/index.html`: Portal navigation.
   - `frontend/user/index.html`: Citizen report submission, GPS/search, Open-Meteo weather fetch, Leaflet pin, history tab.
   - `frontend/admin/index.html`: Municipal operator dashboard, KPI summary, Leaflet map with risk filters, report review modal, triage queue, trend chart.
6. **Existing Database / Reports:**
   - `backend/database/reports.json`: Flat JSON array with 10 existing sample reports.
7. **Existing Weather & Traffic:**
   - Open-Meteo current precipitation API. Falling back to `Math.floor(Math.random() * 40) + 10` on 0 rain or failure.
   - Traffic: hour-of-day heuristic + random perturbation `[-10, 10]`.
8. **Existing Risk Formula:**
   - `base_score = 80 if Damaged else 30`
   - `risk_score = base_score + 0.3*rainfall + 0.2*traffic`
   - `failure_probability = min(100, risk_score + random.randint(0, 10))`
   - `estimated_failure_days = 7 / 15 / 30` step function.

---

## 2. Upgraded Architecture

```
User Image + Geolocation (Lat/Lon)
       │
       ▼
Vision Inference (MobileNetV3 / Legacy KNN Fallback)
       │
       ▼
Calibrated Damage Probability (Temperature Scaled x_sev ∈ [0, 1])
       │
       ├── Historical 5-day antecedent rainfall (Open-Meteo) ──► AMC Class (I, II, III)
       ├── Forecast rainfall P (mm) ──────────────────────────► SCS Curve Number Runoff (Q mm)
       ▼                                                          │
Rainfall Hazard Covariate ◄───────────────────────────────────────┘
       x_rain = 1 - exp(-Q / Q_ref) ∈ [0, 1)
       │
       ├── Deterministic Traffic Covariate x_traf ∈ [0, 1]
       │
       ▼
AHP Hazard Weighting (Saaty Matrix: Severity, Rainfall, Traffic; CR < 0.10)
       β = B · w,  B = ln(10)
       │
       ├── Asset Type Prior: Weibull scale η (years), shape k (wear-out prior = 2.0)
       ├── Asset Age a (years) & Horizon Δ (30d, 90d)
       ▼
Weibull Proportional Hazards Model
       H(t|x) = (t / η)^k · exp(β · x)
       P_fail(Δ|a, x) = 1 - exp(-[((a+Δ)/η)^k - (a/η)^k] · exp(β · x))
       Horizon to 10% Conditional Failure: Δ* = (a^k + η^k · ΔH)^(1/k) - a
       │
       ├── Consequence Covariates (Facility proximity, Building density, Road class, Asset criticality)
       ├── AHP Consequence Weighting (CR < 0.10)
       ▼
Consequence Index C ∈ [0, 1]
       │
       ▼
Expected Risk R = P_fail_90d · C
       │
       ▼
Normalised Display Score: 100 · R / R_ref (Worst-case calibration)
       │
       ▼
Deterministic Monte Carlo Uncertainty (N=500, seed=hash(report_id)): p5, p50, p95
       │
       ├── Multi-Report Spatial Fusion (Haversine 50m, Bayesian Odds Update)
       ├── Spatial Hotspot Detection (Getis-Ord Gi* with distance weighting)
       └── Repair Prioritisation (Greedy R / Cost fractional knapsack relaxation)
```

---

## 3. Implementation Phases

- **Phase 1:** Configuration & Mathematical Risk Engine
  - `backend/config.py`: Single source of truth for all literature priors, AHP matrices, SCS-CN tables, asset parameters.
  - `backend/risk_engine.py`: Pure Python/NumPy/SciPy engine with Weibull PH, SCS-CN, AHP, Monte Carlo uncertainty.
  - `docs/MATH.md`: Rigorous mathematical proofs, identities, theorems, and classifications.
  - `backend/tests/test_risk_engine.py`: Full property test suite (19 test categories).

- **Phase 2:** API v2 Endpoints & Backward Compatibility
  - `POST /api/v2/analyze`: Full v2 risk pipeline with explainability breakdown.
  - `POST /api/v2/simulate`: What-if simulation engine without image requirements.
  - `GET /api/v2/model-info`: Mathematical parameters, AHP weights, CR, model metadata.
  - Retain and wrap `/api/analyze`, `/api/report/save`, `/api/reports`, etc.

- **Phase 3:** Vision Pipeline (PyTorch MobileNetV3-Small / ONNX / Legacy KNN Fallback)
  - Train MobileNetV3-Small on 250 Positive / 250 Negative (stratified 70/15/15).
  - Temperature scaling calibration for validation NLL and ECE reduction.
  - Occlusion sensitivity (5×5 grid) explainability heatmap.
  - Legacy KNN seamless fallback.

- **Phase 4:** Spatial Intelligence, Fusion & Context
  - `backend/context.py`: OpenStreetMap / Overpass facility, building density, and road class query with disk caching and graceful fallbacks.
  - `backend/spatial.py`: Getis-Ord Gi* hotspot analysis and multi-report Bayesian evidence fusion.
  - Extended Open-Meteo: 5-day antecedent precipitation (AMC) + 24h forecast.

- **Phase 5:** Repair Prioritisation & Optimization
  - Budget-constrained greedy selection based on $R / \text{Cost}$.

- **Phase 6:** Frontend Upgrades
  - Admin Dashboard: What-if rainfall slider, risk vs rain dynamic chart, hotspot layer, triage breakdown accordion.
  - User Dashboard: "Explain This Score" card with step-by-step breakdown.
  - Model & Math Page (`frontend/model/index.html`): KaTeX mathematical derivations, proofs, AHP matrices, and calibration plots.

- **Phase 7:** Synthetic Validation & Documentation
  - `scripts/seed_synthetic.py`: Realistic Coimbatore cluster dataset (60 nodes).
  - `scripts/validate_engine.py`: Reliability curves, Brier score, Kendall tau, tornado plots in `docs/figures/`.
  - Comprehensive `README.md`, `docs/DEMO_SCRIPT.md`, and 20 Viva examination Q&As.

