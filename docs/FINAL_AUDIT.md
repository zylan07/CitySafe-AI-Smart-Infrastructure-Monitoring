# CitySafe / ZylanZest V2 — Final Engineering Audit & Scientific Transparency Report

**Institution:** KPR Institute of Engineering and Technology, Department of Computer Science and Engineering  
**Project:** CitySafe — AI-Powered Framework for Real-Time Vulnerability Assessment of Unmapped Urban Infrastructure  
**Project Stage:** Proto Studio Advanced Upgrade  
**Team Members:**
- **Sathya R V** — 24CS192
- **Shayden S** — 24CS195
- **Sudharsan S M** — 24CS218
- **Tajmal Hussain S** — 25CSL20

---

## 1. Executive Summary & Epistemic Stance

This document provides a rigorous, scientifically honest audit of the CitySafe V2 platform prior to academic evaluation. In accordance with strict engineering ethics:
- No metrics have been invented or altered.
- The platform uses **literature-informed and expert-elicited priors**, validating the mathematical risk pipeline via controlled synthetic simulations.
- **The system has not learned failure rates from real municipal maintenance records**, as historical Coimbatore municipal infrastructure failure logs are unmapped and un-digitized.
- All pseudo-random numbers have been completely purged from the production prediction path. Identical inputs strictly yield bitwise-identical outputs.

---

## 2. Production Vision Architecture & Model Selection Policy

### 2.1 Implemented Vision Architectures
Two vision classifiers are implemented and co-exist in [`backend/vision_service.py`](file:///c:/Users/Admin/OneDrive/Attachments/Documents/Design%20Studio/CitySafe-Dashboard/backend/vision_service.py):
1. **Experimental Deep Neural Network (`CitySafeCNN`):**
   - Architecture: Lightweight MobileNet-inspired Depthwise-Separable Convolutional Neural Network (MobileNetV3-Small style) designed for $224 \times 224 \times 3$ RGB input.
   - Features: Inverted depthwise separable blocks, batch normalization, ReLU6 activations, global average pooling, dropout (0.2), and a linear binary classification head.
   - Post-hoc Calibration: Temperature scaling layer with fitted temperature $T = 0.8695$.
2. **Legacy Baseline (`KNeighborsClassifier`):**
   - Architecture: 7-Nearest Neighbors ($k=7$) trained on flattened $20 \times 20$ grayscale downsampled images (400 features).

---

### 2.2 Actual Measured Test Set Results (Held-Out Test Set, $N = 74$)
Evaluation was performed on the exact same stratified 70/15/15 split (350 train, 76 val, 74 test; 37 Positive, 37 Negative):

| Metric | Experimental MobileNet CNN | Legacy KNN Baseline ($k=7$) | Stronger Classifier |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.6081** (45 / 74) | **0.8108** (60 / 74) | **KNN (+20.27%)** |
| **Recall (Damaged Class)** | **0.2162** (8 / 37) | **0.6216** (23 / 37) | **KNN (+40.54%)** |
| **Precision (Damaged Class)**| **1.0000** (8 / 8) | **1.0000** (23 / 23) | Tie (1.0000) |
| **F1 Score** | **0.3556** | **0.7667** | **KNN (+41.11%)** |
| **ROC-AUC** | **0.9671** | **0.8755** | CNN (+0.0916) |
| **PR-AUC** | **0.9745** | **0.8785** | CNN (+0.0960) |
| **Sensitivity ($\text{Se}$)** | **0.2162** | **0.6216** | **KNN** |
| **Specificity ($\text{Sp}$)** | **1.0000** | **1.0000** | Tie (1.0000) |

---

### 2.3 Transparent Model Verdict & Operational Recommendation
> **Critical Audit Finding:** The experimental MobileNet CNN must **NOT** be presented as superior to the baseline model.
> - While the CNN achieves an impressive continuous ranking capability ($\text{ROC-AUC} = 0.9671, \text{PR-AUC} = 0.9745$), its discrete classification performance under the default $0.50$ probability threshold is severely impaired by a high false-negative rate ($\text{Recall} = 0.2162$). It missed 29 out of 37 damaged concrete test samples.
> - In stark contrast, the **Legacy KNN baseline achieves significantly higher Accuracy (81.08% vs. 60.81%) and substantially higher Recall (62.16% vs. 21.62%)** while maintaining perfect Precision ($1.0000$).
> - **Operational Conclusion:** The current CNN is experimental. For production triage on this 500-image dataset, **KNN remains the stronger and recommended baseline** unless a lower operating decision threshold (e.g., classifying as Damaged if $P > 0.15$) or a significantly larger training corpus is deployed.

---

### 2.4 Temperature Scaling Calibration Reality Check
- **Fitted Temperature:** $T = 0.8695$ (optimized via Negative Log-Likelihood minimization on validation logits).
- **Expected Calibration Error (ECE) Before Scaling:** **0.1315**
- **Expected Calibration Error (ECE) After Scaling:** **0.1406**
- **Scientific Audit Finding:** Temperature scaling optimized validation log-likelihood, but on the held-out test set, ECE slightly shifted from 0.1315 to 0.1406. **We do not claim that temperature scaling improved ECE on the test set.** On small validation sets ($N=76$), temperature scaling can slightly overfit dispersion without improving empirical bin calibration.

---

## 3. Mathematical Risk Engine Verification

### 3.1 Determinism & Repeatability Verification
All stochastic elements and pseudo-random calls have been removed:
- Production risk engine ([`backend/risk_engine.py`](file:///c:/Users/Admin/OneDrive/Attachments/Documents/Design%20Studio/CitySafe-Dashboard/backend/risk_engine.py)) has zero unseeded randomness.
- Monte Carlo uncertainty ($N=500$) uses a deterministic integer seed derived from `hashlib.md5(report_id.encode()).hexdigest()`.
- Determinism test verified: calling the engine or `/api/v2/simulate` repeatedly with identical parameters produces bitwise-identical outputs (`res1 == res2: PASSED`).

### 3.2 Hydrological Properties Verified
- **Runoff Monotonicity:** $\dfrac{dQ}{du} = \dfrac{u(u + 2S)}{(u + S)^2} > 0$ for all $u = P - I_a > 0$ (Automated property test: **PASSED**).
- **Runoff Mass Conservation:** $u - Q = \dfrac{uS}{u + S} > 0 \implies Q < P - I_a < P$ (Automated property test: **PASSED**).
- **Rainfall Hazard Covariate:** $x_{\text{rain}} = 1 - \exp(-Q / 100) \in [0, 1)$, bounded against float overflow (Automated property test: **PASSED**).

### 3.3 AHP Consistency Ratios
- **Hazard Comparison Matrix ($n=3$, $\text{RI}=0.58$):** $\lambda_{\max} = 3.0092, \text{CI} = 0.0046, \mathbf{\text{CR} = 0.0079 < 0.10}$.
- **Consequence Comparison Matrix ($n=4$, $\text{RI}=0.90$):** $\lambda_{\max} = 4.0206, \text{CI} = 0.0069, \mathbf{\text{CR} = 0.0076 < 0.10}$.
- Both matrices satisfy Saaty's transitivity inequality ($\text{CR} < 0.10$).

---

## 4. Synthetic Simulation Validation (Controlled Synthetic Ground Truth)

> **Mandatory Disclaimer:** The metrics below were evaluated on **1,000 synthetic simulation assets** with known Weibull PH failure times generated via inverse CDF sampling. They demonstrate mathematical recovery of ground truth and are **not fitted to real municipal records**.

- **Brier Score Loss:** **0.0156** (indicates high probability calibration against the underlying synthetic process).
- **Kendall's $\tau$ Ranking Stability:** **0.2432** (maintains relative triage order under noise).
- **Top-20% Risk Recall:** **37.5%** (The top 20% highest-ranked assets captured 6 out of 16 actual synthetic failure events).

---

## 5. Inventory of Modeling Assumptions ([`backend/config.py`](file:///c:/Users/Admin/OneDrive/Attachments/Documents/Design%20Studio/CitySafe-Dashboard/backend/config.py))

| Parameter / Model | Configured Value | Epistemic Classification | Rationale & Justification |
| :--- | :--- | :--- | :--- |
| **Weibull Lifespan $\eta$** | Culvert: 45y, Drain: 30y, Wall: 40y, Path: 20y, Road: 15y | *Expert Prior* | Civil asset management guidelines (ASCE/IRC); must be fitted via MLE on municipal work orders. |
| **Weibull Shape $k$** | $2.0$ ($1.1 \le k \le 3.5$) | *Expert Prior* | Structural wear-out degradation ($k > 1$). $k=1$ would falsely assume concrete never wears out. |
| **Initial Abstraction $I_a$**| $0.20 \cdot S$ | *Empirical Literature* | USDA NRCS National Engineering Handbook (NEH-4). |
| **Rainfall Normalization $Q_{\text{ref}}$**| $100.0\text{ mm}$ | *Modelling Choice* | Calibrated against severe urban storm runoff (~120–150 mm precipitation). |
| **Hazard Scaling $B$** | $\ln(10) \approx 2.3026$ | *Modelling Choice* | Bounds maximum combined environmental covariate multiplier to $10\times$. |
| **Worst-Case Reference $R_{\text{ref}}$**| Evaluated at $x=[1,1,1], a=0.8\eta, C=1.0$ | *Modelling Choice* | Calibrates display score onto $[0, 100]$ using model physics rather than arbitrary magic numbers. |
| **Monte Carlo Variance** | $\sigma_{\eta} = 0.25, \sigma_k = 0.20, \sigma_{\beta} = 0.20$ | *Modelling Choice* | Literature-derived perturbation bands for epistemic uncertainty quantification ($N=500$). |
| **Getis-Ord $G_i^*$ Band** | $d = 500\text{ m}, n \ge 10$ | *Statistical Inference* | Ord & Getis (1995) binary distance band formulation. |

---

## 6. Known System Weaknesses & Failure Modes

1. **Epistemic Data Gap (Lack of Municipal Work Orders):**
   The single biggest limitation of the current system is the absence of real municipal failure records. Parameters $\eta$ and $k$ are literature priors. While the mathematical pipeline is proven and invariant, municipal field adoption will require fitting these parameters using municipal historical work orders.
2. **CNN False Negative Rate on Small Dataset:**
   With only 350 training images, the MobileNet CNN learned very conservative decision boundaries ($\text{Precision} = 1.0$, but $\text{Recall} = 0.2162$). For life-safety inspection, false negatives are hazardous. Hence, the legacy KNN ($62.16\%$ recall) must remain active as the primary or ensemble fallback.
3. **Optical Surface Boundary:**
   Computer vision can only assess visible surface fissures, spalling, and cracking. Sub-surface void formation, internal rebar corrosion, and subterranean culvert blockages cannot be detected through optical imagery alone.
4. **Third-Party API Dependency:**
   Live weather depends on Open-Meteo, and live facility exposure queries depend on OpenStreetMap Overpass. While local disk caching and neutral fallbacks prevent system crashes during outages, live telemetry can introduce 1–3 seconds of latency.
5. **Getis-Ord Minimum Sample Constraint:**
   Spatial hotspot analysis requires at least $n \ge 10$ reports within the local region to satisfy asymptotic normal validity. When fewer than 10 reports exist, the system safely refuses to output mock hotspots.

---

## 7. Verification Checklist

- [x] All 4 student team members listed.
- [x] All AHP reporting states $\text{CR} < 0.10$ (no $\text{RI} < 0.10$).
- [x] Vision architecture explicitly identified as MobileNet-inspired depthwise-separable CNN (MobileNetV3-Small style).
- [x] CNN is NOT presented as superior to KNN; actual measured metrics are preserved.
- [x] KNN explicitly documented as the stronger baseline on the 500-image dataset.
- [x] Calibration discussion accurately states test ECE changed from 0.1315 to 0.1406.
- [x] Brier, Kendall, and top-20% recall explicitly labeled as synthetic simulation validation.
- [x] Verified zero pseudo-random numbers in the production risk calculation.
- [x] Verified deterministic identical input-output behavior.
- [x] Automated test suite passing (26/26 tests passed).

