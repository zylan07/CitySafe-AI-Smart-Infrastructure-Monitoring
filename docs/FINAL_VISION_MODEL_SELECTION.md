# CitySafe / ZylanZest V2 — Final Vision Model Selection Audit

**Authors:** Sathya R V (24CS192), Shayden S (24CS195), Sudharsan S M (24CS218), Tajmal Hussain S (25CSL20)  
**Institution:** KPR Institute of Engineering and Technology, Department of Computer Science and Engineering  
**Date:** November 2026  
**Document Status:** Production Configuration Verification & Epistemic Audit  

---

## 1. Definitive Production Model Status

The production model currently deployed in the CitySafe backend service is:

### **A. Original CNN + Operating Threshold $\theta = 0.23$**

```
Model File Loaded:            backend/model/citysafe_cnn.pt (122,941 bytes)
Fallback File:                backend/model/concrete_knn_model.pkl (1,283,842 bytes)
Threshold Configuration:      backend/model/vision_metrics_optimized.json -> optimal_threshold = 0.23
Service Module:               backend/vision_service.py (Line 59: _operating_threshold = 0.23)
Active Prediction Label:      "Damaged" if damage_probability >= 0.23 else "Safe"
```

The Focal Loss CNN evaluated during Phase 4 of `scripts/optimize_vision.py` was evaluated as an **experimental research candidate**. It has **not** replaced the physical weights file `backend/model/citysafe_cnn.pt` on disk.

---

## 2. Production Model A: Exact Configuration & Metrics

### 2.1 Architecture Specification
- **Model Class:** `CitySafeCNN` (in `backend/train_vision.py`)
- **Structure:** Depthwise-Separable Convolutional Neural Network (MobileNet-style)
  - Layer 1: Standard Conv2d ($3 \to 16$, $3 \times 3$, stride 2, padding 1) + BatchNorm + ReLU
  - Layer 2: Depthwise Conv2d ($16 \to 16$, $3 \times 3$, stride 1, padding 1) + Pointwise Conv2d ($16 \to 32$) + BatchNorm + ReLU
  - Layer 3: Depthwise Conv2d ($32 \to 32$, $3 \times 3$, stride 2, padding 1) + Pointwise Conv2d ($32 \to 64$) + BatchNorm + ReLU
  - Layer 4: Depthwise Conv2d ($64 \to 64$, $3 \times 3$, stride 2, padding 1) + Pointwise Conv2d ($64 \to 128$) + BatchNorm + ReLU
  - Global Average Pooling ($128 \times 1 \times 1$)
  - Linear Classification Head ($128 \to 2$)
- **Total Parameters:** $17,298$ parameters ($69.2\text{ KB}$ raw float32)
- **Input Dimensions:** $224 \times 224 \times 3$ RGB, normalized via ImageNet statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

### 2.2 Training Configuration (Original CNN)
- **Loss Function:** Standard Binary Cross-Entropy / Multi-class Cross-Entropy with uniform label weights:
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
- **Optimizer:** AdamW ($\text{learning rate} = 1 \times 10^{-3}$, $\text{weight decay} = 1 \times 10^{-4}$)
- **Batch Size:** $32$
- **Epochs:** $10$ with early stopping on validation loss ($\text{patience} = 3$)
- **Data Partitions:**
  - Training Set: $N = 350$ ($175$ Damaged, $175$ Sound)
  - Validation Set: $N = 76$ ($38$ Damaged, $38$ Sound)
  - Held-Out Test Set: $N = 74$ ($37$ Damaged, $37$ Sound) — untouched during training

### 2.3 Operating Threshold Selection (Policy B: Max $F_2$)
Because default softmax probability outputs for damaged concrete concentrated between $0.15$ and $0.48$ (mean $\approx 0.30$), the decision boundary was calibrated via a **pure validation sweep**:

- **Sweep Range:** $\theta \in [0.05, 0.95]$ at $\Delta\theta = 0.01$ strictly on $N = 76$ validation predictions.
- **Selection Criterion:** **Policy B (Maximize $F_2$)**
  $$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
  *Reasoning:* Prioritizes recall for civil safety infrastructure. Missing a structural crack (False Negative) has catastrophic failure risk, whereas dispatching an engineer to sound concrete (False Positive) is a minor operational overhead.
- **Validation Metrics at Selected $\theta^* = 0.23$:**
  - Validation Recall: $0.9474$ ($36/38$)
  - Validation Precision: $0.9000$ ($36/40$)
  - Validation $F_1$ Score: $0.9231$
  - Validation $F_2$ Score: **$0.9375$ (Maximum across all $\theta$)**
  - Validation Accuracy: $0.9211$ ($70/76$)

### 2.4 Strict Confirmation of Zero Data Leakage
- **Pre-Registration:** Policy B was defined as the selection criterion **before** evaluating the test set.
- **Single Evaluation:** The test set was accessed **exactly once** after freezing $\theta^* = 0.23$.
- **No Feedback:** No test set metrics were fed back to modify $\theta^*$, network weights, or feature maps.

### 2.5 Production Model A: Untouched Held-Out Test Set Results ($N=74$)
- **Accuracy:** $0.9054$ ($67/74$)
- **Recall (Damaged Class):** $0.9189$ ($34/37$)
- **Precision:** $0.8947$ ($34/38$)
- **$F_1$ Score:** $0.9067$
- **$F_2$ Score:** $0.9140$
- **Specificity:** $0.8919$ ($33/37$)
- **Balanced Accuracy:** $0.9054$
- **ROC-AUC:** $0.9671$
- **PR-AUC:** $0.9745$

#### Exact Test Confusion Matrix (Production Model A @ $\theta = 0.23$)
```
                         Predicted Sound (0)     Predicted Damaged (1)       Total
Actual Sound Concrete            33 (TN)                  4 (FP)               37
Actual Damaged Concrete           3 (FN)                 34 (TP)               37
Total                            36                      38                    74
```

---

## 3. Experimental Candidate Model B: Focal Loss CNN

During the Phase 4 loss experiments, a variant of the MobileNet-style CNN was trained with Focal Loss. Here is its complete audit trail:

### 3.1 Exact Training Configuration
- **Loss Function:** Binary Focal Loss with hard negative mining:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  where $\alpha = 0.65$ (class weighting favoring damaged class) and $\gamma = 2.0$ (focusing parameter down-weighting well-classified background concrete).
- **Optimizer:** AdamW ($\text{lr} = 1 \times 10^{-3}$, $\text{weight decay} = 1 \times 10^{-4}$)
- **Batch Size:** $32$
- **Epochs:** $6$ epochs on $N = 350$ training images with early stopping based on validation PR-AUC ($\text{patience} = 3$).
- **Seed:** Deterministic seed `SEED = 42`.

### 3.2 Validation Metrics
- **Validation PR-AUC:** $1.0000$
- **Validation ROC-AUC:** $1.0000$

### 3.3 Test Set Performance (Experimental Evaluation Only)
- **Threshold:** $\theta = 0.50$ (standard default)
- **Accuracy:** $0.9865$ ($73/74$)
- **Recall (Damaged Class):** $0.9730$ ($36/37$)
- **Precision:** $1.0000$ ($36/36$)
- **$F_1$ Score:** $0.9863$
- **$F_2$ Score:** $0.9783$
- **Specificity:** $1.0000$ ($37/37$)
- **ROC-AUC:** $0.9985$
- **PR-AUC:** $0.9986$

#### Exact Test Confusion Matrix (Experimental Focal Loss CNN @ $\theta = 0.50$)
```
                         Predicted Sound (0)     Predicted Damaged (1)       Total
Actual Sound Concrete            37 (TN)                  0 (FP)               37
Actual Damaged Concrete           1 (FN)                 36 (TP)               37
Total                            38                      36                    74
```

### 3.4 Why Model B is NOT Yet the Production Model
1. **Serialization Discipline:** Model B was trained in-memory within `scripts/optimize_vision.py` to benchmark loss formulation improvements. Its weights were evaluated to prove that loss re-weighting addresses confidence calibration, but were **not written over `backend/model/citysafe_cnn.pt`**.
2. **Production Stability:** `backend/vision_service.py` explicitly loads `citysafe_cnn.pt` (the original trained model).
3. **Audit Rule:** Per strict instructions, no changes or file overwrites are made without explicit user direction.

---

## 4. Production Summary Table

| Attribute | Production Model (Active) | Experimental Candidate |
| :--- | :--- | :--- |
| **Model Name** | **Model A: Original CNN + $\theta = 0.23$** | **Model B: Focal Loss CNN** |
| **Weights File** | `backend/model/citysafe_cnn.pt` | In-memory checkpoint (`scripts/optimize_vision.py`) |
| **Loss Used During Training** | Standard Cross-Entropy | Binary Focal Loss ($\alpha=0.65, \gamma=2.0$) |
| **Operating Threshold** | $\theta^* = 0.23$ (Policy B: Max $F_2$) | $\theta = 0.50$ (Default) |
| **Selection Criterion** | Max Validation $F_2$ score on $N=76$ | Max Validation PR-AUC on $N=76$ |
| **Test Accuracy** | **$0.9054$** | $0.9865$ |
| **Test Damaged Recall** | **$0.9189$** ($34/37$) | $0.9730$ ($36/37$) |
| **Test Precision** | **$0.8947$** ($34/38$) | $1.0000$ ($36/36$) |
| **Test $F_1$ Score** | **$0.9067$** | $0.9863$ |
| **Test $F_2$ Score** | **$0.9140$** | $0.9783$ |
| **Bootstrap 95% Recall CI** | **$[0.8275, \; 1.0000]$** | $[0.9189, \; 1.0000]$ |
| **Production Status** | **LIVE & ACTIVE IN SERVICE** | **BENCHMARKED RESEARCH VARIANT** |

---

## 5. Conclusion & Verification

- The backend service (`backend/vision_service.py`) is verified to be running **Option A: Original CNN + threshold 0.23**.
- The model file loaded is `backend/model/citysafe_cnn.pt`.
- No retraining, file renaming, or weight modification has taken place.
- All test results were produced strictly once on the untouched held-out test partition.

