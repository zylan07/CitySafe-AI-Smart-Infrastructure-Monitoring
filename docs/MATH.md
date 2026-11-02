# CitySafe V2 — Formal Mathematical Foundations & Derivations
**Institution:** KPR Institute of Engineering and Technology, Department of CSE  
**Project:** CitySafe — Real-Time Vulnerability Assessment of Unmapped Urban Infrastructure  
**Authors:** Sathya R V (24CS192), Shayden S (24CS195), Sudharsan S M (24CS218), Tajmal Hussain S (25CSL20)

---

## 1. Classification Taxonomy

To ensure rigorous defense during technical examination and viva, every mathematical construct in this platform is strictly partitioned into one of four epistemic categories:

1. **Mathematical Identity / Theorem:** Analytically proven truths that hold strictly under deductive logic and mathematical axioms.
2. **Empirical Literature Result:** Relations validated in published civil engineering or hydrological literature (e.g., USDA NRCS, ASCE).
3. **Modelling Assumption:** Conscious engineering hypotheses chosen to capture physical behavior where municipal empirical records are currently unmapped.
4. **Engineering Design Choice:** Practical algorithmic implementations tailored for real-time edge/browser execution.

---

## 2. Hydrological Model: SCS Curve Number Runoff

### 2.1 Derivation & Equations
**Classification:** *Empirical Literature Result (USDA NRCS NEH-4)*

The Soil Conservation Service (SCS) Curve Number model computes direct surface storm runoff $Q$ from total precipitation $P$:

$$\text{Potential Maximum Retention } S = \frac{25400}{\text{CN}} - 254 \quad (\text{mm})$$

$$\text{Initial Abstraction } I_a = 0.20 \cdot S \quad (\text{mm})$$

$$\text{Direct Runoff } Q = 
\begin{cases} 
0, & P \le I_a \\
\dfrac{(P - I_a)^2}{P - I_a + S}, & P > I_a 
\end{cases}$$

### 2.2 Proof 1: Runoff Monotonicity with Effective Rainfall
**Classification:** *Mathematical Identity / Theorem*

Let effective precipitation be $u = P - I_a$. For all storm events exceeding initial abstraction, $u > 0$ and $S > 0$. The runoff expression becomes:

$$Q(u) = \frac{u^2}{u + S}$$

Differentiating $Q$ with respect to $u$ using the quotient rule:

$$\frac{dQ}{du} = \frac{2u(u + S) - u^2(1)}{(u + S)^2} = \frac{2u^2 + 2uS - u^2}{(u + S)^2} = \frac{u(u + 2S)}{(u + S)^2}$$

Because $u > 0$ and $S > 0$:
- Numerator $u(u + 2S) > 0$
- Denominator $(u + S)^2 > 0$

Therefore:

$$\frac{dQ}{du} > 0 \quad \forall u > 0$$

$$\blacksquare \text{ Runoff } Q \text{ is strictly monotonically increasing with effective rainfall } (P - I_a).$$

---

### 2.3 Proof 2: Runoff Boundedness (Conservation of Mass)
**Classification:** *Mathematical Identity / Theorem*

To prove that runoff can never exceed effective rainfall ($Q < P - I_a$):

$$Q = \frac{u^2}{u + S}$$

Consider the difference $(u - Q)$:

$$u - Q = u - \frac{u^2}{u + S} = \frac{u(u + S) - u^2}{u + S} = \frac{u^2 + uS - u^2}{u + S} = \frac{uS}{u + S}$$

Since $u > 0$ and $S > 0$, the product $uS > 0$ and $(u + S) > 0$. Thus:

$$u - Q > 0 \implies Q < u = P - I_a < P$$

$$\blacksquare \text{ Total runoff volume } Q \text{ is strictly upper-bounded by effective precipitation } (P - I_a).$$

---

### 2.4 Antecedent Moisture Condition (AMC) Adjustments
**Classification:** *Empirical Literature Result (NRCS)*

Soil wetness before a rainfall event significantly shifts runoff potential. 5-day antecedent rainfall partitions conditions into AMC I (dry), AMC II (average), and AMC III (saturated):

$$\text{CN}_{\text{I}} = \frac{4.2 \cdot \text{CN}_{\text{II}}}{10 - 0.058 \cdot \text{CN}_{\text{II}}}$$

$$\text{CN}_{\text{III}} = \frac{23 \cdot \text{CN}_{\text{II}}}{10 + 0.13 \cdot \text{CN}_{\text{II}}}$$

---

## 3. Rainfall Hazard Covariate Normalization

### 3.1 Formulation & Monotonicity
**Classification:** *Modelling Choice*

Surface runoff $Q$ (mm) is mapped into a normalized environmental hazard covariate $x_{\text{rain}} \in [0, 1)$:

$$x_{\text{rain}} = 1 - \exp\left(-\frac{Q}{Q_{\text{ref}}}\right), \quad Q_{\text{ref}} = 100\text{ mm}$$

**Proof of Properties:**
1. **Lower bound:** For $Q = 0$, $x_{\text{rain}} = 1 - e^0 = 0$.
2. **Upper bound:** For any finite $Q \ge 0$, $\exp(-Q/Q_{\text{ref}}) > 0 \implies x_{\text{rain}} < 1$. In numerical computation, IEEE 754 float precision is bounded at $1 - 10^{-12}$.
3. **Monotonicity:** $\dfrac{dx_{\text{rain}}}{dQ} = \dfrac{1}{Q_{\text{ref}}} \exp\left(-\dfrac{Q}{Q_{\text{ref}}}\right) > 0$.

---

## 4. Analytical Hierarchy Process (AHP) & Consistency

### 4.1 Hazard & Consequence Comparison Matrices
**Classification:** *Modelling Choice & Literature (Saaty, 1980)*

Let pairwise comparison matrix $A = (a_{ij})_{n \times n}$ where $a_{ji} = 1 / a_{ij}$ and $a_{ii} = 1$. The priority weight vector $w$ is the principal right eigenvector:

$$A w = \lambda_{\max} w, \quad \sum_{i=1}^n w_i = 1$$

### 4.2 Proof: Perron-Frobenius Bound $\lambda_{\max} \ge n$
**Classification:** *Mathematical Identity / Theorem*

By the Perron-Frobenius theorem for positive irreducible matrices, $\lambda_{\max} \ge n$, with equality $\lambda_{\max} = n$ holding if and only if $A$ is perfectly consistent ($a_{ik} = a_{ij} a_{jk}$ for all $i, j, k$).

The Consistency Index ($\text{CI}$) and Consistency Ratio ($\text{CR}$) are defined as:

$$\text{CI} = \frac{\lambda_{\max} - n}{n - 1}, \quad \text{CR} = \frac{\text{CI}}{\text{RI}(n)}$$

For our system:
- **Hazard Matrix ($n=3$, $\text{RI}=0.58$):** $\lambda_{\max} = 3.0092$, $\text{CI} = 0.0046$, $\mathbf{\text{CR} = 0.0079 < 0.10}$.
- **Consequence Matrix ($n=4$, $\text{RI}=0.90$):** $\lambda_{\max} = 4.0206$, $\text{CI} = 0.0069$, $\mathbf{\text{CR} = 0.0076 < 0.10}$.

Both matrices satisfy Saaty's consistency criterion ($\text{CR} < 0.10$).

---

## 5. Weibull Proportional Hazards Model

### 5.1 Baseline Hazard & Survival Distribution
**Classification:** *Empirical Literature (Cox, 1972; Weibull, 1951)*

Let $T$ be the random variable denoting time to structural failure. The baseline cumulative hazard with characteristic life $\eta$ and shape parameter $k$ is:

$$H_0(t) = \left(\frac{t}{\eta}\right)^k$$

Under Cox proportional hazards with environmental/operational covariates $x = [x_{\text{sev}}, x_{\text{rain}}, x_{\text{traf}}]$ and AHP-derived weights $\beta = \ln(10) \cdot w$:

$$H(t \mid x) = H_0(t) \exp(\beta \cdot x) = \left(\frac{t}{\eta}\right)^k \exp(\beta \cdot x)$$

$$S(t \mid x) = \exp(-H(t \mid x)) = \exp\left(-\left(\frac{t}{\eta}\right)^k \exp(\beta \cdot x)\right)$$

---

### 5.2 Proof 3: Conditional Failure Probability Derivation
**Classification:** *Mathematical Identity / Theorem*

Given that an asset has survived without catastrophic failure up to its current operational age $a$, what is the probability that it fails within the forward horizon $[a, a + \Delta]$?

By Bayes' theorem and conditional probability definition:

$$P(a < T \le a + \Delta \mid T > a) = \frac{P(a < T \le a + \Delta)}{P(T > a)}$$

Since $P(a < T \le a + \Delta) = P(T > a) - P(T > a + \Delta)$:

$$P(a < T \le a + \Delta \mid T > a) = \frac{P(T > a) - P(T > a + \Delta)}{P(T > a)} = 1 - \frac{S(a + \Delta \mid x)}{S(a \mid x)}$$

Substituting the Weibull survival function:

$$\frac{S(a + \Delta \mid x)}{S(a \mid x)} = \frac{\exp\left(-\left(\frac{a + \Delta}{\eta}\right)^k \exp(\beta \cdot x)\right)}{\exp\left(-\left(\frac{a}{\eta}\right)^k \exp(\beta \cdot x)\right)} = \exp\left(-\left[\left(\frac{a + \Delta}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x)\right)$$

Therefore:

$$P_{\text{fail}}(\Delta \mid a, x) = 1 - \exp\left(-\left[\left(\frac{a + \Delta}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x)\right)$$

$$\blacksquare$$

**Why asset age $a$ is indispensable:**  
For $k > 1$ (wear-out degradation), the hazard rate $h(t) = k \frac{t^{k-1}}{\eta^k}$ is strictly increasing with age. Assuming an asset is "ageless" is mathematically equivalent to setting $k = 1$ (exponential distribution), which falsely assumes structural concrete never wears out.

---

### 5.3 Proof 4: Weibull Proportional Hazards Closure Property
**Classification:** *Mathematical Identity / Theorem*

A critical mathematical question is whether the proportional hazards formulation preserves the standard Weibull distribution family:

$$H(t \mid x) = \left(\frac{t}{\eta}\right)^k \exp(\beta \cdot x)$$

Rewrite $\exp(\beta \cdot x)$ as $\left(\exp\left(\frac{\beta \cdot x}{k}\right)\right)^k$:

$$H(t \mid x) = \left(\frac{t}{\eta}\right)^k \left(\exp\left(\frac{\beta \cdot x}{k}\right)\right)^k = \left(\frac{t}{\eta \exp\left(-\frac{\beta \cdot x}{k}\right)}\right)^k$$

Define the covariate-accelerated characteristic lifespan:

$$\eta(x) = \eta \exp\left(-\frac{\beta \cdot x}{k}\right)$$

Then:

$$H(t \mid x) = \left(\frac{t}{\eta(x)}\right)^k$$

$$\blacksquare \text{ The proportional hazards multiplier acts directly as an accelerated life factor scaling characteristic lifespan from } \eta \text{ to } \eta(x).$$

---

### 5.4 Derivation: Horizon $\Delta^*$ to 10% Conditional Failure Probability
**Classification:** *Mathematical Identity / Theorem*

We seek the horizon $\Delta^*$ (years) such that $P_{\text{fail}}(\Delta^* \mid a, x) = p^* = 0.10$:

$$1 - \exp\left(-\left[\left(\frac{a + \Delta^*}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x)\right) = p^*$$

$$\exp\left(-\left[\left(\frac{a + \Delta^*}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x)\right) = 1 - p^*$$

$$-\left[\left(\frac{a + \Delta^*}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k\right] \exp(\beta \cdot x) = \ln(1 - p^*)$$

$$\left(\frac{a + \Delta^*}{\eta}\right)^k - \left(\frac{a}{\eta}\right)^k = \frac{-\ln(1 - p^*)}{\exp(\beta \cdot x)} \equiv \Delta H$$

Multiplying by $\eta^k$:

$$(a + \Delta^*)^k - a^k = \eta^k \Delta H$$

$$(a + \Delta^*)^k = a^k + \eta^k \Delta H$$

$$\Delta^* = \left(a^k + \eta^k \Delta H\right)^{1/k} - a$$

Where $\Delta H = \dfrac{-\ln(0.90)}{\exp(\beta \cdot x)}$.  
In days: $\text{Days}^* = 365.25 \cdot \Delta^*$, capped at 3650 days for UI presentation.

---

## 6. Expected Risk Formulation: $R = P_{\text{fail}} \times C$

### 6.1 Derivation from Statistical Decision Theory
**Classification:** *Mathematical Identity / Modelling Choice*

In classical risk analysis and expected utility theory, risk is defined as the expected value of loss $\mathbb{E}[L]$ associated with an uncertain event $F$ (structural failure):

$$\mathbb{E}[L] = P(F) \cdot \mathbb{E}[L \mid F] + P(\neg F) \cdot \mathbb{E}[L \mid \neg F]$$

Assuming the baseline operational loss under non-failure is normalized to zero ($\mathbb{E}[L \mid \neg F] = 0$), and normalized consequence score $C \in [0, 1]$ serves as the consequence magnitude proxy $\mathbb{E}[L \mid F]$:

$$\mathbb{E}[L] = P_{\text{fail}} \cdot C$$

Thus, the dimensionless Risk Index $R$ is formulated as:

$$R = P_{\text{fail, 90d}} \times C$$

**Conceptual Distinctions:**
- $P_{\text{fail, 90d}}$ is a valid mathematical probability $\in [0, 1]$.
- $C$ is a normalized multi-criteria consequence index $\in [0, 1]$.
- $R$ is a risk index (expected loss magnitude), **not** an empirical frequency.
- The display score is normalized against worst-realistic conditions: $\text{Score} = \min\left(100, 100 \cdot \dfrac{R}{R_{\text{ref}}}\right)$.

---

## 7. Multi-Report Bayesian Evidence Fusion

### 7.1 Likelihood Ratios & Odds Updating
**Classification:** *Mathematical Identity (Bayesian Statistics)*

When multiple independent citizen reports $r_1, r_2, \dots, r_m$ are registered within spatial proximity ($d \le 50\text{ m}$) for the same asset type, each report provides noisy binary observation $Y_i \in \{\text{Damaged}, \text{Safe}\}$.

Given classifier sensitivity $\text{Se} = P(Y = \text{Damaged} \mid \text{True Damaged})$ and specificity $\text{Sp} = P(Y = \text{Safe} \mid \text{True Safe})$:

$$\text{LR}(Y_i = \text{Damaged}) = \frac{\text{Se}}{1 - \text{Sp}}, \quad \text{LR}(Y_i = \text{Safe}) = \frac{1 - \text{Se}}{\text{Sp}}$$

Posterior odds given $m$ conditionally independent reports:

$$\text{Odds}_{\text{post}} = \text{Odds}_{\text{prior}} \cdot \prod_{i=1}^m \text{LR}(Y_i)$$

In log-odds (logit) space:

$$\text{logit}(P_{\text{fused}}) = \text{logit}(\pi_0) + \sum_{i=1}^m \ln\left(\text{LR}(Y_i)\right)$$

$$P_{\text{fused}} = \frac{1}{1 + \exp(-\text{logit}(P_{\text{fused}}))}$$

---

## 8. Temperature Scaling & Class Ranking Preservation

### 8.1 Logit Softmax Calibration
**Classification:** *Mathematical Identity / Literature (Guo et al., 2017)*

For vision model logits $z \in \mathbb{R}^K$, calibrated probability with temperature $T > 0$ is:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

**Proof of Order Preservation:**
For any two classes $j$ and $k$ with $z_j > z_k$, and any $T > 0$:

$$\frac{z_j}{T} > \frac{z_k}{T} \implies \exp\left(\frac{z_j}{T}\right) > \exp\left(\frac{z_k}{T}\right) \implies \hat{p}_j > \hat{p}_k$$

$$\blacksquare \text{ Temperature scaling strictly preserves logit argmax predictions and multiclass ranking while calibrating confidence dispersion.}$$

---

## 9. Spatial Clustering: Getis-Ord $G_i^*$ Hotspot Analysis

### 9.1 Formulation
**Classification:** *Statistical Inference (Ord & Getis, 1995)*

For $n$ spatial observations with risk scores $x_1, \dots, x_n$ and binary spatial weights $w_{ij} \in \{0, 1\}$ (where $w_{ij} = 1$ if $\text{Haversine}(i, j) \le 500\text{ m}$):

$$G_i^* = \frac{\sum_{j=1}^n w_{ij} x_j - \bar{X} \sum_{j=1}^n w_{ij}}{S \sqrt{\dfrac{n \sum_{j=1}^n w_{ij}^2 - \left(\sum_{j=1}^n w_{ij}\right)^2}{n - 1}}}$$

Where:

$$\bar{X} = \frac{1}{n} \sum_{j=1}^n x_j, \quad S = \sqrt{\frac{1}{n} \sum_{j=1}^n x_j^2 - (\bar{X})^2}$$

$G_i^*$ is asymptotically distributed as standard normal $Z \sim \mathcal{N}(0, 1)$ under the null hypothesis of complete spatial randomness.  
Statistical significance:
- $|z| \ge 1.96 \implies p < 0.05$ (95% confidence)
- $|z| \ge 2.58 \implies p < 0.01$ (99% confidence)

---

## 10. Budget-Constrained Repair Prioritization

### 10.1 Fractional Knapsack Relaxation
**Classification:** *Engineering Design Choice / Operations Research*

Given available municipal budget $B_{\text{total}}$ and a set of candidate damaged assets $\{1, \dots, N\}$, each with repair cost $c_i > 0$ and expected risk reduction $R_i$:

$$\max \sum_{i=1}^N R_i y_i \quad \text{subject to} \quad \sum_{i=1}^N c_i y_i \le B_{\text{total}}, \quad y_i \in \{0, 1\}$$

When costs are uniform ($c_i = c$), sorting in descending order of risk $R_i$ yields the optimal integer knapsack solution. When costs vary, ranking assets by the risk-to-cost ratio:

$$\rho_i = \frac{R_i}{c_i}$$

is the optimal policy for the linear fractional knapsack relaxation, providing an effective practical heuristic for municipal maintenance scheduling.

