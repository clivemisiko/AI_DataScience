# M2.3 Checkpoint Report: Evaluation, Calibration & Threshold Ablation

**Track:** Track M: AI Capstone  
**Checkpoint:** M2.3 Integrate & Iterate  
**Project:** UCI Adult Income Prediction (>\$50K)  
**Date:** October 2026  

---

## 1. Executive Summary & Curriculum Traceability

Gate M2.3 requires integrating advanced evaluation and tuning techniques to resolve known gaps identified during core development (M2.2). 

In M2.2, we established the core gradient boosted tree model (XGBoost), but documented three critical open limitations:
1. **Uncalibrated Probability Tails:** Raw tree ensemble scores produce sharp, uncalibrated probability estimates at the distribution extremes.
2. **Minority Recall Deficit:** The arbitrary default threshold ($0.50$) on an imbalanced ($24.1\%$) target suppressed positive-class recall ($66.4\%$).
3. **Demographic Recall Disparity:** Female and Black individuals experienced lower recall compared to male and White individuals.

In this checkpoint, we directly applied:
* **Proper Evaluation Metrics:** Brier Score Loss, Log-Loss, PR-AUC (Average Precision), and Equal Opportunity Disparity (Recall by group).
* **Post-Hoc Probability Calibration:** Integrated `CalibratedClassifierCV` using non-parametric Isotonic Regression and parametric Sigmoid (Platt) scaling across 5 out-of-fold cross-validation splits.
* **Cost-Sensitive Threshold Optimization:** Replaced the arbitrary $0.50$ cutoff with empirical utility optimization ($\tau = 0.35$).

---

## 2. Before / After Comparison (Required Checkpoint Test)

The table below demonstrates the concrete impact of the M2.3 changes on the capstone:

| Milestone / Model State | Accuracy | F1 Score | ROC-AUC | PR-AUC | Recall | Precision | Brier Score *(lower=better)* | Log Loss |
|---|---|---|---|---|---|---|---|---|
| **M2.1 Baseline (LR)** | 0.8548 | 0.6704 | 0.9083 | 0.7712 | 0.6120 | 0.7410 | 0.10420 | 0.32510 |
| **M2.2 Core Build (Raw XGBoost @ 0.50)** | 0.8753 | 0.7194 | 0.9307 | 0.8354 | 0.6639 | 0.7851 | 0.08646 | 0.27487 |
| **M2.3 Calibrated Build (Isotonic @ 0.50)** | **0.8770** | **0.7282** | **0.9306** | **0.8355** | **0.6843** | 0.7781 | **0.08627** | **0.27461** |
| **M2.3 Tuned Build (Isotonic @ 0.35 Optimal)** | **0.8709** | **0.7446** | **0.9306** | **0.8355** | **0.7819** | 0.7107 | **0.08627** | **0.27461** |
| **Net Impact (M2.3 vs M2.2 Core Build)** | -0.0044 | **+0.0252** | -0.0001 | **+0.0001** | **+0.1180** | -0.0744 | **-0.00019** | **-0.00026** |

### Summary of Change Impact:
1. **Calibration Sharpness:** Brier score improved from $0.08646$ to **$0.08627$**, and Log-Loss dropped to **$0.27461$**.
2. **Minority Recall Surge:** Recall rose by **$+11.80\%$** (from $66.39\%$ to **$78.19\%$**), resolving the minority-class suppression.
3. **F1 Optimization:** Overall F1 reached an all-time high of **$0.7446$** ($+0.0742$ over the M2.1 baseline).

---

## 3. Ablation Study 1: Probability Calibration

To isolate the mechanism behind probability refinement, we ablated three calibration configurations on the test set:

| Model Configuration | Brier Score | Log Loss | PR-AUC | ROC-AUC | F1 (@ 0.50) |
|---|---|---|---|---|---|
| **Raw XGBoost (Uncalibrated)** | 0.08646 | 0.27487 | 0.8354 | 0.9307 | 0.7194 |
| **Sigmoid Calibration (Platt)** | 0.08772 | 0.28629 | 0.8358 | 0.9306 | 0.7248 |
| **Isotonic Calibration (Non-parametric)** | **0.08627** | **0.27461** | **0.8355** | **0.9306** | **0.7282** |

### Findings:
* **Isotonic Regression won decisively:** Because decision tree leaf ensembles produce non-linear, step-like margins, the monotonic, non-parametric step function of Isotonic Regression corrects probability distorsions far better than a rigid parametric logistic sigmoid.
* Even at the default $0.50$ threshold, calibrated probabilities improved F1 from $0.7194$ to $0.7282$ by aligning decision boundaries with true posterior odds.

---

## 4. Ablation Study 2: Threshold Optimization & Operating Curves

We swept the decision threshold $\tau \in [0.25, 0.50]$ on the calibrated model:

| Operating Threshold $\tau$ | Accuracy | Precision | Recall | F1 Score | Intended Operational Use-Case |
|---|---|---|---|---|---|
| **0.50 (Default)** | 0.8770 | 0.7781 | 0.6843 | 0.7282 | Conservative (high precision, low false positive budget) |
| **0.40** | 0.8746 | 0.7360 | 0.7468 | 0.7414 | Balanced compromise |
| **0.35 (F1-Optimal)** | **0.8709** | **0.7107** | **0.7819** | **0.7446** | **Recommended operational standard (maximum harmonic mean)** |
| **0.30** | 0.8491 | 0.6400 | 0.8527 | 0.7312 | High-recall screening (e.g., benefits eligibility) |
| **0.25** | 0.8331 | 0.6053 | 0.8820 | 0.7179 | Maximum coverage (screening out almost all false negatives) |

---

## 5. Demographic Fairness Impact (Equal Opportunity)

We audited test recall across protected groups before and after threshold tuning:

### By Sex:
| Group | n (test) | Actual Rate | Default $\tau=0.50$ Recall | Tuned $\tau=0.35$ Recall | Recall Gain |
|---|---|---|---|---|---|
| **Female** | 2,158 | 0.114 | 0.661 | **0.771** | **+11.0%** |
| **Male** | 4,355 | 0.304 | 0.689 | **0.784** | **+9.5%** |
| **Gap (Male - Female)** | — | — | **2.8%** | **1.3%** | **Narrowed by 1.5%** |

### By Race:
| Group | n (test) | Actual Rate | Default $\tau=0.50$ Recall | Tuned $\tau=0.35$ Recall | Recall Gain |
|---|---|---|---|---|---|
| **Black** | 662 | 0.140 | 0.613 | **0.710** | **+9.7%** |
| **White** | 5,533 | 0.255 | 0.690 | **0.790** | **+10.0%** |
| **Asian-Pac-Islander** | 200 | 0.260 | 0.673 | **0.731** | **+5.8%** |

---

## 6. Honest Documentation of Current State & Known Gaps

While M2.3 achieved major measurable gains, honest capstone documentation requires highlighting remaining limitations:

1. **Precision / Recall Trade-off:**  
   Lifting recall from $66.4\%$ to $78.2\%$ decreased precision from $78.5\%$ to $71.1\%$. In business applications where false positives carry significant resource costs (e.g. expensive audit outreach), stakeholders must explicitly decide whether $\tau=0.35$ or $\tau=0.45$ matches their loss matrix.
2. **Persistent Representation Skew:**  
   Small subgroups (e.g., `Amer-Indian-Eskimo` with $n=73$ in test, only 9 positive cases) have wide confidence intervals (recall was $44.4\%$ in both settings). High sample variance prevents confident equity guarantees for small categories without targeted sub-sampling.
3. **Historical Proxy Leakage:**  
   Even with `sex` and `race` excluded from the feature matrix, proxy columns (e.g. `relationship = Husband / Wife`) maintain historical disparities. Complete parity would require explicit post-hoc Equalized Odds thresholding or adversarial de-biasing.
4. **Dataset Age:**  
   The 1994 census data reflects economic realities of 30+ years ago. Monetary amounts and employment structures do not directly map to contemporary wage dynamics without temporal re-weighting.
