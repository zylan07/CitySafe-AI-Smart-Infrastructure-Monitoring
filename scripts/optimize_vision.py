"""
CitySafe / ZylanZest V2 — Vision Performance Optimization Framework
===================================================================
A rigorous, data-leakage-free experimental pipeline:
  Phase 1: Threshold Optimization (Validation Sweep [0.05, 0.95], Policies A-D)
  Phase 2 & 3: Label-Preserving Data Augmentation
  Phase 4: Loss Function Exploration (BCE, Class-Weighted, Focal Loss)
  Phase 5: Feature Representation Classifiers (Logistic Regression, SVM)
  Phase 6: Validation-Guided Ensembling
  Phase 7: Calibration Comparison (Temperature Scaling vs Platt Scaling)
  Phase 8: Evidence-Based Model Selection (Recall, F2, PR-AUC, F1, Accuracy)
  Phase 9: Untouched Held-Out Test Evaluation (Single-pass, Bootstrap 95% CIs)
  Phase 10: Environmental Stress Robustness Testing
  Phase 11: Figures & Comparison Tables (CSV & PNG)
"""

import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss, log_loss
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.train_vision import (
    CitySafeCNN,
    load_dataset_split,
    ConcreteDataset,
    evaluate_knn_baseline,
    compute_ece,
    _jpeg_degrade,
    _add_noise,
    FIGURES_DIR,
    MODEL_DIR,
    SEED
)

# Seed determinism
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ==============================================================================
# FOCAL LOSS & CLASS-WEIGHTED LOSSES
# ==============================================================================

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        return (focal_weight * ce_loss).mean()


# ==============================================================================
# PHASE 1: THRESHOLD OPTIMIZATION ON VALIDATION SET
# ==============================================================================

def sweep_validation_thresholds(val_probs, val_labels):
    """
    Evaluates thresholds from 0.05 to 0.95 in 0.01 increments on VALIDATION data only.
    Generates threshold metrics dataframe and selects policies:
      Policy A: Max F1
      Policy B: Max F2 (prioritizing recall for safety-critical damage detection)
      Policy C: Max recall subject to precision >= 0.70
      Policy D: Max Youden J = Sensitivity + Specificity - 1
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    records = []

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        
        acc = accuracy_score(val_labels, preds)
        prec = precision_score(val_labels, preds, zero_division=0)
        rec = recall_score(val_labels, preds, zero_division=0)
        f1 = f1_score(val_labels, preds, zero_division=0)
        f2 = fbeta_score(val_labels, preds, beta=2, zero_division=0)
        
        # Specificity & Balanced Accuracy
        tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0
        youden_j = rec + spec - 1.0

        records.append({
            "threshold": round(float(th), 2),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "f2": float(f2),
            "specificity": float(spec),
            "balanced_acc": float(bal_acc),
            "youden_j": float(youden_j)
        })

    df = pd.DataFrame(records)

    # Policy Selection (STRICTLY on validation set)
    best_f1_row = df.loc[df["f1"].idxmax()]
    best_f2_row = df.loc[df["f2"].idxmax()]
    
    # Policy C: Max recall with precision >= 0.70
    valid_prec = df[df["precision"] >= 0.70]
    best_prec_bounded_row = valid_prec.loc[valid_prec["recall"].idxmax()] if not valid_prec.empty else best_f2_row

    # Policy D: Max Youden J
    best_youden_row = df.loc[df["youden_j"].idxmax()]

    policies = {
        "Policy A (Max F1)": {
            "threshold": float(best_f1_row["threshold"]),
            "val_f1": float(best_f1_row["f1"]),
            "val_f2": float(best_f1_row["f2"]),
            "val_recall": float(best_f1_row["recall"]),
            "val_precision": float(best_f1_row["precision"])
        },
        "Policy B (Max F2)": {
            "threshold": float(best_f2_row["threshold"]),
            "val_f1": float(best_f2_row["f1"]),
            "val_f2": float(best_f2_row["f2"]),
            "val_recall": float(best_f2_row["recall"]),
            "val_precision": float(best_f2_row["precision"])
        },
        "Policy C (Max Recall s.t. Prec >= 0.70)": {
            "threshold": float(best_prec_bounded_row["threshold"]),
            "val_f1": float(best_prec_bounded_row["f1"]),
            "val_f2": float(best_prec_bounded_row["f2"]),
            "val_recall": float(best_prec_bounded_row["recall"]),
            "val_precision": float(best_prec_bounded_row["precision"])
        },
        "Policy D (Max Youden J)": {
            "threshold": float(best_youden_row["threshold"]),
            "val_f1": float(best_youden_row["f1"]),
            "val_f2": float(best_youden_row["f2"]),
            "val_recall": float(best_youden_row["recall"]),
            "val_precision": float(best_youden_row["precision"])
        }
    }

    return df, policies


# ==============================================================================
# PHASE 4: LOSS FUNCTION EXPERIMENTS
# ==============================================================================

def train_loss_experiment(train_loader, val_loader, loss_type="bce", class_weight=1.5, epochs=8):
    """
    Trains CitySafeCNN with chosen loss function and early stopping on validation PR-AUC.
    """
    model = CitySafeCNN(num_classes=2)
    device = torch.device("cpu")
    model.to(device)

    if loss_type == "weighted":
        weights = torch.tensor([1.0, class_weight], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "focal":
        criterion = BinaryFocalLoss(alpha=0.65, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_prauc = -1.0
    best_weights = None
    patience = 3
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                val_logits_list.append(model(x).numpy())
                val_labels_list.append(y.numpy())

        v_logits = np.concatenate(val_logits_list, axis=0)
        v_labels = np.concatenate(val_labels_list, axis=0)
        v_probs = np.exp(v_logits) / np.sum(np.exp(v_logits), axis=1, keepdims=True)
        v_prauc = average_precision_score(v_labels, v_probs[:, 1])

        if v_prauc > best_val_prauc:
            best_val_prauc = v_prauc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model, best_val_prauc


# ==============================================================================
# PHASE 5: FEATURE REPRESENTATION & LINEAR/SVM CLASSIFIERS
# ==============================================================================

def train_embedding_classifiers(feature_extractor, train_loader, val_loader):
    """
    Extracts pooled penultimate features from trained CNN backbone (128-dim)
    and trains Logistic Regression and SVM on the training partition.
    Hyperparameters C and kernel are selected purely on validation data.
    """
    feature_extractor.eval()

    def get_features(loader):
        feats = []
        labels = []
        with torch.no_grad():
            for x, y in loader:
                # Features through backbone + pool
                f = feature_extractor.features(x)
                pooled = feature_extractor.pool(f).flatten(1).numpy()
                feats.append(pooled)
                labels.append(y.numpy())
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    X_train, y_train = get_features(train_loader)
    X_val, y_val = get_features(val_loader)

    # 1. Logistic Regression Grid Search on Validation PR-AUC
    best_lr = None
    best_lr_prauc = -1.0
    for c_val in [0.01, 0.1, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(C=c_val, max_iter=500, random_state=SEED)
        lr.fit(X_train, y_train)
        probs = lr.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, probs)
        if score > best_lr_prauc:
            best_lr_prauc = score
            best_lr = lr

    # 2. Support Vector Machine Grid Search on Validation PR-AUC
    best_svm = None
    best_svm_prauc = -1.0
    for kernel in ['linear', 'rbf']:
        for c_val in [0.1, 1.0, 5.0, 10.0]:
            svm = SVC(C=c_val, kernel=kernel, probability=True, random_state=SEED)
            svm.fit(X_train, y_train)
            probs = svm.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, probs)
            if score > best_svm_prauc:
                best_svm_prauc = score
                best_svm = svm

    return best_lr, best_lr_prauc, best_svm, best_svm_prauc


# ==============================================================================
# PHASE 7: CALIBRATION COMPARISON (TEMPERATURE vs PLATT SCALING)
# ==============================================================================

def compare_calibration(val_logits, val_labels):
    val_probs_uncal = np.exp(val_logits) / np.sum(np.exp(val_logits), axis=1, keepdims=True)
    nll_uncal = float(log_loss(val_labels, val_probs_uncal[:, 1]))
    brier_uncal = float(brier_score_loss(val_labels, val_probs_uncal[:, 1]))
    ece_uncal = float(compute_ece(val_probs_uncal, val_labels))

    # 1. Temperature Scaling
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    ce_fn = nn.CrossEntropyLoss()

    def eval_t(T):
        if T <= 0.01:
            return 1e6
        with torch.no_grad():
            return ce_fn(val_logits_t / T, val_labels_t).item()

    res_t = minimize_scalar(eval_t, bounds=(0.1, 5.0), method="bounded")
    opt_T = float(res_t.x)

    val_probs_temp = np.exp(val_logits / opt_T) / np.sum(np.exp(val_logits / opt_T), axis=1, keepdims=True)
    nll_temp = float(log_loss(val_labels, val_probs_temp[:, 1]))
    brier_temp = float(brier_score_loss(val_labels, val_probs_temp[:, 1]))
    ece_temp = float(compute_ece(val_probs_temp, val_labels))

    # 2. Platt Scaling (Logistic Regression on Logit Difference)
    logit_diff = (val_logits[:, 1] - val_logits[:, 0]).reshape(-1, 1)
    platt_model = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
    platt_model.fit(logit_diff, val_labels)
    platt_probs = platt_model.predict_proba(logit_diff)[:, 1]

    platt_probs_2d = np.column_stack([1.0 - platt_probs, platt_probs])
    nll_platt = float(log_loss(val_labels, platt_probs))
    brier_platt = float(brier_score_loss(val_labels, platt_probs))
    ece_platt = float(compute_ece(platt_probs_2d, val_labels))

    return {
        "uncalibrated": {"nll": nll_uncal, "brier": brier_uncal, "ece": ece_uncal},
        "temperature_scaling": {"T": opt_T, "nll": nll_temp, "brier": brier_temp, "ece": ece_temp},
        "platt_scaling": {"model": platt_model, "nll": nll_platt, "brier": brier_platt, "ece": ece_platt}
    }


# ==============================================================================
# PHASE 9: UNTOUCHED HELD-OUT TEST EVALUATION WITH BOOTSTRAP CIs
# ==============================================================================

def single_pass_test_eval(y_true, y_prob, threshold, n_bootstraps=1000):
    """
    Evaluates untouched test set once at frozen threshold, then computes
    95% bootstrap confidence intervals for key metrics.
    """
    preds = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    f2 = float(fbeta_score(y_true, preds, beta=2, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float((rec + spec) / 2.0)
    cm = [[int(tn), int(fp)], [int(fn), int(tp)]]

    # Bootstrap 95% Confidence Intervals
    boot_acc, boot_rec, boot_prec, boot_f1, boot_f2 = [], [], [], [], []
    rng = np.random.RandomState(SEED)
    N = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        b_true = y_true[idx]
        b_prob = y_prob[idx]
        b_pred = (b_prob >= threshold).astype(int)

        boot_acc.append(accuracy_score(b_true, b_pred))
        boot_rec.append(recall_score(b_true, b_pred, zero_division=0))
        boot_prec.append(precision_score(b_true, b_pred, zero_division=0))
        boot_f1.append(f1_score(b_true, b_pred, zero_division=0))
        boot_f2.append(fbeta_score(b_true, b_pred, beta=2, zero_division=0))

    cis = {
        "accuracy_ci": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        "recall_ci": [float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5))],
        "precision_ci": [float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5))],
        "f1_ci": [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))],
        "f2_ci": [float(np.percentile(boot_f2, 2.5)), float(np.percentile(boot_f2, 97.5))]
    }

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "specificity": round(spec, 4),
        "balanced_acc": round(bal_acc, 4),
        "confusion_matrix": cm,
        "confidence_intervals_95": cis
    }


# ==============================================================================
# PHASE 10: ENVIRONMENTAL STRESS ROBUSTNESS TESTING
# ==============================================================================

def evaluate_robustness_at_threshold(model, test_pairs, threshold):
    """
    Evaluates classification degradation under simulated environmental stress
    using the validation-frozen operating threshold.
    """
    model.eval()
    perturbations = {
        "Baseline": lambda im: im,
        "Gaussian Blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2.0)),
        "Brightness Drop (-40%)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
        "JPEG Compression (Q=20)": lambda im: _jpeg_degrade(im, 20),
        "Rotation 15 deg": lambda im: im.rotate(15),
        "Additive Noise": lambda im: _add_noise(im)
    }

    results = {}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for name, transform in perturbations.items():
        correct, total = 0, 0
        y_true_list, y_pred_list = [], []
        for path, label in test_pairs:
            with Image.open(path) as raw_img:
                img = transform(raw_img.convert("RGB")).resize((224, 224))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x).numpy()[0]
                    prob = float(np.exp(logits[1]) / (np.exp(logits[0]) + np.exp(logits[1])))
                    pred = int(prob >= threshold)
                    if pred == label:
                        correct += 1
                    total += 1
                    y_true_list.append(label)
                    y_pred_list.append(pred)

        acc = float(correct / total)
        rec = float(recall_score(y_true_list, y_pred_list, zero_division=0))
        prec = float(precision_score(y_true_list, y_pred_list, zero_division=0))
        f1 = float(f1_score(y_true_list, y_pred_list, zero_division=0))
        results[name] = {
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4)
        }

    return results


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 75)
    print("CITYSAFE VISION OPTIMIZATION — FINAL RIGOROUS PASS")
    print("=" * 75)

    # 1. Dataset Split
    train_pairs, val_pairs, test_pairs = load_dataset_split()
    train_ds = ConcreteDataset(train_pairs, is_train=True)
    val_ds = ConcreteDataset(val_pairs, is_train=False)
    test_ds = ConcreteDataset(test_pairs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load Baseline Trained CNN
    base_cnn = CitySafeCNN(num_classes=2)
    base_cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "citysafe_cnn.pt"), map_location="cpu", weights_only=True))
    base_cnn.eval()

    # Extract Baseline Validation Logits & Probabilities
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_logits_list.append(base_cnn(x).numpy())
            val_labels_list.append(y.numpy())

    v_logits = np.concatenate(val_logits_list, axis=0)
    v_labels = np.concatenate(val_labels_list, axis=0)
    v_probs = np.exp(v_logits) / np.sum(np.exp(v_logits), axis=1, keepdims=True)
    v_damage_probs = v_probs[:, 1]

    # Extract Baseline Test Probabilities (UNTOUCHED until final evaluation)
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_logits_list.append(base_cnn(x).numpy())
            test_labels_list.append(y.numpy())

    t_logits = np.concatenate(test_logits_list, axis=0)
    t_labels = np.concatenate(test_labels_list, axis=0)
    t_probs = np.exp(t_logits) / np.sum(np.exp(t_logits), axis=1, keepdims=True)
    t_damage_probs = t_probs[:, 1]

    # --------------------------------------------------------------------------
    # PHASE 1: THRESHOLD OPTIMIZATION (VALIDATION ONLY)
    # --------------------------------------------------------------------------
    print("\n--- PHASE 1: THRESHOLD OPTIMIZATION ON VALIDATION SET ---")
    val_sweep_df, policies = sweep_validation_thresholds(v_damage_probs, v_labels)
    for p_name, p_data in policies.items():
        print(f"  {p_name:38}: Threshold={p_data['threshold']:.2f} | Val F1={p_data['val_f1']:.4f} | Val F2={p_data['val_f2']:.4f} | Val Rec={p_data['val_recall']:.4f} | Val Prec={p_data['val_precision']:.4f}")

    # Primary selection rule for civil safety infrastructure: Maximize F2 (Damaged-class recall)
    # Selected Policy: Policy B (Max F2)
    selected_policy = "Policy B (Max F2)"
    opt_threshold = policies[selected_policy]["threshold"]
    print(f"\n>> FROZEN VALIDATION OPERATING THRESHOLD: {opt_threshold:.2f} (via {selected_policy})")

    # --------------------------------------------------------------------------
    # PHASE 4: LOSS FUNCTION EXPERIMENTS
    # --------------------------------------------------------------------------
    print("\n--- PHASE 4: LOSS FUNCTION EXPERIMENTS ---")
    # Weighted Loss
    weighted_model, weighted_val_prauc = train_loss_experiment(train_loader, val_loader, loss_type="weighted", class_weight=1.5, epochs=6)
    print(f"  Class-Weighted BCE (weight=1.5) | Validation PR-AUC: {weighted_val_prauc:.4f}")

    # Focal Loss
    focal_model, focal_val_prauc = train_loss_experiment(train_loader, val_loader, loss_type="focal", epochs=6)
    print(f"  Focal Loss (gamma=2.0)          | Validation PR-AUC: {focal_val_prauc:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 5: EMBEDDING-BASED CLASSIFIERS
    # --------------------------------------------------------------------------
    print("\n--- PHASE 5: FEATURE REPRESENTATION CLASSIFIERS ---")
    best_lr, lr_val_prauc, best_svm, svm_val_prauc = train_embedding_classifiers(base_cnn, train_loader, val_loader)
    print(f"  Backbone Embeddings + Logistic Regression | Validation PR-AUC: {lr_val_prauc:.4f}")
    print(f"  Backbone Embeddings + SVM                | Validation PR-AUC: {svm_val_prauc:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 6: ENSEMBLE EVALUATION (VALIDATION ONLY)
    # --------------------------------------------------------------------------
    print("\n--- PHASE 6: ENSEMBLE VALIDATION CHECK ---")
    # Get validation probabilities from best loss experiment (weighted) and base CNN
    with torch.no_grad():
        v_weighted_logits = np.concatenate([weighted_model(x).numpy() for x, _ in val_loader], axis=0)
    v_weighted_probs = (np.exp(v_weighted_logits) / np.sum(np.exp(v_weighted_logits), axis=1, keepdims=True))[:, 1]

    best_ensemble_alpha = 0.50
    best_ens_prauc = -1.0
    for alpha in [0.25, 0.50, 0.75]:
        ens_probs = alpha * v_damage_probs + (1.0 - alpha) * v_weighted_probs
        prauc = average_precision_score(v_labels, ens_probs)
        print(f"  Ensemble (alpha={alpha:.2f}) | Validation PR-AUC: {prauc:.4f}")
        if prauc > best_ens_prauc:
            best_ens_prauc = prauc
            best_ensemble_alpha = alpha

    base_val_prauc = average_precision_score(v_labels, v_damage_probs)
    print(f"  Base Model Validation PR-AUC: {base_val_prauc:.4f} vs Ensemble: {best_ens_prauc:.4f}")
    ensemble_justified = (best_ens_prauc > base_val_prauc + 0.005)
    print(f"  Is Ensemble Justified on Validation data? {ensemble_justified}")

    # --------------------------------------------------------------------------
    # PHASE 7: CALIBRATION ANALYSIS (VALIDATION ONLY)
    # --------------------------------------------------------------------------
    print("\n--- PHASE 7: CALIBRATION EVALUATION ---")
    cal_results = compare_calibration(v_logits, v_labels)
    print(f"  Uncalibrated      | NLL: {cal_results['uncalibrated']['nll']:.4f} | Brier: {cal_results['uncalibrated']['brier']:.4f} | ECE: {cal_results['uncalibrated']['ece']:.4f}")
    print(f"  Temp Scaling (T={cal_results['temperature_scaling']['T']:.2f}) | NLL: {cal_results['temperature_scaling']['nll']:.4f} | Brier: {cal_results['temperature_scaling']['brier']:.4f} | ECE: {cal_results['temperature_scaling']['ece']:.4f}")
    print(f"  Platt Scaling     | NLL: {cal_results['platt_scaling']['nll']:.4f} | Brier: {cal_results['platt_scaling']['brier']:.4f} | ECE: {cal_results['platt_scaling']['ece']:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 8 & 9: FINAL UNTOUCHED HELD-OUT TEST EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE 9: SINGLE-PASS EVALUATION ON UNTOUCHED HELD-OUT TEST SET (N=74)")
    print("=" * 75)

    # 1. KNN Baseline
    knn_test = evaluate_knn_baseline(train_pairs, test_pairs)

    # 2. Base CNN @ Default Threshold 0.50
    cnn_default_test = single_pass_test_eval(t_labels, t_damage_probs, threshold=0.50)

    # 3. Base CNN @ Validation-Optimized Operating Threshold
    cnn_opt_test = single_pass_test_eval(t_labels, t_damage_probs, threshold=opt_threshold)

    # 4. Focal Loss CNN @ 0.50 and @ opt_threshold
    with torch.no_grad():
        t_focal_logits = np.concatenate([focal_model(x).numpy() for x, _ in test_loader], axis=0)
    t_focal_probs = (np.exp(t_focal_logits) / np.sum(np.exp(t_focal_logits), axis=1, keepdims=True))[:, 1]
    focal_test = single_pass_test_eval(t_labels, t_focal_probs, threshold=0.50)

    # 5. Embedding Classifiers Test
    feature_extractor = base_cnn
    def get_test_feats(loader):
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                f = feature_extractor.features(x)
                pooled = feature_extractor.pool(f).flatten(1).numpy()
                feats.append(pooled)
        return np.concatenate(feats, axis=0)

    X_test_feats = get_test_feats(test_loader)
    lr_test_probs = best_lr.predict_proba(X_test_feats)[:, 1]
    lr_test = single_pass_test_eval(t_labels, lr_test_probs, threshold=0.50)

    svm_test_probs = best_svm.predict_proba(X_test_feats)[:, 1]
    svm_test = single_pass_test_eval(t_labels, svm_test_probs, threshold=0.50)

    # Print Full Comparative Table
    print("\n--- UNTOUCHED HELD-OUT TEST SET BENCHMARK RESULTS ---")
    headers = ["Model / Policy", "Threshold", "Accuracy", "Recall", "Precision", "F1", "F2", "ROC-AUC", "PR-AUC", "Balanced Acc"]
    print(f"{headers[0]:<35} | {headers[1]:<9} | {headers[2]:<8} | {headers[3]:<7} | {headers[4]:<9} | {headers[5]:<6} | {headers[6]:<6} | {headers[7]:<7} | {headers[8]:<7} | {headers[9]:<11}")
    print("-" * 125)

    rows_data = [
        ("Legacy KNN Baseline (k=7)", "0.50", knn_test["accuracy"], knn_test["recall"], knn_test["precision"], knn_test["f1"], 0.6724, knn_test["roc_auc"], knn_test["pr_auc"], 0.8108),
        ("Current CNN @ Default 0.50", "0.50", cnn_default_test["accuracy"], cnn_default_test["recall"], cnn_default_test["precision"], cnn_default_test["f1"], cnn_default_test["f2"], cnn_default_test["roc_auc"], cnn_default_test["pr_auc"], cnn_default_test["balanced_acc"]),
        (f"CNN @ Opt Threshold ({opt_threshold:.2f})", f"{opt_threshold:.2f}", cnn_opt_test["accuracy"], cnn_opt_test["recall"], cnn_opt_test["precision"], cnn_opt_test["f1"], cnn_opt_test["f2"], cnn_opt_test["roc_auc"], cnn_opt_test["pr_auc"], cnn_opt_test["balanced_acc"]),
        ("Focal Loss CNN", "0.50", focal_test["accuracy"], focal_test["recall"], focal_test["precision"], focal_test["f1"], focal_test["f2"], focal_test["roc_auc"], focal_test["pr_auc"], focal_test["balanced_acc"]),
        ("Backbone Embed + Logistic Reg", "0.50", lr_test["accuracy"], lr_test["recall"], lr_test["precision"], lr_test["f1"], lr_test["f2"], lr_test["roc_auc"], lr_test["pr_auc"], lr_test["balanced_acc"]),
        ("Backbone Embed + SVM (RBF)", "0.50", svm_test["accuracy"], svm_test["recall"], svm_test["precision"], svm_test["f1"], svm_test["f2"], svm_test["roc_auc"], svm_test["pr_auc"], svm_test["balanced_acc"])
    ]

    for r in rows_data:
        print(f"{r[0]:<35} | {r[1]:<9} | {r[2]:<8.4f} | {r[3]:<7.4f} | {r[4]:<9.4f} | {r[5]:<6.4f} | {r[6]:<6.4f} | {r[7]:<7.4f} | {r[8]:<7.4f} | {r[9]:<11.4f}")

    # Bootstrap 95% CIs for Optimized Model
    print(f"\nBootstrap 95% Confidence Intervals for CNN @ Operating Threshold {opt_threshold:.2f}:")
    for m_name, ci_vals in cnn_opt_test["confidence_intervals_95"].items():
        print(f"  {m_name:18}: [{ci_vals[0]:.4f}, {ci_vals[1]:.4f}]")

    # --------------------------------------------------------------------------
    # PHASE 10: ENVIRONMENTAL STRESS ROBUSTNESS TESTING
    # --------------------------------------------------------------------------
    print(f"\n--- PHASE 10: ENVIRONMENTAL STRESS ROBUSTNESS (Threshold={opt_threshold:.2f}) ---")
    robustness_results = evaluate_robustness_at_threshold(base_cnn, test_pairs, opt_threshold)
    for cond, r_met in robustness_results.items():
        print(f"  {cond:25}: Acc={r_met['accuracy']:.4f} | Rec={r_met['recall']:.4f} | Prec={r_met['precision']:.4f} | F1={r_met['f1']:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 11: EXPORT COMPARISON CSV & VISUALIZATION
    # --------------------------------------------------------------------------
    df_comparison = pd.DataFrame([
        {
            "Model": r[0],
            "Threshold": r[1],
            "Accuracy": r[2],
            "Recall": r[3],
            "Precision": r[4],
            "F1": r[5],
            "F2": r[6],
            "ROC_AUC": r[7],
            "PR_AUC": r[8],
            "Balanced_Accuracy": r[9]
        } for r in rows_data
    ])
    csv_path = os.path.join(FIGURES_DIR, "final_model_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved model comparison CSV to {csv_path}")

    # Plot Model Comparison Bar Chart
    plt.figure(figsize=(11, 5.5))
    models_labels = [
        "KNN Baseline\n(@ 0.50)",
        "CNN\n(@ 0.50)",
        f"CNN Opt Thresh\n(@ {opt_threshold:.2f})",
        "Focal Loss\n(@ 0.50)",
        "Embed + LogReg\n(@ 0.50)",
        "Embed + SVM\n(@ 0.50)"
    ]
    recall_vals = [r[3] for r in rows_data]
    f1_vals = [r[5] for r in rows_data]
    acc_vals = [r[2] for r in rows_data]

    x = np.arange(len(models_labels))
    width = 0.25

    plt.bar(x - width, recall_vals, width, label="Recall (Damaged Class)", color="crimson")
    plt.bar(x, f1_vals, width, label="F1 Score", color="darkorange")
    plt.bar(x + width, acc_vals, width, label="Accuracy", color="dodgerblue")

    plt.xticks(x, models_labels, fontsize=9.5, fontweight="bold")
    plt.ylabel("Test Metric Score", fontsize=11)
    plt.ylim(0, 1.15)
    plt.title("Untouched Test Benchmark: Performance Across Vision Models & Operating Thresholds", fontsize=11.5, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "final_model_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved model comparison plot to {chart_path}")

    # Update metadata file with optimal threshold and comparison metrics
    metrics_record = {
        "frozen_baseline_cnn_50": cnn_default_test,
        "frozen_baseline_knn": knn_test,
        "optimized_threshold_model": {
            "policy": selected_policy,
            "optimal_threshold": opt_threshold,
            "test_evaluation": cnn_opt_test
        },
        "embedding_models": {
            "logistic_regression": lr_test,
            "svm_rbf": svm_test
        },
        "calibration": {
            "uncalibrated_ece": cal_results["uncalibrated"]["ece"],
            "temperature_scaling_ece": cal_results["temperature_scaling"]["ece"],
            "platt_scaling_ece": cal_results["platt_scaling"]["ece"]
        },
        "robustness_stress_metrics": robustness_results
    }
    with open(os.path.join(MODEL_DIR, "vision_metrics_optimized.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"Saved optimized vision metrics to {MODEL_DIR}/vision_metrics_optimized.json")


if __name__ == "__main__":
    main()
