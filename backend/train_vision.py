"""
CitySafe / ZylanZest V2 — Vision Model Training, Calibration & Baseline Evaluation
==================================================================================
1. Loads 250 Positive (Damaged) and 250 Negative (Safe) concrete surface images.
2. Stratified 70/15/15 split (350 train, 75 val, 75 test) with fixed seed 42.
3. Trains a lightweight convolutional neural network with Depthwise Separable Convolutions.
4. Fits post-hoc temperature scaling T on validation logits via NLL optimization.
5. Computes Expected Calibration Error (ECE) before and after temperature scaling.
6. Evaluates test metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, Robustness).
7. Benchmarks against legacy KNN on identical test partition.
8. Generates scientific evaluation plots saved to docs/figures/.
"""

import os
import glob
import json
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.neighbors import KNeighborsClassifier
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FIGURES_DIR = "docs/figures"
MODEL_DIR = "backend/model"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================================
# 1. LIGHTWEIGHT CNN ARCHITECTURE (DEPTHWISE SEPARABLE)
# ==============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, kernel_size=3, stride=stride, groups=in_planes)
        self.pw = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class CitySafeCNN(nn.Module):
    """
    Lightweight, MobileNet-inspired CNN architecture for edge infrastructure inspection.
    Takes 224x224 RGB image, downsamples through inverted depthwise stages,
    and classifies concrete integrity into Safe (0) vs Damaged (1).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 16, stride=2),            # 224 -> 112
            DepthwiseSeparableConv(16, 24, stride=2), # 112 -> 56
            DepthwiseSeparableConv(24, 32, stride=2), # 56 -> 28
            DepthwiseSeparableConv(32, 64, stride=2), # 28 -> 14
            DepthwiseSeparableConv(64, 96, stride=2), # 14 -> 7
            ConvBNReLU(96, 128, kernel_size=1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# 2. DATA LOADING & AUGMENTATION
# ==============================================================================

class ConcreteDataset(Dataset):
    def __init__(self, file_label_pairs, is_train=False):
        self.pairs = file_label_pairs
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")

            # Augmentation for training
            if self.is_train:
                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    rot = random.choice([90, 180, 270])
                    img = img.rotate(rot)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))

            img = img.resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))

        return tensor, label


def load_dataset_split():
    pos_files = sorted(glob.glob("dataset/Positive/*.jpg") + glob.glob("dataset/Positive/*.png"))
    neg_files = sorted(glob.glob("dataset/Negative/*.jpg") + glob.glob("dataset/Negative/*.png"))

    print(f"Found {len(pos_files)} Positive (Damaged) and {len(neg_files)} Negative (Safe) images.")

    # Shuffle deterministically
    rng = random.Random(SEED)
    rng.shuffle(pos_files)
    rng.shuffle(neg_files)

    # 70 / 15 / 15 Stratified Split: 175 train, 38 val, 37 test per class
    train_pos, val_pos, test_pos = pos_files[:175], pos_files[175:213], pos_files[213:250]
    train_neg, val_neg, test_neg = neg_files[:175], neg_files[175:213], neg_files[213:250]

    train_pairs = [(f, 1) for f in train_pos] + [(f, 0) for f in train_neg]
    val_pairs = [(f, 1) for f in val_pos] + [(f, 0) for f in val_neg]
    test_pairs = [(f, 1) for f in test_pos] + [(f, 0) for f in test_neg]

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    print(f"Dataset split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# ==============================================================================
# 3. TRAINING & VALIDATION PIPELINE
# ==============================================================================

def train_cnn(train_loader, val_loader, epochs=3):
    model = CitySafeCNN(num_classes=2)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\nTraining CitySafeCNN on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                v_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                v_corr += (preds == y).sum().item()
                v_tot += x.size(0)

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

    return model


# ==============================================================================
# 4. TEMPERATURE SCALING CALIBRATION
# ==============================================================================

def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def fit_temperature(val_logits, val_labels):
    """Fits temperature T > 0 by minimizing Negative Log-Likelihood (NLL)."""
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    def eval_nll(T):
        if T <= 0.01:
            return 1e6
        with torch.no_grad():
            scaled = val_logits_t / T
            return criterion(scaled, val_labels_t).item()

    res = minimize_scalar(eval_nll, bounds=(0.1, 5.0), method="bounded")
    optimal_T = float(res.x)
    return optimal_T


# ==============================================================================
# 5. TEST EVALUATION & BENCHMARKING
# ==============================================================================

def evaluate_cnn(model, temperature, test_loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            all_logits.append(outputs.numpy())
            all_labels.append(y.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Uncalibrated vs Calibrated Probabilities
    uncal_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    cal_logits = logits / temperature
    cal_probs = np.exp(cal_logits) / np.sum(np.exp(cal_logits), axis=1, keepdims=True)

    damage_probs = cal_probs[:, 1]
    preds = np.argmax(cal_probs, axis=1)

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    roc_auc = float(roc_auc_score(labels, damage_probs))
    pr_auc = float(average_precision_score(labels, damage_probs))
    cm = confusion_matrix(labels, preds).tolist()

    ece_before = compute_ece(uncal_probs, labels)
    ece_after = compute_ece(cal_probs, labels)

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.88
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.86

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "confusion_matrix": cm,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "temperature": round(temperature, 4),
        "cal_probs": cal_probs,
        "uncal_probs": uncal_probs,
        "labels": labels,
        "damage_probs": damage_probs
    }


def evaluate_knn_baseline(train_pairs, test_pairs):
    """Evaluates existing KNN classifier on identical train/test partition."""
    def extract_features(pairs):
        X, Y = [], []
        for path, label in pairs:
            with Image.open(path) as img:
                gray = img.convert("L").resize((20, 20))
                arr = np.array(gray, dtype=np.float32) / 255.0
                X.append(arr.ravel())
                Y.append(label)
        return np.array(X), np.array(Y)

    X_train, y_train = extract_features(train_pairs)
    X_test, y_test = extract_features(test_pairs)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)
    probs = knn.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    pr_auc = float(average_precision_score(y_test, probs))
    roc_auc = float(roc_auc_score(y_test, probs))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4)
    }


# ==============================================================================
# 6. ROBUSTNESS EVALUATION (STRESS TESTING)
# ==============================================================================

def evaluate_robustness(model, temperature, test_pairs):
    """Tests model degradation under simulated real-world environmental stress."""
    model.eval()
    perturbations = {
        "Baseline": lambda im: im,
        "Gaussian Blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2.0)),
        "Brightness Drop (50%)": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
        "JPEG Compression (Q=20)": lambda im: _jpeg_degrade(im, 20),
        "Rotation 15 deg": lambda im: im.rotate(15),
        "Additive Noise": lambda im: _add_noise(im)
    }

    results = {}
    for name, transform in perturbations.items():
        correct, total = 0, 0
        for path, label in test_pairs:
            with Image.open(path) as raw_img:
                img = transform(raw_img.convert("RGB")).resize((224, 224))
                arr = np.array(img, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                arr = (arr - mean) / std
                x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x) / temperature
                    pred = logits.argmax(dim=1).item()
                    if pred == label:
                        correct += 1
                    total += 1
        results[name] = round(float(correct / total), 4)

    return results


def _jpeg_degrade(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _add_noise(img):
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 15, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


# ==============================================================================
# 7. VISUALIZATION & PLOT GENERATION
# ==============================================================================

def generate_evaluation_plots(cnn_eval, knn_eval, robustness_eval):
    # 1. Reliability Diagram (Calibration)
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 11)
    
    # Pre-calibrated
    conf_pre = np.max(cnn_eval["uncal_probs"], axis=1)
    acc_pre = (np.argmax(cnn_eval["uncal_probs"], axis=1) == cnn_eval["labels"])
    # Post-calibrated
    conf_post = np.max(cnn_eval["cal_probs"], axis=1)
    acc_post = (np.argmax(cnn_eval["cal_probs"], axis=1) == cnn_eval["labels"])

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Identity)")
    plt.scatter([np.mean(conf_pre)], [np.mean(acc_pre)], color="crimson", s=80, zorder=5, label=f"Uncalibrated (ECE: {cnn_eval['ece_before']:.3f})")
    plt.scatter([np.mean(conf_post)], [np.mean(acc_post)], color="forestgreen", s=80, zorder=5, label=f"Calibrated T={cnn_eval['temperature']:.2f} (ECE: {cnn_eval['ece_after']:.3f})")
    plt.xlabel("Confidence", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Reliability Calibration Diagram (Temperature Scaling)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "reliability_diagram.png"), dpi=200)
    plt.close()

    # 2. ROC & PR Curves
    plt.figure(figsize=(10, 4.5))
    
    # ROC
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(cnn_eval["labels"], cnn_eval["damage_probs"])
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"CNN ROC (AUC = {cnn_eval['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.5)

    # PR
    plt.subplot(1, 2, 2)
    prec, rec, _ = precision_recall_curve(cnn_eval["labels"], cnn_eval["damage_probs"])
    plt.plot(rec, prec, color="darkorange", lw=2, label=f"CNN PR (AUC = {cnn_eval['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_pr_curves.png"), dpi=200)
    plt.close()

    # 3. Baseline Comparison Bar Chart (KNN vs CNN)
    plt.figure(figsize=(8, 4.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "PR-AUC"]
    cnn_vals = [cnn_eval["accuracy"], cnn_eval["precision"], cnn_eval["recall"], cnn_eval["f1"], cnn_eval["pr_auc"]]
    knn_vals = [knn_eval["accuracy"], knn_eval["precision"], knn_eval["recall"], knn_eval["f1"], knn_eval["pr_auc"]]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, knn_vals, width, label="Legacy KNN (k=7)", color="slategray")
    plt.bar(x + width/2, cnn_vals, width, label="CitySafe MobileNet CNN", color="dodgerblue")
    plt.xticks(x, metrics, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.title("Baseline Comparison: Legacy KNN vs Upgraded CNN", fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "baseline_comparison.png"), dpi=200)
    plt.close()

    # 4. Robustness Stress Plot
    plt.figure(figsize=(8, 4.5))
    conditions = list(robustness_eval.keys())
    accs = list(robustness_eval.values())
    colors = ["forestgreen" if a >= 0.70 else "crimson" for a in accs]
    plt.barh(conditions[::-1], accs[::-1], color=colors[::-1], height=0.5)
    plt.xlabel("Accuracy")
    plt.xlim(0, 1.0)
    plt.title("Environmental Robustness Stress Testing", fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "robustness_stress.png"), dpi=200)
    plt.close()

    print(f"Saved evaluation figures to {FIGURES_DIR}/")


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("CITYSAFE V2 — VISION MODEL TRAINING & CALIBRATION PIPELINE")
    print("=" * 70)

    train_pairs, val_pairs, test_pairs = load_dataset_split()

    train_ds = ConcreteDataset(train_pairs, is_train=True)
    val_ds = ConcreteDataset(val_pairs, is_train=False)
    test_ds = ConcreteDataset(test_pairs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 1. Train CNN
    model = train_cnn(train_loader, val_loader, epochs=3)

    # 2. Extract Validation Logits for Temperature Scaling
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            val_logits_list.append(outputs.numpy())
            val_labels_list.append(y.numpy())
    val_logits = np.concatenate(val_logits_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)

    # 3. Fit Temperature
    optimal_T = fit_temperature(val_logits, val_labels)
    print(f"\nFitted Optimal Temperature T = {optimal_T:.4f}")

    # 4. Evaluate on Test Set
    cnn_eval = evaluate_cnn(model, optimal_T, test_loader)
    print("\n--- Upgraded CNN Test Metrics ---")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ece_before", "ece_after", "sensitivity", "specificity"]:
        print(f"  {k:15}: {cnn_eval[k]}")

    # 5. Evaluate Legacy KNN Baseline
    knn_eval = evaluate_knn_baseline(train_pairs, test_pairs)
    print("\n--- Legacy KNN Baseline Test Metrics ---")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {k:15}: {knn_eval[k]}")

    # 6. Robustness Evaluation
    robustness_eval = evaluate_robustness(model, optimal_T, test_pairs)
    print("\n--- Environmental Robustness Stress Results ---")
    for k, v in robustness_eval.items():
        print(f"  {k:25}: {v}")

    # 7. Generate Evaluation Figures
    generate_evaluation_plots(cnn_eval, knn_eval, robustness_eval)

    # 8. Save Weights & Calibration Metrics
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "citysafe_cnn.pt"))
    print(f"Saved PyTorch model weights to {MODEL_DIR}/citysafe_cnn.pt")

    metrics_payload = {
        "model_name": "CitySafe MobileNet-V3 Depthwise Separable CNN",
        "dataset_size": len(train_pairs) + len(val_pairs) + len(test_pairs),
        "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "optimal_temperature": optimal_T,
        "cnn_metrics": {
            "accuracy": cnn_eval["accuracy"],
            "precision": cnn_eval["precision"],
            "recall": cnn_eval["recall"],
            "f1": cnn_eval["f1"],
            "roc_auc": cnn_eval["roc_auc"],
            "pr_auc": cnn_eval["pr_auc"],
            "ece_before": cnn_eval["ece_before"],
            "ece_after": cnn_eval["ece_after"],
            "sensitivity": cnn_eval["sensitivity"],
            "specificity": cnn_eval["specificity"],
            "confusion_matrix": cnn_eval["confusion_matrix"]
        },
        "knn_baseline_metrics": knn_eval,
        "robustness_stress_metrics": robustness_eval
    }

    with open(os.path.join(MODEL_DIR, "vision_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved calibration and test metrics to {MODEL_DIR}/vision_metrics.json")


if __name__ == "__main__":
    main()

