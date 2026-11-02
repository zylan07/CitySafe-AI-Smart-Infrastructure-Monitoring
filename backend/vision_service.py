"""
CitySafe / ZylanZest V2 — Vision Service & Occlusion Explainability
==================================================================
Handles image inference and explainability:
  1. Primary: CitySafe MobileNet CNN with Temperature Scaling Calibration
  2. Fallback: Legacy KNN (k=7) Model (Explicitly labeled)
  3. Explainability: 5x5 Occlusion Sensitivity Map (Perturbation Heatmap)
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import pickle

from backend.model.citysafe_cnn import CitySafeCNN

MODEL_DIR = "backend/model"
CNN_WEIGHTS = os.path.join(MODEL_DIR, "citysafe_cnn.pt")
METRICS_FILE = os.path.join(MODEL_DIR, "vision_metrics.json")
OPTIMIZED_METRICS_FILE = os.path.join(MODEL_DIR, "vision_metrics_optimized.json")
KNN_FILE = os.path.join(MODEL_DIR, "concrete_knn_model.pkl")

# Global model state
_cnn_model = None
_temperature = 1.0
_operating_threshold = 0.23
_knn_model = None
_model_source = "Uninitialized"


def init_vision_models():
    global _cnn_model, _temperature, _operating_threshold, _knn_model, _model_source

    # Try loading CNN
    if os.path.exists(CNN_WEIGHTS):
        try:
            model = CitySafeCNN(num_classes=2)
            state_dict = torch.load(CNN_WEIGHTS, map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            _cnn_model = model
            _model_source = "CitySafe MobileNet CNN"

            # Load fitted temperature
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    _temperature = float(meta.get("optimal_temperature", 1.0))

            # Load validation-optimized operating threshold
            if os.path.exists(OPTIMIZED_METRICS_FILE):
                with open(OPTIMIZED_METRICS_FILE, "r", encoding="utf-8") as f:
                    opt_meta = json.load(f)
                    _operating_threshold = float(opt_meta.get("optimized_threshold_model", {}).get("optimal_threshold", 0.23))

            print(f"Loaded CitySafe MobileNet CNN with temperature T={_temperature:.4f}, operating threshold theta={_operating_threshold:.2f}")
        except Exception as e:
            print(f"Notice: Failed to initialize CNN model: {e}")
            _cnn_model = None

    # Load legacy KNN fallback
    if os.path.exists(KNN_FILE):
        try:
            with open(KNN_FILE, "rb") as f:
                _knn_model = pickle.load(f)
            if _cnn_model is None:
                _model_source = "Legacy KNN fallback model"
            print("Loaded legacy KNN model fallback.")
        except Exception as e:
            print(f"Notice: Failed to load legacy KNN: {e}")


# Initialize on import
init_vision_models()


def preprocess_for_cnn(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def predict_damage(pil_img: Image.Image):
    """
    Evaluates image and returns:
      prediction: "Damaged" or "Safe"
      damage_probability: float in [0, 1]
      confidence: float in [0, 1]
      model_source: str
    """
    global _cnn_model, _knn_model, _model_source, _temperature

    if _cnn_model is not None:
        try:
            x = preprocess_for_cnn(pil_img)
            with torch.no_grad():
                logits = _cnn_model(x) / _temperature
                probs = torch.softmax(logits, dim=1)[0].numpy()
                damage_p = float(probs[1])
                pred_label = "Damaged" if damage_p >= _operating_threshold else "Safe"
                conf = float(max(probs[0], probs[1]))
                return pred_label, damage_p, conf, f"CitySafe MobileNet CNN (Operating Threshold theta={_operating_threshold:.2f})"
        except Exception as e:
            print(f"CNN prediction failed, falling back to KNN: {e}")

    # Fallback to KNN
    if _knn_model is not None:
        try:
            gray = pil_img.convert("L").resize((20, 20))
            flattened = (np.array(gray, dtype=np.float32) / 255.0).ravel()
            pred_int = int(_knn_model.predict([flattened])[0])
            try:
                probs = _knn_model.predict_proba([flattened])[0]
                conf = float(max(probs))
                damage_p = float(probs[1]) if len(probs) > 1 else (0.85 if pred_int == 1 else 0.15)
            except Exception:
                conf = 0.80
                damage_p = 0.85 if pred_int == 1 else 0.15

            pred_label = "Damaged" if pred_int == 1 else "Safe"
            return pred_label, damage_p, conf, "Legacy KNN fallback model"
        except Exception as e:
            print(f"KNN fallback also failed: {e}")

    # Neutral fallback
    return "Damaged", 0.70, 0.70, "Heuristic prior (models offline)"


def generate_occlusion_heatmap(pil_img: Image.Image, grid_size: int = 5) -> str:
    """
    Generates a 5x5 occlusion sensitivity heatmap:
    Masks one grid cell at a time with neutral gray, observes the drop in damage probability,
    and returns a base64 PNG data URI overlaying the heatmap.
    """
    orig_img = pil_img.convert("RGB").resize((224, 224))
    _, base_damage_p, _, _ = predict_damage(orig_img)

    w, h = orig_img.size
    cell_w = w // grid_size
    cell_h = h // grid_size

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    for r in range(grid_size):
        for c in range(grid_size):
            # Create occluded copy
            occluded = orig_img.copy()
            draw = ImageDraw.Draw(occluded)
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))

            _, occ_damage_p, _, _ = predict_damage(occluded)
            # Drop in damage probability indicates regional importance
            importance = max(0.0, base_damage_p - occ_damage_p)
            heatmap[r, c] = importance

    # Normalize heatmap
    max_val = np.max(heatmap)
    if max_val > 1e-6:
        heatmap_norm = heatmap / max_val
    else:
        heatmap_norm = heatmap

    # Upsample heatmap to 224x224
    heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8), mode="L")
    heatmap_upsampled = heatmap_img.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert to RGBA red-yellow heatmap
    heat_arr = np.array(heatmap_upsampled, dtype=np.float32) / 255.0
    rgba = np.zeros((224, 224, 4), dtype=np.uint8)
    rgba[:, :, 0] = (heat_arr * 255).astype(np.uint8)              # Red
    rgba[:, :, 1] = ((1.0 - heat_arr) * heat_arr * 255).astype(np.uint8) # Orange/Yellow
    rgba[:, :, 2] = 0
    rgba[:, :, 3] = (heat_arr * 160).astype(np.uint8)             # Semi-transparent alpha

    overlay = Image.fromarray(rgba, mode="RGBA")
    composite = Image.alpha_composite(orig_img.convert("RGBA"), overlay)

    buffered = io.BytesIO()
    composite.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

