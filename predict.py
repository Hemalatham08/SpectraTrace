"""
SpectraTrace — Prediction Module
=================================
Loads the trained scikit-learn RandomForest model and extracts color
features from a fixed ROI of an input image to classify the substance.

To use with a different model:
  - Replace MODEL_PATH with the new .pkl path
  - Adjust ROI_* constants if the camera/card geometry changes
  - Adjust CONTROL_RGB if the blank-card baseline changes
"""

import os
import numpy as np
import pandas as pd
import cv2
import joblib

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "spectratrace_model.pkl")

# Fixed region-of-interest in pixel coordinates (x, y, width, height)
ROI_X, ROI_Y, ROI_W, ROI_H = 540, 825, 70, 100

# RGB baseline of a blank/control card — used to compute color_change
CONTROL_RGB = (213, 216, 211)

# Feature order must match training schema exactly
FEATURES = ["rgb_r", "rgb_g", "rgb_b", "hue", "saturation", "brightness", "color_change"]

# ---------------------------------------------------------------------------
# Model loading (cached on first call)
# ---------------------------------------------------------------------------

_model = None


def load_model():
    """Load the model from disk once and cache it in the module."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {os.path.abspath(MODEL_PATH)}\n"
                "Run scripts/train_model.py first to generate it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> list:
    """
    Read an image, crop the fixed ROI, and return a 7-element feature vector:
        [rgb_r, rgb_g, rgb_b, hue, saturation, brightness, color_change]

    Args:
        image_path: Absolute path to the image file (JPEG / PNG).

    Returns:
        List of 7 float values.

    Raises:
        ValueError: If the image cannot be read or the ROI is out of bounds.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image at path: {image_path}")

    h_img, w_img = img_bgr.shape[:2]
    if ROI_Y + ROI_H > h_img or ROI_X + ROI_W > w_img:
        raise ValueError(
            f"ROI ({ROI_X},{ROI_Y},{ROI_W},{ROI_H}) exceeds image size "
            f"{w_img}×{h_img}. Check camera resolution or ROI constants."
        )

    # Crop ROI
    roi_bgr = img_bgr[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]

    # ── RGB (note: OpenCV stores as BGR) ──────────────────────────────────
    avg_bgr = roi_bgr.mean(axis=(0, 1))          # [B_avg, G_avg, R_avg]
    rgb_b, rgb_g, rgb_r = float(avg_bgr[0]), float(avg_bgr[1]), float(avg_bgr[2])

    # ── HSV ──────────────────────────────────────────────────────────────
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    avg_hsv = roi_hsv.mean(axis=(0, 1))          # [H_avg, S_avg, V_avg]
    hue, saturation, brightness = float(avg_hsv[0]), float(avg_hsv[1]), float(avg_hsv[2])

    # ── Color change — Euclidean distance from blank-card baseline ────────
    ctrl_r, ctrl_g, ctrl_b = CONTROL_RGB
    color_change = float(
        np.sqrt((rgb_r - ctrl_r) ** 2 + (rgb_g - ctrl_g) ** 2 + (rgb_b - ctrl_b) ** 2)
    )

    return [rgb_r, rgb_g, rgb_b, hue, saturation, brightness, color_change]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(image_path: str) -> tuple[str, float]:
    """
    Run the full feature-extraction + model-inference pipeline.

    Args:
        image_path: Path to the uploaded image file.

    Returns:
        Tuple of (predicted_class: str, confidence: float) where confidence
        is in the range [0, 100] (percentage).
    """
    model = load_model()
    features = extract_features(image_path)

    X = pd.DataFrame([features], columns=FEATURES)
    proba = model.predict_proba(X)[0]

    best_idx = int(np.argmax(proba))
    predicted_class = model.classes_[best_idx]
    confidence = float(proba[best_idx]) * 100.0

    return predicted_class, confidence


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    cls, conf = predict(sys.argv[1])
    print(f"Predicted class : {cls}")
    print(f"Confidence      : {conf:.1f}%")
    print(f"Status          : {'Probable Match' if conf >= 70 else 'Inconclusive'}")
