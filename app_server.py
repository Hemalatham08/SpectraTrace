"""
SpectraTrace — Flask Prediction API
=====================================
POST /predict  — accepts a multipart image upload, runs it through the
                 CV + ML pipeline, and returns a JSON classification result.

Start with:
    python app_server.py

Requires the dependencies in requirements.txt to be installed.
"""

import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import predict from the same directory
from predict import predict

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Allow all origins so the Vite dev server (localhost:5173) can call this API.
# In production, restrict origins to the specific frontend domain.
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok", "service": "SpectraTrace Prediction API"})


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    -------------
    Body: multipart/form-data
        image  (required) — the captured sample image file

    Response 200:
        {
            "predicted_class": "Baking Soda",
            "confidence": 94.3,          // percentage [0-100]
            "status": "Probable Match"   // or "Inconclusive"
        }

    Response 400 / 500:
        { "error": "<message>" }
    """

    # ── Validate request ──────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file in request. Send field name: 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Received empty filename"}), 400

    # ── Save to a temporary file ──────────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # ── Run prediction ────────────────────────────────────────────────
        predicted_class, confidence = predict(tmp_path)

        status = "Probable Match" if confidence >= 70.0 else "Inconclusive"

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,   # float, percentage
            "status": status,
        })

    except FileNotFoundError as e:
        # Model not found — actionable error message
        return jsonify({"error": str(e)}), 503

    except ValueError as e:
        # Bad image or ROI mismatch
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        app.logger.exception("Unhandled error during prediction")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  SpectraTrace Prediction API")
    print("  http://localhost:5000/predict")
    print("  http://localhost:5000/health")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=True)
