"""
prediction_server.py  —  EcoMed-AI Public REST API
====================================================
Flask API that your team/friends can call directly.
No Google Cloud auth needed — uses Gemini API key directly.

Run locally:
    python prediction_server.py

Endpoints:
    GET  /health        → health check
    POST /predict       → ML prediction + Gemini AI explanation

Request format:
    {
        "instances": [{"arsenic": 0.05, "lead": 0.01, ...}],
        "explain": true
    }

Response format:
    {
        "predictions": [{
            "verdict": "SAFE",
            "prediction_score": 0.58,
            "ai_explanation": "Your water is safe to drink...",
            "model_version": "EcoMed-AI v2 (Integrated)"
        }]
    }
"""

from flask import Flask, request, jsonify
from integrated_pipeline import IntegratedWaterSafetyPipeline
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

# ── Gemini (google-genai SDK — no Google Cloud auth needed) ──────────────────
try:

    from google import genai as _genai
    _GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not _GEMINI_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    _gemini_client = _genai.Client(api_key=_GEMINI_KEY)
    GEMINI_ENABLED = True
    print("✅ Gemini (google-genai) initialized.")
except Exception as _e:
    _gemini_client = None
    GEMINI_ENABLED = False
    print(f"⚠️  Gemini not available: {_e}")


app = Flask(__name__)

# ── Load ML pipeline once at startup ─────────────────────────────────────────
print("Loading EcoMed-AI Pipeline...")
try:
    pipeline = IntegratedWaterSafetyPipeline()
    print("✅ Pipeline loaded successfully.")
except Exception as e:
    print(f"❌ ERROR loading pipeline: {e}")
    pipeline = None

# ── WHO/EPA limits for contaminant identification ─────────────────────────────
WHO_LIMITS = {
    "arsenic": 0.01, "lead": 0.01, "mercury": 0.006,
    "cadmium": 0.003, "chromium": 0.05, "aluminium": 0.2,
    "nitrates": 50.0, "bacteria": 0.0, "viruses": 0.0,
    "radium": 5.0, "uranium": 0.015,
}

def _gemini_explain(verdict: str, score: float, instance: dict) -> str:
    """Generate a plain-language health advisory using Gemini 2.5 Flash."""
    if not GEMINI_ENABLED or _gemini_client is None:
        return "AI explanation unavailable — Gemini not configured."

    violations = []
    for param, limit in WHO_LIMITS.items():
        val = float(instance.get(param, 0))
        if val > limit:
            violations.append(f"{param} ({val:.4f} — WHO limit {limit})")

    contaminants_str = (", ".join(violations)
                        if violations else "No major contaminants detected")

    prompt = f"""You are EcoMed-AI, a public health water safety assistant.

A water quality analysis returned:
- Verdict: {verdict}
- Safety Score: {score:.2f} out of 1.0  (1.0 = perfectly safe)
- Contaminants exceeding WHO/EPA limits: {contaminants_str}

Your task:
1. Explain the result to a non-expert in simple, plain language (no jargon).
2. If UNSAFE or CAUTION, name the dangerous contaminants and why they are harmful.
3. Give exactly 2 practical, low-cost actions they can take RIGHT NOW.
4. Tone: calm but urgent. Short sentences. Max 4 sentences total.
"""
    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    """Root route — shows API info in the browser."""
    status = "✅ healthy" if pipeline else "❌ model not loaded"
    gemini = "✅ enabled" if GEMINI_ENABLED else "❌ not available"
    return f"""
    <html><head><title>EcoMed-AI API</title>
    <style>
      body {{ font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9;
               display: flex; justify-content: center; margin-top: 60px; }}
      .card {{ background: #161b22; border-radius: 12px; padding: 40px;
               max-width: 600px; width: 100%; border: 1px solid #30363d; }}
      h1 {{ color: #58a6ff; margin-bottom: 4px; }}
      .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
                background: #1f6feb; font-size: 0.85rem; margin-bottom: 24px; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
      td {{ padding: 10px 8px; border-bottom: 1px solid #21262d; font-size: 0.95rem; }}
      td:first-child {{ color: #8b949e; width: 40%; }}
      code {{ background: #21262d; padding: 2px 8px; border-radius: 6px;
              font-size: 0.9rem; color: #79c0ff; }}
    </style></head>
    <body><div class="card">
      <h1>💧 EcoMed-AI</h1>
      <div class="badge">Water Safety Prediction API</div>
      <table>
        <tr><td>Pipeline</td><td>{status}</td></tr>
        <tr><td>Gemini AI</td><td>{gemini}</td></tr>
        <tr><td>Health check</td><td><code>GET /health</code></td></tr>
        <tr><td>Predict</td><td><code>POST /predict</code></td></tr>
        <tr><td>Request body</td>
            <td><code>{{"instances": [{{...params...}}], "explain": true}}</code></td></tr>
      </table>
    </div></body></html>
    """, 200


@app.route('/health', methods=['GET'])
@app.route('/health_check', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy" if pipeline else "unhealthy — model load failed"
    code = 200 if pipeline else 503
    return jsonify({
        "status": status,
        "gemini_enabled": GEMINI_ENABLED,
        "model": "EcoMed-AI v2 (Integrated)"
    }), code


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Request:
        { "instances": [{...water params...}], "explain": true }

    Response:
        { "predictions": [{"verdict": "SAFE", "prediction_score": 0.58,
                           "ai_explanation": "...", "model_version": "..."}] }
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json(force=True)
        instances = data.get("instances", [])
        do_explain = data.get("explain", True)   # default: always explain

        predictions = []
        for instance in instances:
            # 1. ML Prediction
            result = pipeline.predict(instance)
            verdict = result.get("verdict", "UNKNOWN")
            score   = result.get("prediction_score", 0.0)

            # 🚨 SAFETY OVERRIDE: Check WHO limits
            # If model says SAFE but we found Arsenic/Lead/etc, force CAUTION
            viol_count = sum([
                float(instance.get("aluminium", 0)) > 0.2,
                float(instance.get("arsenic", 0))   > 0.01,
                float(instance.get("cadmium", 0))   > 0.003,
                float(instance.get("lead", 0))      > 0.01,
                float(instance.get("mercury", 0))   > 0.006,
                float(instance.get("chromium", 0))  > 0.05,
                float(instance.get("nitrates", 0))  > 50,
                float(instance.get("bacteria", 0))  > 0,
                float(instance.get("viruses", 0))   > 0,
                float(instance.get("radium", 0))    > 5,
                float(instance.get("uranium", 0))   > 0.015
            ])

            if verdict == "SAFE" and viol_count > 0:
                print(f"⚠️ Safety Override: {viol_count} violations found. Downgrading SAFE -> CAUTION.")
                verdict = "CAUTION"
                result["verdict"] = verdict

            # 2. Gemini AI Explanation
            result["ai_explanation"] = (
                _gemini_explain(verdict, score, instance)
                if do_explain else "Explanation skipped."
            )

            predictions.append(result)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"\n🚀 EcoMed-AI REST API running on port {port}")
    print(f"   POST http://localhost:{port}/predict")
    print(f"   GET  http://localhost:{port}/health\n")
    app.run(host='0.0.0.0', port=port, debug=False)
