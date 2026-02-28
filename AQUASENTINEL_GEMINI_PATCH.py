"""
AQUASENTINEL GEMINI INTEGRATION PATCH
======================================
Add this to your Cloud Run FastAPI app (main.py or wherever your
/analyze-water-quality endpoint is defined).

Step 1: Add google-genai to your requirements.txt
Step 2: Add GEMINI_API_KEY env var to your Cloud Run service
Step 3: Replace the hardcoded `action_required` with this Gemini call

Deploy command:
  gcloud run deploy ecomed-backend \
    --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY \
    --region us-central1

OR set it in Cloud Console:
  Cloud Run → ecomed-backend → Edit & Deploy → Variables & Secrets → Add GEMINI_API_KEY
"""

# ── ADD THESE IMPORTS AT THE TOP OF YOUR main.py ──────────────────────────────
import os
from google import genai

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if not _GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
_gemini_client = genai.Client(api_key=_GEMINI_KEY)


# ── REPLACE YOUR hardcoded action_required WITH THIS FUNCTION ─────────────────
def generate_recommendation(risk_assessment: dict, affected_communities: list) -> dict:
    """
    Call Gemini to generate a real action recommendation
    based on the AquaSentinel risk assessment results.
    """
    risk_level  = risk_assessment.get("overall_risk", "unknown").upper()
    risk_score  = risk_assessment.get("risk_score", 0)
    contaminants = risk_assessment.get("contaminants_detected", [])
    spread       = risk_assessment.get("spread_prediction", {})
    communities  = [c["name"] for c in affected_communities[:3]]

    prompt = f"""You are AquaSentinel, an AI water contamination response system.

A water contamination event analysis has returned:
- Risk Level: {risk_level} (score: {risk_score}/100)
- Contaminants detected: {contaminants if contaminants else "None confirmed, monitoring active"}
- Spread direction: {spread.get("direction", "unknown")} at {spread.get("velocity", "unknown")}
- Time to reach communities: {spread.get("time_to_reach_communities", "unknown")}
- Affected communities: {", ".join(communities) if communities else "None identified"}

Provide:
1. ONE immediate action (short, urgent, practical)
2. ONE investigation action (for the field team)
Keep each to 1 sentence. No jargon.
Respond ONLY in this JSON format:
{{"immediate": "...", "investigation": "..."}}
"""

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        # Clean markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        import json
        return json.loads(text)
    except Exception as e:
        # Fallback to rule-based if Gemini fails
        if risk_level == "HIGH":
            return {
                "immediate": f"Issue boil water notice to all {len(affected_communities)} downstream communities immediately.",
                "investigation": "Deploy field team to trace contamination source upstream within 1 hour."
            }
        elif risk_level == "MEDIUM":
            return {
                "immediate": "Alert downstream communities to monitor water quality and avoid direct consumption.",
                "investigation": "Collect water samples from 3 upstream points for lab analysis."
            }
        else:
            return {
                "immediate": "Continue routine monitoring — no immediate action required.",
                "investigation": "Run standard contamination checks at next scheduled inspection."
            }


# ── IN YOUR @app.post("/analyze-water-quality") ENDPOINT ─────────────────────
# REPLACE the hardcoded:
#   "action_required": {
#       "immediate": "Issue boil water notice to 3 downstream communities",
#       "investigation": "Send inspection team to coordinates 12.345, 56.789 (suspected source)"
#   }
#
# WITH:
#   "action_required": generate_recommendation(risk_assessment, affected_communities)
#
# Example full response:
#
#   return {
#       "risk_assessment":       risk_assessment,
#       "affected_communities":  affected_communities,
#       "action_required":       generate_recommendation(risk_assessment, affected_communities)
#   }
