"""
src/llm_layer.py
----------------
Constrained LLM explanation layer using the Google Gemini API with robust template fallback.

Project Architectural Rules:
1. The LLM is STRICTLY an explanation / verbalization layer. It rephrases the ML model's
   prediction and SHAP drivers in plain language.
2. The LLM must NEVER decide, invent, or hallucinate non-data reasons.
3. System prompt strictly enforces:
   - "You are an academic advisor assistant. Use ONLY the facts in the JSON. Do not mention any factor, number, or cause that is not in the JSON."
   - Association phrasing ("associated with lower/higher predicted score"), never causal ("because", "you failed due to").
   - Ethical guardrails: NEVER advise on gender, family income, school type, parental education, or any NON_ACTIONABLE factor.
   - Fixed 4-part structure:
     (a) 1-line result summary
     (b) 2-3 sentences on main factors
     (c) 2-3 actionable suggestions using the what-if numbers
     (d) 1-line model estimate disclaimer
   - Low temperature (0.2), supportive tone, concise (under 150 words).
4. Template Fallback: If GEMINI_API_KEY is missing, placeholder, or the API call fails,
   the system seamlessly falls back to a deterministic template that preserves identical
   structure and ethical constraints. The app NEVER crashes due to the API.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configurable constants
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE: float = 0.2


def template_fallback_explanation(payload: Dict[str, Any], audience: str = "student") -> str:
    """
    Generates a structured, deterministic explanation directly from the SHAP payload
    without relying on external LLM APIs. Guarantees 100% uptime and exact format compliance.

    Args:
        payload: JSON-serializable dictionary from src/explain.py.
        audience: Target reader ("student" or "teacher").

    Returns:
        Structured explanation string under 150 words.
    """
    score = payload.get("predicted_score", 0.0)
    band = payload.get("band", "Borderline")
    pos_factors = payload.get("top_positive_factors", [])
    neg_factors = payload.get("top_negative_factors", [])
    suggestions = payload.get("actionable_suggestions", [])

    is_student = (audience.lower() == "student")

    # (a) 1-line result summary
    if is_student:
        part_a = f"Result Summary: Your estimated exam score is {score:.1f} out of 100, placing you in the '{band}' category."
    else:
        part_a = f"Result Summary: The student has a predicted exam score of {score:.1f} out of 100 ({band} category)."

    # (b) 2-3 sentences on main factors
    pos_snippets = []
    for f in pos_factors[:2]:
        feat_name = f["feature"].replace("_", " ")
        pos_snippets.append(f"{feat_name} ({f['value']})")

    neg_snippets = []
    for f in neg_factors[:2]:
        feat_name = f["feature"].replace("_", " ")
        neg_snippets.append(f"{feat_name} ({f['value']})")

    factor_sentences = []
    if pos_snippets:
        factor_sentences.append(
            f"Academic strengths associated with a higher predicted score include " + ", and ".join(pos_snippets) + "."
        )
    if neg_snippets:
        factor_sentences.append(
            f"Conversely, factors associated with a lower predicted score include " + ", and ".join(neg_snippets) + "."
        )
    if not factor_sentences:
        factor_sentences.append("Student performance is evenly distributed across baseline evaluation factors.")

    part_b = " ".join(factor_sentences)

    # (c) 2-3 specific, actionable suggestions using what-if numbers
    suggestion_lines = []
    if suggestions:
        for s in suggestions[:3]:
            rec = s.get("recommendation", "")
            if rec:
                suggestion_lines.append(f"- {rec}")
    else:
        if is_student:
            suggestion_lines.append("- Continue maintaining your current study rhythm and active classroom participation.")
        else:
            suggestion_lines.append("- Encourage the student to maintain current attendance and study habits.")

    part_c = "Actionable Steps:\n" + "\n".join(suggestion_lines)

    # (d) 1-line model estimate disclaimer
    part_d = "Note: This assessment is an automated statistical estimate based on historical patterns, not a guaranteed final outcome."

    # Assemble complete formatted text
    full_text = f"{part_a}\n\n{part_b}\n\n{part_c}\n\n{part_d}"
    return full_text


def build_system_prompt(audience: str = "student") -> str:
    """
    Constructs the strictly constrained system prompt for the Gemini LLM.
    """
    persona = "a compassionate academic advisor speaking directly to a student" if audience.lower() == "student" else "an analytical academic specialist briefing a teacher"

    return f"""You are {persona}.
Your task is to translate an Explainable AI (SHAP) evaluation into plain, encouraging English.

CRITICAL RULES:
1. Use ONLY the facts, features, and numbers provided in the input JSON payload. Do not invent, assume, or extrapolate any outside factors or causes.
2. Phrasing constraint: You must ALWAYS phrase reasons as "associated with a higher predicted score" or "associated with a lower predicted score". NEVER use causal terms such as "because", "due to your failure", "you failed because", or "this caused".
3. Ethical guardrail: NEVER suggest changing or comment negatively on Gender, Family Income, School Type, Parental Education Level, or any non-actionable background attribute.
4. Actionable recommendations: Suggestions must come ONLY from the 'actionable_suggestions' list in the JSON using the exact numbers specified there.
5. Tone: Supportive, clear, simple, and under 150 words total.

OUTPUT FORMAT (Follow exactly):
Result Summary: [Single sentence stating predicted score and performance band]
Key Factors: [2-3 sentences detailing positive and negative drivers using the exact values and associated impacts from the JSON]
Actionable Steps:
- [Actionable recommendation 1 using what-if numbers]
- [Actionable recommendation 2 using what-if numbers if available]
Note: [A one-line disclaimer stating this is a model estimate, not a certainty]"""


def explain_in_words(
    payload: Dict[str, Any],
    audience: str = "student",
    model_name: str = DEFAULT_GEMINI_MODEL
) -> str:
    """
    Verbalizes the SHAP explanation payload in plain English using the Gemini API.
    Falls back gracefully to template_fallback_explanation if the API is unavailable.

    Args:
        payload: JSON-serializable explanation payload from src/explain.py.
        audience: Target audience ("student" or "teacher").
        model_name: Gemini model identifier (default: gemini-1.5-flash).

    Returns:
        Structured explanation string under 150 words.
    """
    # Prefer Streamlit secrets (for Cloud deployment), fall back to .env for local dev
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Fallback if API key is not configured or is a placeholder
    if not api_key or api_key == "your_gemini_api_key_here":
        print("[LLM Layer] Notice: GEMINI_API_KEY not set. Using template-based fallback explanation.")
        return template_fallback_explanation(payload, audience=audience)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        system_instruction = build_system_prompt(audience=audience)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config={
                "temperature": TEMPERATURE,
            }
        )

        user_content = f"Here is the student's evaluation JSON payload:\n```json\n{json.dumps(payload, indent=2)}\n```\nProvide the explanation now."
        response = model.generate_content(user_content)

        if response and response.text:
            return response.text.strip()
        else:
            print("[LLM Layer] Warning: Empty response from Gemini API. Falling back to template.")
            return template_fallback_explanation(payload, audience=audience)

    except Exception as e:
        print(f"[LLM Layer] Warning: Gemini API call failed ({e}). Falling back to template.")
        return template_fallback_explanation(payload, audience=audience)


if __name__ == "__main__":
    from explain import generate_explanation_payload
    import joblib

    print("=" * 70)
    print("TASK 5: TESTING LLM EXPLANATION LAYER")
    print("=" * 70)

    models_dir = Path("models")
    if not (models_dir / "best_model.joblib").exists():
        print("Model artifacts not found. Please run src/train.py and src/explain.py first.")
        sys.exit(1)

    model = joblib.load(models_dir / "best_model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")
    X_test_raw = joblib.load(models_dir / "X_test_raw.joblib")
    y_test = joblib.load(models_dir / "y_test.joblib")

    # Pick 3 diverse students: Low, Mid, High
    sorted_idx = y_test.sort_values().index
    indices = [
        ("STUDENT 1 (Low Score / At-Risk)", sorted_idx[0], "student"),
        ("STUDENT 2 (Borderline Score)", sorted_idx[len(sorted_idx) // 2], "student"),
        ("STUDENT 3 (High Score - Teacher View)", sorted_idx[-1], "teacher"),
    ]

    for label, idx, aud in indices:
        student_data = X_test_raw.loc[idx].to_dict()
        actual_score = y_test.loc[idx]
        payload = generate_explanation_payload(
            student_dict=student_data,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names
        )

        print(f"\n{'=' * 50}")
        print(f"{label} (Actual Score: {actual_score}, Audience: {aud.upper()})")
        print(f"{'=' * 50}")

        explanation = explain_in_words(payload, audience=aud)
        print(explanation)
        print(f"\n[Word Count: {len(explanation.split())} words]")
