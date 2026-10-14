"""
predict.py - Reusable Prediction and Urgency Triage Module
Project: Student Announcement Intelligence System (SWYNEX Internship Task 2)
Author: Diya Patel

Provides modular, safe functions to:
1. Load the trained scikit-learn model pipeline.
2. Predict announcement category and compute class probabilities.
3. Estimate priority (HIGH, MEDIUM, LOW) using transparent, documented rules.
4. Produce a structured, action-oriented summary using deterministic logic.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
# pyrefly: ignore [missing-import]
import joblib


# Global cache for the loaded pipeline
_CACHED_MODEL = None


def get_default_model_path() -> Path:
    """Return the absolute path to the default serialized model file."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "models" / "announcement_classifier.joblib"


def load_model(model_path: Optional[Path] = None):
    """
    Safely load the trained Pipeline from disk.
    Caches the model in memory so it is not reloaded on every call.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    path = Path(model_path) if model_path else get_default_model_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {path}. "
            "Please train the model first by running: python train_model.py"
        )

    try:
        model = joblib.load(path)
        if model_path is None:
            _CACHED_MODEL = model
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {str(e)}")


def predict_category(text: str, model=None) -> Dict[str, Any]:
    """
    Predict the category and calculate class probabilities for given announcement text.
    
    Parameters:
        text (str): Announcement content to classify.
        model: Optional pre-loaded sklearn Pipeline. If None, uses cached default.
        
    Returns:
        dict: {
            "category": str,
            "confidence": float,
            "probabilities": dict[str, float]
        }
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Announcement text must be a non-empty string.")

    cleaned_text = text.strip()

    if model is None:
        model = load_model()

    # Predict class
    predicted_category = model.predict([cleaned_text])[0]

    # Calculate probabilities if supported
    probabilities = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned_text])[0]
        classes = model.classes_
        for cls_name, prob in zip(classes, probs):
            probabilities[str(cls_name)] = round(float(prob), 4)
        confidence = probabilities.get(str(predicted_category), 0.0)
    else:
        # Fallback if probability is unavailable
        confidence = 1.0
        probabilities[str(predicted_category)] = 1.0

    # Sort probabilities descending
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    return {
        "category": str(predicted_category),
        "confidence": confidence,
        "probabilities": sorted_probs
    }


def estimate_priority(text: str, category: str) -> Dict[str, str]:
    """
    Estimate priority (HIGH, MEDIUM, LOW) using transparent, documented rules.
    
    NOTE:
    Priority estimation is explicitly RULE-BASED, deterministic, and separate from
    the machine learning category prediction. It inspects urgency markers, deadlines,
    penalties, and category sensitivity.
    
    Rules:
    - HIGH: Imminent deadlines (today, tomorrow, within 24h), mandatory compliance,
            penalties, fee overdue/hall ticket withheld, attendance debarment, exams tomorrow.
    - MEDIUM: Action required within 3-7 days, homework releases, upcoming tests,
              registration deadlines next week, general academic submissions.
    - LOW: Routine informational circulars, cultural activities, general advisories,
           long-term vacation declarations, library updates without imminent penalty.
    """
    text_lower = text.lower()
    cat_upper = category.upper()

    # High urgency patterns
    high_urgency_keywords = [
        "urgent", "immediately", "immediate", "today", "tonight", "tomorrow",
        "last date", "last day", "final deadline", "strictly", "mandatory",
        "compulsory", "debarred", "withheld", "late fee", "penalty",
        "within 24 hours", "asap", "warning", "hall ticket", "admit card",
        "emergency", "closed tomorrow", "cancelled tomorrow", "strike",
        "discrepancy"
    ]

    # Medium urgency patterns
    medium_urgency_keywords = [
        "upcoming", "reminder", "due on", "deadline", "submit by", "next week",
        "register before", "scheduled for", "shortlisted", "submission",
        "guidelines", "mock", "installment", "verification", "grace period"
    ]

    matched_high = [kw for kw in high_urgency_keywords if kw in text_lower]
    matched_medium = [kw for kw in medium_urgency_keywords if kw in text_lower]

    # Decision Logic
    if matched_high:
        reasons = f"Triggered by high-urgency keywords: '{', '.join(matched_high[:3])}'"
        if cat_upper in ["EXAM", "FEE", "PLACEMENT"]:
            reasons += f" in critical category [{cat_upper}]."
        return {
            "priority": "HIGH",
            "reason": reasons
        }

    # Category-specific overrides
    if cat_upper == "EXAM":
        if matched_medium or any(k in text_lower for k in ["timetable", "schedule", "quiz", "viva", "seating"]):
            return {
                "priority": "HIGH",
                "reason": "Examination announcements carry inherently elevated priority for students."
            }
        return {
            "priority": "MEDIUM",
            "reason": "General examination circular without immediate deadline."
        }

    if cat_upper == "FEE":
        if any(k in text_lower for k in ["due", "payment", "portal", "receipt", "installment"]):
            return {
                "priority": "HIGH",
                "reason": "Fee payment deadlines involve direct financial and registration consequences."
            }
        return {
            "priority": "MEDIUM",
            "reason": "General fee information notice."
        }

    if cat_upper == "ASSIGNMENT":
        if matched_medium or "due" in text_lower or "submit" in text_lower:
            return {
                "priority": "MEDIUM",
                "reason": "Active academic assignment requiring student submission within standard coursework window."
            }
        return {
            "priority": "MEDIUM",
            "reason": "Coursework assignment notification."
        }

    if cat_upper == "PLACEMENT":
        if any(k in text_lower for k in ["drive", "interview", "assessment", "register", "apply"]):
            return {
                "priority": "MEDIUM",
                "reason": "Placement recruitment opportunity requiring student registration or preparation."
            }
        return {
            "priority": "LOW",
            "reason": "General placement informational update."
        }

    if cat_upper == "HOLIDAY":
        if any(k in text_lower for k in ["cyclone", "rainfall", "weather", "emergency", "shut"]):
            return {
                "priority": "HIGH",
                "reason": "Emergency campus closure due to severe weather/administrative alerts."
            }
        return {
            "priority": "LOW",
            "reason": "Scheduled institutional holiday/vacation announcement. No student action required."
        }

    # GENERAL or default
    if matched_medium:
        return {
            "priority": "MEDIUM",
            "reason": f"Moderate urgency keywords detected: '{', '.join(matched_medium[:2])}'."
        }

    return {
        "priority": "LOW",
        "reason": "Routine informational announcement. No urgent deadline or mandatory compliance required."
    }


def extract_potential_deadline(text: str) -> Optional[str]:
    """Extract a potential deadline or date phrase using regex heuristics."""
    patterns = [
        r"(?:before|by|on|until|till)\s+([A-Za-z]+ \d{1,2}(?:st|nd|rd|th)?(?:, \d{4})?)",
        r"(?:before|by|on|until|till)\s+(\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+(?: \d{4})?)",
        r"(?:before|by|at|until)\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)",
        r"(?:by|before)\s+(tomorrow(?: evening| morning| night)?)",
        r"(?:by|before)\s+(tonight|midnight)",
        r"(?:on|before|by)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def generate_action_summary(text: str, category: str, priority: str) -> Dict[str, Any]:
    """
    Produce a short, action-oriented summary using deterministic, transparent logic.
    
    IMPORTANT:
    This summary is NOT produced by a generative AI model (such as GPT or Gemini).
    It uses deterministic template synthesis and regex-extracted context to give
    clear, actionable guidance to the student without hallucinations.
    """
    cat_upper = category.upper()
    priority_upper = priority.upper()
    deadline = extract_potential_deadline(text)

    # Base templates by category
    category_templates = {
        "EXAM": {
            "action_title": "Examination Notice & Preparation",
            "primary_action": "Check schedule, verify room allocation, and download admit card.",
            "checklist": [
                "Verify subject codes and session timings on the official exam portal.",
                "Ensure physical student ID and hall ticket are printed and stamped.",
                "Confirm calculator and stationery allowances before entering the hall."
            ]
        },
        "ASSIGNMENT": {
            "action_title": "Coursework & Assignment Submission",
            "primary_action": "Complete assigned tasks and submit deliverables on the designated platform.",
            "checklist": [
                "Review submission requirements (PDF, code repo, or hard copy).",
                "Run originality/plagiarism checks and ensure formatting compliance.",
                "Upload deliverables ahead of the deadline to avoid portal congestion."
            ]
        },
        "EVENT": {
            "action_title": "Campus Event & Activity Participation",
            "primary_action": "Review event itinerary, register if interested, and save to calendar.",
            "checklist": [
                "Check eligibility, registration fees (if any), and team size requirements.",
                "Submit registration before the slot capacity is reached.",
                "Note down event venue, reporting time, and mandatory dress code."
            ]
        },
        "HOLIDAY": {
            "action_title": "Holiday & Campus Closure Notice",
            "primary_action": "Note non-instructional dates and plan personal academic schedule.",
            "checklist": [
                "Verify whether library, labs, and mess facilities remain operational.",
                "Adjust travel plans and project milestones around the non-working days.",
                "Track makeup/rescheduled class announcements from faculty."
            ]
        },
        "PLACEMENT": {
            "action_title": "Career & Placement Drive Opportunity",
            "primary_action": "Verify eligibility criteria and complete drive registration promptly.",
            "checklist": [
                "Update resume, CGPA, and contact credentials on the placement portal.",
                "Review interview rounds, technical topics, and coding assessment slots.",
                "Prepare formal business attire and carry copies of resume and college ID."
            ]
        },
        "FEE": {
            "action_title": "Institutional Fee Payment & Dues Clearance",
            "primary_action": "Pay dues through official online ERP gateway to prevent penalties.",
            "checklist": [
                "Verify exact payable amount and fee category on the student portal.",
                "Complete digital transaction before the surcharge/late fee deadline.",
                "Download and retain transaction receipt for departmental verification."
            ]
        },
        "GENERAL": {
            "action_title": "General Campus Advisory & Notice",
            "primary_action": "Read advisory instructions and comply with university administration guidelines.",
            "checklist": [
                "Take note of affected facilities, adjusted timings, or maintenance windows.",
                "Contact designated office or student desk if clarification is required.",
                "Adhere to university code of conduct and safety regulations."
            ]
        }
    }

    template = category_templates.get(cat_upper, category_templates["GENERAL"])

    deadline_text = f"Identified deadline/timing: {deadline}." if deadline else "No specific deadline identified in text."

    if priority_upper == "HIGH":
        urgency_note = "HIGH URGENCY: Immediate action recommended within 24 hours to prevent academic or financial penalties."
    elif priority_upper == "MEDIUM":
        urgency_note = "MEDIUM URGENCY: Action required within the week. Track upcoming deadline."
    else:
        urgency_note = "LOW URGENCY: Informational update. Read and keep for reference."

    return {
        "title": template["action_title"],
        "primary_action": template["primary_action"],
        "checklist": template["checklist"],
        "detected_deadline": deadline,
        "deadline_note": deadline_text,
        "urgency_note": urgency_note,
        "methodology": "Deterministic rule-based summary (Extractive heuristic, not Generative LLM)"
    }


def detect_language(text: str) -> str:
    """
    Detect whether the input text contains Gujarati script characters.
    Gujarati Unicode block: U+0A80 to U+0AFF.
    """
    if any("\u0A80" <= ch <= "\u0AFF" for ch in text):
        return "Gujarati"
    return "English"


def analyze_announcement(text: str, model=None) -> Dict[str, Any]:
    """
    Comprehensive pipeline function that orchestrates:
    1. Input Validation & Language Detection
    2. ML Category Classification (scikit-learn TF-IDF + Logistic Regression)
    3. Rule-based Urgency Priority Estimation (HIGH, MEDIUM, LOW)
    4. Deterministic Action-Oriented Summary Generation
    
    Returns a unified, structured dictionary suitable for API and UI consumption.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "is_valid": False,
            "error_message": "Announcement text is empty or invalid. Please paste valid announcement content.",
            "category": None,
            "confidence": 0.0,
            "probabilities": {},
            "priority": None,
            "priority_reason": None,
            "action_summary": None,
            "language_detected": "Unknown",
            "language_notice": None,
            "confidence_explanation": None
        }

    cleaned_text = text.strip()

    # Detect language
    lang = detect_language(cleaned_text)
    lang_notice = None
    if lang == "Gujarati":
        lang_notice = (
            "Notice: Gujarati script detected. The current ML model is trained exclusively "
            "on an English corpus. Category predictions on non-English text may be inaccurate "
            "due to vocabulary mismatch. See the 'Language Support' guide for multilingual expansion."
        )

    # Input length check
    warning_notice = None
    if len(cleaned_text) < 10:
        warning_notice = "Warning: Announcement text is very short (< 10 characters). Predictions may have low confidence."

    try:
        # 1. ML Category Prediction
        cat_result = predict_category(cleaned_text, model=model)
        category = cat_result["category"]
        confidence = cat_result["confidence"]
        probabilities = cat_result["probabilities"]

        # 2. Rule-based Priority
        priority_info = estimate_priority(cleaned_text, category)
        priority = priority_info["priority"]
        priority_reason = priority_info["reason"]

        # 3. Action-oriented Summary
        action_summary = generate_action_summary(cleaned_text, category, priority)

        # 4. Confidence explanation
        confidence_explanation = (
            f"Model statistical confidence is {confidence * 100:.1f}%. "
            "This probability is derived from the Logistic Regression softmax output over TF-IDF n-gram "
            "frequencies and reflects similarity to training patterns. It is not an absolute factual guarantee."
        )

        return {
            "is_valid": True,
            "error_message": None,
            "warning_message": warning_notice,
            "input_text": cleaned_text,
            "category": category,
            "confidence": confidence,
            "probabilities": probabilities,
            "priority": priority,
            "priority_reason": priority_reason,
            "action_summary": action_summary,
            "language_detected": lang,
            "language_notice": lang_notice,
            "confidence_explanation": confidence_explanation
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_message": f"Prediction failed: {str(e)}",
            "warning_message": None,
            "category": None,
            "confidence": 0.0,
            "probabilities": {},
            "priority": None,
            "priority_reason": None,
            "action_summary": None,
            "language_detected": lang,
            "language_notice": lang_notice,
            "confidence_explanation": None
        }


# ==========================================
# Command Line Interface (CLI)
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("  STUDENT ANNOUNCEMENT INTELLIGENCE SYSTEM — PREDICTION CLI")
    print("=" * 70)

    if len(sys.argv) > 1:
        sample_input = " ".join(sys.argv[1:])
    else:
        sample_input = (
            "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. "
            "All students must download their admit cards before Friday 5:00 PM."
        )

    print(f"\nInput Announcement:\n\"{sample_input}\"\n")

    result = analyze_announcement(sample_input)

    if not result["is_valid"]:
        print(f"[ERROR] {result['error_message']}")
        sys.exit(1)

    print(f"Predicted Category : {result['category']} (Confidence: {result['confidence'] * 100:.2f}%)")
    print(f"Estimated Priority : {result['priority']}")
    print(f"Priority Reason    : {result['priority_reason']}")
    print(f"\nTop Class Probabilities:")
    for cat, prob in list(result["probabilities"].items())[:4]:
        print(f"  - {cat:<12}: {prob * 100:.2f}%")
    print(f"\nRecommended Action : {result['action_summary']['primary_action']}")
    print(f"Urgency Note       : {result['action_summary']['urgency_note']}")
    print(f"Deadline Context   : {result['action_summary']['deadline_note']}")
    print("=" * 70)
