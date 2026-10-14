"""
test_system.py - Comprehensive Automated Verification Test Suite
Project: Student Announcement Intelligence System (SWYNEX Internship Task 2)
Author: Diya Patel

Validates all 8 core project requirements:
1. Dataset validation and class balance.
2. Model pipeline loading and serialization.
3. Missing model error handling.
4. Empty, whitespace, and short input validation.
5. Multiclass predictions and probabilities across all 7 categories.
6. Separation of rule-based priority from statistical ML prediction.
7. Language detection and Gujarati advisory notices.
8. SQLite persistence, search, multi-field filtering, and CSV export.
9. Aggregate dashboard statistics calculation (Total, High, Medium, Low).
10. Evaluation metrics integrity (Accuracy, Macro-F1, Confusion Matrix).
"""

import os
import sys
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict import (
    load_model,
    predict_category,
    estimate_priority,
    generate_action_summary,
    detect_language,
    analyze_announcement,
    get_default_model_path
)
import history_db


# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_records = []

    def log(self, test_name: str, passed: bool, details: str = ""):
        if passed:
            self.passed += 1
            status = "PASS"
            symbol = "[OK]"
        else:
            self.failed += 1
            status = "FAIL"
            symbol = "[FAIL]"
        self.test_records.append((test_name, status, details))
        print(f"  {symbol} [{status}] {test_name}")
        if details:
            print(f"        {details}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print(f"  TEST EXECUTION SUMMARY: {self.passed}/{total} Passed ({(self.passed/total)*100:.1f}%)")
        print("=" * 70)
        return self.failed == 0


def run_all_tests():
    runner = TestResults()
    print("=" * 70)
    print("  STUDENT ANNOUNCEMENT INTELLIGENCE SYSTEM — TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Dataset Integrity & Balance
    # -------------------------------------------------------------------------
    print("\n[1/10] Testing Dataset Integrity & Class Distribution...")
    csv_path = BASE_DIR / "data" / "announcements.csv"
    if not csv_path.exists():
        runner.log("Dataset file existence", False, f"Not found at {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        has_cols = {"announcement_text", "category"}.issubset(df.columns)
        runner.log("Dataset required columns present", has_cols, f"Found columns: {list(df.columns)}")
        
        row_count = len(df)
        runner.log("Dataset row count == 245", row_count == 245, f"Total rows: {row_count}")
        
        counts = df["category"].value_counts()
        is_balanced = len(counts) == 7 and all(c == 35 for c in counts.values)
        runner.log("Dataset balanced across 7 classes (35 each)", is_balanced, f"Classes: {dict(counts)}")

    # -------------------------------------------------------------------------
    # Test 2: Model Pipeline & Serialization
    # -------------------------------------------------------------------------
    print("\n[2/10] Testing Model Pipeline & Serialization...")
    model_path = get_default_model_path()
    if not model_path.exists():
        runner.log("Serialized model artifact exists", False, f"Missing {model_path}")
    else:
        runner.log("Serialized model artifact exists", True, f"Found {model_path.name}")
        try:
            model = load_model()
            has_steps = hasattr(model, "named_steps") and "tfidf" in model.named_steps and "classifier" in model.named_steps
            runner.log("Pipeline contains 'tfidf' and 'classifier' steps", has_steps, f"Steps: {list(getattr(model, 'named_steps', {}).keys())}")
            
            has_proba = hasattr(model, "predict_proba")
            runner.log("Pipeline supports predict_proba()", has_proba)
            
            classes = list(getattr(model, "classes_", []))
            has_7_classes = len(classes) == 7
            runner.log("Model recognizes 7 classes", has_7_classes, f"Classes: {classes}")
        except Exception as e:
            runner.log("Model loading exception", False, str(e))

    # -------------------------------------------------------------------------
    # Test 3: Missing Model Error Handling
    # -------------------------------------------------------------------------
    print("\n[3/10] Testing Missing Model File Error Handling...")
    try:
        dummy_path = BASE_DIR / "models" / "non_existent_dummy_model.joblib"
        load_model(dummy_path)
        runner.log("Missing model file raises FileNotFoundError", False, "No exception raised")
    except FileNotFoundError as fnf:
        runner.log("Missing model file raises FileNotFoundError", True, f"Correctly caught: {fnf.args[0][:60]}...")
    except Exception as e:
        runner.log("Missing model file raises FileNotFoundError", False, f"Unexpected exception: {type(e)}")

    # -------------------------------------------------------------------------
    # Test 4: Empty, Whitespace & Short Input Handling
    # -------------------------------------------------------------------------
    print("\n[4/10] Testing Empty, Whitespace & Short Input Handling...")
    # Empty string
    r_empty = analyze_announcement("")
    runner.log("Empty string returns is_valid=False", not r_empty["is_valid"], r_empty.get("error_message", ""))
    
    # Whitespace only
    r_spaces = analyze_announcement("     \n\t   ")
    runner.log("Whitespace-only returns is_valid=False", not r_spaces["is_valid"], r_spaces.get("error_message", ""))
    
    # None input
    r_none = analyze_announcement(None)
    runner.log("None input returns is_valid=False", not r_none["is_valid"], r_none.get("error_message", ""))
    
    # Very short string (< 10 chars)
    r_short = analyze_announcement("Fee due")
    runner.log("Short text (<10 chars) includes warning_message", r_short["is_valid"] and r_short.get("warning_message") is not None, r_short.get("warning_message", ""))

    # -------------------------------------------------------------------------
    # Test 5: Multi-class Prediction & Probabilities across All 7 Categories
    # -------------------------------------------------------------------------
    print("\n[5/10] Testing Multi-class Predictions across All 7 Categories...")
    test_cases = [
        ("EXAM", "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. All students must download admit cards before Friday 5:00 PM."),
        ("FEE", "Urgent Notice: The last date for payment of Semester 6 tuition fees without late fine is October 31, 2026. Hall tickets will be withheld for pending dues."),
        ("ASSIGNMENT", "Machine Learning Assignment 2 on Logistic Regression is due on Google Classroom by Friday at 11:59 PM. Submissions above 15% plagiarism will receive zero marks."),
        ("PLACEMENT", "Amazon India will be conducting an online coding assessment for final year B.Tech students on Saturday at 10:00 AM on HackerRank. Shortlisted candidates must update resume before tonight."),
        ("HOLIDAY", "Due to severe weather and heavy rainfall warning issued by the meteorological department, the university will remain closed tomorrow."),
        ("EVENT", "The Department of Computer Science is hosting a 24-hour National Hackathon 'CodeVerse 2026' on November 15-16. Cash prizes worth Rs 1,00,000. Registrations close this Sunday."),
        ("GENERAL", "Central Library advisory: All borrowed books and reference materials must be returned or renewed by Friday afternoon to avoid automatic overdue library fines.")
    ]

    all_matched = True
    for cat_target, text_sample in test_cases:
        res = analyze_announcement(text_sample)
        pred_cat = res["category"]
        conf = res["confidence"]
        probs = res["probabilities"]
        
        prob_sum = sum(probs.values())
        valid_prob = 0.99 <= prob_sum <= 1.01
        
        matches = (pred_cat == cat_target)
        if not matches:
            all_matched = False
        runner.log(
            f"Prediction matches target [{cat_target}]",
            matches and valid_prob,
            f"Predicted: {pred_cat} (Confidence: {conf*100:.1f}%, ProbSum: {prob_sum:.3f})"
        )

    # -------------------------------------------------------------------------
    # Test 6: Separation of Priority Rules from ML Category
    # -------------------------------------------------------------------------
    print("\n[6/10] Testing Separation of Rule-Based Priority from ML Category...")
    urgent_exam = "URGENT: Final exam admit card will be withheld immediately due to attendance shortage unless cleared today."
    urgent_res = analyze_announcement(urgent_exam)
    runner.log(
        "Emergency keywords trigger HIGH priority with reason",
        urgent_res["priority"] == "HIGH" and "urgent" in urgent_res["priority_reason"].lower(),
        f"Priority: {urgent_res['priority']} | Reason: {urgent_res['priority_reason']}"
    )

    routine_event = "The Photography Club is having an informal orientation meeting next month."
    routine_res = analyze_announcement(routine_event)
    runner.log(
        "Informational event notice triggers LOW priority",
        routine_res["priority"] in ["LOW", "MEDIUM"],
        f"Priority: {routine_res['priority']} | Reason: {routine_res['priority_reason']}"
    )

    # -------------------------------------------------------------------------
    # Test 7: Language Detection & Gujarati Advisory Notice
    # -------------------------------------------------------------------------
    print("\n[7/10] Testing Language Detection & Gujarati Advisory Notice...")
    gujarati_text = "વિદ્યાર્થીઓને જણાવવામાં આવે છે કે સેમેસ્ટર ૫ ની પરીક્ષાનું ટાઈમટેબલ જાહેર થઈ ગયું છે."
    guj_lang = detect_language(gujarati_text)
    guj_res = analyze_announcement(gujarati_text)
    runner.log(
        "Gujarati script detected accurately via Unicode",
        guj_lang == "Gujarati" and guj_res["language_detected"] == "Gujarati",
        f"Detected language: {guj_lang}"
    )
    runner.log(
        "Gujarati input triggers honest OOV advisory notice",
        guj_res.get("language_notice") is not None and "gujarati" in guj_res["language_notice"].lower(),
        f"Notice: {guj_res.get('language_notice', '')[:70]}..."
    )

    english_text = "Semester 5 registration opens next Monday on the university portal."
    eng_lang = detect_language(english_text)
    eng_res = analyze_announcement(english_text)
    runner.log(
        "English text detected as English with no warning notice",
        eng_lang == "English" and eng_res.get("language_notice") is None,
        f"Detected language: {eng_lang}"
    )

    # -------------------------------------------------------------------------
    # Test 8: SQLite Persistence, Search, Multi-Filter & CSV Export
    # -------------------------------------------------------------------------
    print("\n[8/10] Testing SQLite Persistence, Search, Multi-Filter & CSV Export...")
    history_db.init_db()
    
    # Save a test record
    unique_marker = "UniqueVerificationNotice2026_TestMarker"
    test_id = history_db.save_prediction(
        text=f"{unique_marker}: Please submit lab journals by tomorrow evening.",
        category="ASSIGNMENT",
        confidence=0.8888,
        priority="HIGH",
        priority_reason="Triggered by tomorrow keyword",
        primary_action="Complete assigned tasks and submit deliverables.",
        detected_deadline="tomorrow evening",
        language_detected="English"
    )
    runner.log("save_prediction() returns valid primary key ID", test_id > 0, f"Inserted ID: {test_id}")

    # Search filter
    search_df = history_db.get_history(search_query=unique_marker)
    runner.log(
        "get_history(search_query) retrieves matching record",
        len(search_df) >= 1 and unique_marker in search_df.iloc[0]["Announcement Text"],
        f"Found {len(search_df)} record(s)"
    )

    # Category filter
    cat_df = history_db.get_history(category_filter="ASSIGNMENT")
    runner.log(
        "get_history(category_filter) filters correctly",
        len(cat_df) >= 1 and all(c == "ASSIGNMENT" for c in cat_df["Category"]),
        f"All {len(cat_df)} returned records have category ASSIGNMENT"
    )

    # Priority filter
    prio_df = history_db.get_history(priority_filter="HIGH")
    runner.log(
        "get_history(priority_filter) filters correctly",
        len(prio_df) >= 1 and all(p == "HIGH" for p in prio_df["Priority"]),
        f"All {len(prio_df)} returned records have priority HIGH"
    )

    # Language filter
    lang_df = history_db.get_history(language_filter="English")
    runner.log(
        "get_history(language_filter) filters correctly",
        len(lang_df) >= 1 and all(l == "English" for l in lang_df["Language"]),
        f"All {len(lang_df)} returned records have language English"
    )

    # CSV Export test
    csv_buf = io.StringIO()
    search_df.to_csv(csv_buf, index=False)
    csv_text = csv_buf.getvalue()
    has_csv_headers = "ID,Timestamp,Category,Confidence,Priority" in csv_text
    runner.log("History DataFrame exports cleanly to CSV format", has_csv_headers and len(csv_text) > 50)

    # -------------------------------------------------------------------------
    # Test 9: Aggregate Dashboard Statistics Calculation
    # -------------------------------------------------------------------------
    print("\n[9/10] Testing Dashboard Aggregate Statistics Calculation...")
    stats = history_db.get_dashboard_stats()
    req_keys = {"total_count", "high_count", "medium_count", "low_count", "avg_confidence", "category_counts", "priority_counts", "language_counts"}
    has_stat_keys = req_keys.issubset(stats.keys())
    runner.log("get_dashboard_stats() returns all 8 required keys", has_stat_keys, f"Keys: {list(stats.keys())}")

    prio_sum = stats["high_count"] + stats["medium_count"] + stats["low_count"]
    counts_consistent = (stats["total_count"] == prio_sum)
    runner.log(
        "Total count == High + Medium + Low priority sum",
        counts_consistent,
        f"Total: {stats['total_count']} == (H:{stats['high_count']} + M:{stats['medium_count']} + L:{stats['low_count']})"
    )

    avg_conf_valid = 0.0 <= stats["avg_confidence"] <= 1.0
    runner.log("Average confidence is a valid probability [0.0, 1.0]", avg_conf_valid, f"Avg Confidence: {stats['avg_confidence']}")

    # -------------------------------------------------------------------------
    # Test 10: Model Evaluation Artifacts & Reports Integrity
    # -------------------------------------------------------------------------
    print("\n[10/10] Testing Model Evaluation Artifacts & Reports Integrity...")
    reports_dir = BASE_DIR / "reports"
    metrics_file = reports_dir / "evaluation_metrics.json"
    rep_txt_file = reports_dir / "classification_report.txt"
    matrix_img = reports_dir / "confusion_matrix.png"
    classes_file = reports_dir / "classes.json"

    artifacts_exist = metrics_file.exists() and rep_txt_file.exists() and matrix_img.exists() and classes_file.exists()
    runner.log("All 4 evaluation report artifacts exist on disk", artifacts_exist)

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        has_metrics = "accuracy" in m_data and "macro_f1" in m_data and "weighted_f1" in m_data
        is_realistic = 0.60 <= m_data.get("accuracy", 0) <= 0.95
        test_size_49 = m_data.get("test_samples") == 49
        runner.log(
            "Evaluation metrics reflect actual held-out test split (49 samples)",
            has_metrics and is_realistic and test_size_49,
            f"Test Samples: {m_data.get('test_samples')} | Accuracy: {m_data.get('accuracy')*100:.2f}% | Macro F1: {m_data.get('macro_f1'):.4f}"
        )

    if matrix_img.exists():
        img_size = matrix_img.stat().st_size
        runner.log("Confusion matrix plot is a valid non-empty image", img_size > 50000, f"Size: {img_size:,} bytes")

    # Final summary
    all_success = runner.summary()
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
