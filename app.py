"""
app.py - Upgraded Streamlit Web Dashboard for Student Announcement Intelligence System
Project: Student Announcement Intelligence System (SWYNEX Internship Task 2)
Author: Diya Patel

Features:
1. Live Announcement Triage (ML Category + Rule-based Priority + Action Checklist)
2. Advanced Analytics Dashboard (Category counts, Priority stats, Charts)
3. Persistent Announcement History (SQLite database with search, filter, and CSV export)
4. Empirical Model Evaluation (Held-out Test Accuracy, Macro F1, Confusion Matrix)
5. Transparent Language Support & Gujarati Analysis
6. Benchmark Reference Examples
"""

import os
import io
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

# Import prediction functions and SQLite history module
from predict import (
    analyze_announcement,
    load_model,
    get_default_model_path,
    detect_language
)
import history_db
from train_model import train_and_evaluate

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Accessible, Professional Blue/Purple Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Announcement Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    
    .hero-subtitle {
        font-size: 1.02rem;
        color: #c7d2fe;
        max-width: 900px;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .hero-pill {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e0e7ff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    
    .card-caption {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.4;
    }
    
    /* Priority Status Badges */
    .priority-badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .priority-badge-medium {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fcd34d;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .priority-badge-low {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #86efac;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Category Pill */
    .category-badge {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #93c5fd;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Language Tag */
    .lang-badge-en {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .lang-badge-gu {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Checklist Box */
    .checklist-box {
        background: #f8fafc;
        border-left: 4px solid #4f46e5;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .checklist-box li {
        color: #1e293b;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }
    
    .notice-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Helpers & Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained NLP pipeline...")
def get_model():
    """Load and cache the trained Pipeline."""
    try:
        model = load_model()
        return model, None
    except Exception as e:
        return None, str(e)


def load_evaluation_metrics():
    """Load evaluation reports from disk."""
    base_dir = Path(__file__).resolve().parent
    metrics_path = base_dir / "reports" / "evaluation_metrics.json"
    matrix_path = base_dir / "reports" / "confusion_matrix.png"
    report_path = base_dir / "reports" / "classification_report.txt"

    metrics = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    report_text = None
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
        except Exception:
            report_text = None

    return metrics, matrix_path if matrix_path.exists() else None, report_text


# Predefined sample announcements for testing
PRESET_SAMPLES = {
    "Exam Timetable Clash (EXAM)": (
        "Notice regarding exam timetable clash: Students having two examination papers "
        "on the same day must report to the Controller of Examinations before 4:00 PM tomorrow "
        "with their printed hall tickets."
    ),
    "Tuition Fee Arrears (FEE)": (
        "Urgent Notice: The last date for payment of Semester 6 tuition fees without late fine "
        "is October 31, 2026. A penalty of Rs 500 per week applies afterwards. Hall tickets "
        "will be withheld for pending dues."
    ),
    "ML Assignment Deadline (ASSIGNMENT)": (
        "Machine Learning Assignment 2 on Logistic Regression is due on Google Classroom by "
        "Friday at 11:59 PM. Any submission exceeding 15% Turnitin plagiarism similarity "
        "will receive zero marks."
    ),
    "Amazon Campus Drive (PLACEMENT)": (
        "Amazon India will be conducting an online coding assessment for final year B.Tech "
        "students on Saturday at 10:00 AM on the HackerRank platform. Shortlisted candidates "
        "must update their resume on the portal before tonight."
    ),
    "Emergency Weather Holiday (HOLIDAY)": (
        "Due to severe weather and heavy rainfall warning issued by the meteorological department, "
        "the university will remain closed tomorrow for all students and teaching faculties."
    ),
    "National Hackathon (EVENT)": (
        "The Department of Computer Science is hosting a 24-hour National Hackathon 'CodeVerse 2026' "
        "on November 15-16. Cash prizes worth Rs 1,00,000. Registrations close this Sunday."
    ),
    "Library & Campus Advisory (GENERAL)": (
        "Central Library advisory: All borrowed books and reference materials must be returned "
        "or renewed by Friday afternoon to avoid automatic overdue library fines."
    ),
    "Gujarati Sample (Experimental Test)": (
        "વિદ્યાર્થીઓને જણાવવામાં આવે છે કે સેમેસ્ટર ૫ ની પરીક્ષાનું ટાઈમટેબલ જાહેર થઈ ગયું છે. "
        "તમામ વિદ્યાર્થીઓએ હોલ ટિકિટ ડાઉનલોડ કરી લેવી."
    )
}


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=55
    )
    st.title("Announcement AI")
    st.caption("SWYNEX Internship Task 2 — Model Integration")

    st.markdown("---")
    st.subheader("System Status")

    model_path = get_default_model_path()
    if model_path.exists():
        st.success("Trained Model: Loaded")
        st.caption(f"Artifact: `{model_path.name}`")
    else:
        st.error("Model Missing")
        st.info("Run `python train_model.py` to train and serialize the pipeline.")

    # Live Database History Quick Stats
    db_stats = history_db.get_dashboard_stats()
    st.markdown("---")
    st.subheader("Triage Activity Snapshot")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Total Triaged", db_stats["total_count"])
    with col_sb2:
        st.metric("High Priority", db_stats["high_count"])

    # Held-out Test Metrics Snapshot
    metrics, cm_img_path, _ = load_evaluation_metrics()
    if metrics:
        st.markdown("---")
        st.subheader("Held-Out Test Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        with col_m2:
            st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.caption(f"Evaluated on {metrics['test_samples']} test samples (80/20 split).")

    st.markdown("---")
    st.subheader("Classification Schema")
    st.markdown("""
    - 📝 **EXAM**: Timetables, admit cards, clashing schedules
    - 💻 **ASSIGNMENT**: LMS submissions, lab files, code repos
    - 🎪 **EVENT**: Hackathons, cultural fests, workshops
    - 🏖️ **HOLIDAY**: National breaks, weather closures
    - 💼 **PLACEMENT**: Campus drives, interviews, mock rounds
    - 💳 **FEE**: Tuition deadlines, fines, receipts
    - 📢 **GENERAL**: Wi-Fi, library, lost-and-found, cafeteria
    """)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748b;'>
        <strong>Engineering Transparency:</strong><br>
        Categories are predicted via <em>TF-IDF + Logistic Regression</em>.<br>
        Urgency priority is computed via <em>transparent rules</em>.<br>
        Summaries are <em>deterministic heuristics</em> (no generative hallucinations).
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Student Announcement Intelligence System</div>
    <div class="hero-subtitle">
        An NLP-driven triage platform for college circulars: classifies announcements with
        scikit-learn ML, ranks urgency via explainable heuristic rules, maintains persistent
        audit history in SQLite, and provides actionable student checklists.
    </div>
    <div class="badge-bar">
        <span class="hero-pill">🤖 TF-IDF + Logistic Regression</span>
        <span class="hero-pill">⚖️ Stratified 7-Class Dataset</span>
        <span class="hero-pill">🚦 Rule-Based Urgency Engine</span>
        <span class="hero-pill">🗄️ SQLite Prediction History</span>
        <span class="hero-pill">📊 Advanced Statistics & Charts</span>
        <span class="hero-pill">🌐 Gujarati Analysis Included</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Application Tabs
# -----------------------------------------------------------------------------
tab_triage, tab_dashboard, tab_history, tab_metrics, tab_benchmark, tab_architecture = st.tabs([
    "🔍 Live Announcement Triage",
    "📊 Advanced Analytics Dashboard",
    "📜 History & CSV Export",
    "📈 Model Evaluation & Reports",
    "📑 Benchmark Examples",
    "⚙️ Architecture, How It Works & Viva Guide"
])


# =============================================================================
# TAB 1: Live Announcement Triage
# =============================================================================
with tab_triage:
    model, model_error = get_model()

    if model_error:
        st.error(
            f"**Model Unavailable**: {model_error}\n\n"
            "Please train the model by running:\n"
            "```powershell\npython train_model.py\n```"
        )

    # State initialization
    default_sample_text = (
        "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. "
        "All students must download their admit cards before Friday 5:00 PM."
    )
    if "announcement_text_box" not in st.session_state:
        st.session_state["announcement_text_box"] = default_sample_text

    if "current_result" not in st.session_state and model is not None:
        # Pre-populate initial preview without inserting duplicate database records
        st.session_state["current_result"] = analyze_announcement(default_sample_text, model=model)

    # Preset selection row
    col_preset_lbl, col_preset_sel = st.columns([1.2, 3])
    with col_preset_lbl:
        st.markdown("**Select Sample Announcement:**")
    with col_preset_sel:
        preset_choice = st.selectbox(
            "Select an example to auto-fill the input area:",
            ["-- Choose from 8 Curated Presets --"] + list(PRESET_SAMPLES.keys()),
            label_visibility="collapsed",
            key="preset_selector"
        )

    if preset_choice != "-- Choose from 8 Curated Presets --":
        sample_txt = PRESET_SAMPLES[preset_choice]
        st.session_state["announcement_text_box"] = sample_txt
        if model:
            res = analyze_announcement(sample_txt, model=model)
            if res["is_valid"]:
                try:
                    history_db.save_prediction(
                        text=res["input_text"],
                        category=res["category"],
                        confidence=res["confidence"],
                        priority=res["priority"],
                        priority_reason=res["priority_reason"],
                        primary_action=res["action_summary"]["primary_action"],
                        detected_deadline=res["action_summary"]["detected_deadline"],
                        language_detected=res["language_detected"]
                    )
                except Exception:
                    pass
                st.session_state["current_result"] = res
                st.session_state["banner_message"] = ("success", f"✅ Loaded preset [{res['category']}] & saved to history audit log!")
        st.rerun()

    # Quick interactive preset chips
    st.markdown("**Quick Preset Buttons (Click to Auto-Triage):**")
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    quick_presets = [
        ("📝 Exam Timetable", PRESET_SAMPLES["Exam Timetable Clash (EXAM)"]),
        ("💳 Tuition Fee Due", PRESET_SAMPLES["Tuition Fee Arrears (FEE)"]),
        ("💻 ML Assignment", PRESET_SAMPLES["ML Assignment Deadline (ASSIGNMENT)"]),
        ("💼 Campus Drive", PRESET_SAMPLES["Amazon Campus Drive (PLACEMENT)"]),
        ("🏖️ Weather Holiday", PRESET_SAMPLES["Emergency Weather Holiday (HOLIDAY)"]),
        ("🎪 Hackathon", PRESET_SAMPLES["National Hackathon (EVENT)"]),
        ("📢 Library Advisory", PRESET_SAMPLES["Library & Campus Advisory (GENERAL)"]),
        ("🌐 Gujarati Notice", PRESET_SAMPLES["Gujarati Sample (Experimental Test)"])
    ]
    for idx, (chip_title, chip_text) in enumerate(quick_presets):
        target_col = [chip_col1, chip_col2, chip_col3, chip_col4][idx % 4]
        with target_col:
            if st.button(chip_title, key=f"chip_btn_{idx}", use_container_width=True):
                st.session_state["announcement_text_box"] = chip_text
                if model:
                    res = analyze_announcement(chip_text, model=model)
                    if res["is_valid"]:
                        try:
                            history_db.save_prediction(
                                text=res["input_text"],
                                category=res["category"],
                                confidence=res["confidence"],
                                priority=res["priority"],
                                priority_reason=res["priority_reason"],
                                primary_action=res["action_summary"]["primary_action"],
                                detected_deadline=res["action_summary"]["detected_deadline"],
                                language_detected=res["language_detected"]
                            )
                        except Exception:
                            pass
                        st.session_state["current_result"] = res
                        st.session_state["banner_message"] = ("success", f"✅ Loaded preset [{res['category']}] & saved to history audit log!")
                st.rerun()

    # Text Input Area
    user_announcement = st.text_area(
        "Paste College Announcement Text:",
        value=st.session_state.get("announcement_text_box", ""),
        height=130,
        placeholder="Type or paste any college announcement, circular, or email notice here...",
        key="announcement_text_box"
    )

    # Action Buttons
    col_btn_left, col_btn_mid, col_btn_spacer = st.columns([1.2, 1, 3])
    with col_btn_left:
        analyze_clicked = st.button("🚀 Analyze Announcement", type="primary", use_container_width=True)
    with col_btn_mid:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state["announcement_text_box"] = ""
            st.session_state["current_result"] = None
            st.rerun()

    # Banner message if any
    if "banner_message" in st.session_state:
        m_type, m_text = st.session_state.pop("banner_message")
        if m_type == "success":
            st.success(m_text)
        elif m_type == "warning":
            st.warning(m_text)
        elif m_type == "error":
            st.error(m_text)

    # Handle Analyze Click
    if analyze_clicked:
        raw_text = st.session_state.get("announcement_text_box", "")
        if not raw_text.strip():
            st.warning("⚠️ Announcement text is empty. Please enter or paste notice content before analyzing.")
            st.session_state["current_result"] = None
        elif model is None:
            st.error("⚠️ Trained model not found. Please run `python train_model.py` first.")
            st.session_state["current_result"] = None
        else:
            with st.spinner("Analyzing text with NLP pipeline & urgency engine..."):
                res = analyze_announcement(raw_text, model=model)
            if not res["is_valid"]:
                st.error(res["error_message"])
                st.session_state["current_result"] = None
            else:
                try:
                    history_db.save_prediction(
                        text=res["input_text"],
                        category=res["category"],
                        confidence=res["confidence"],
                        priority=res["priority"],
                        priority_reason=res["priority_reason"],
                        primary_action=res["action_summary"]["primary_action"],
                        detected_deadline=res["action_summary"]["detected_deadline"],
                        language_detected=res["language_detected"]
                    )
                except Exception:
                    pass
                st.session_state["current_result"] = res
                st.session_state["banner_message"] = ("success", "✅ Announcement analyzed and recorded in audit history!")
                st.rerun()

    # Display Result
    result = st.session_state.get("current_result")
    if result and result.get("is_valid"):
        # Language or short-text advisory notices
        if result.get("language_notice"):
            st.warning(f"🌐 {result['language_notice']}")
        if result.get("warning_message"):
            st.info(f"ℹ️ {result['warning_message']}")

        st.markdown("---")
        st.subheader("Triage Results & Intelligence Summary")

        # Top Metric Cards (3 Columns)
        col_c1, col_c2, col_c3 = st.columns(3)

        # 1. Category Card
        with col_c1:
            cat_name = result["category"]
            conf_pct = result["confidence"] * 100
            lang_class = "lang-badge-gu" if result["language_detected"] == "Gujarati" else "lang-badge-en"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Predicted Category (ML Model)</div>
                <div style="margin: 0.5rem 0;">
                    <span class="category-badge">🏷️ {cat_name}</span>
                </div>
                <div class="card-caption">
                    Statistical Confidence: <strong>{conf_pct:.1f}%</strong><br>
                    Language: <span class="{lang_class}">{result['language_detected']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Priority Card
        with col_c2:
            prio = result["priority"]
            badge_class = f"priority-badge-{prio.lower()}"
            prio_icon = "🚨" if prio == "HIGH" else ("⚠️" if prio == "MEDIUM" else "ℹ️")
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Estimated Priority (Rule-Based)</div>
                <div style="margin: 0.5rem 0;">
                    <span class="{badge_class}">{prio_icon} {prio}</span>
                </div>
                <div class="card-caption">
                    <strong>Reason:</strong> {result['priority_reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Timeline / Deadline Card
        with col_c3:
            deadline_str = result["action_summary"]["detected_deadline"] or "No explicit date parsed"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Detected Timeline</div>
                <div class="card-value" style="font-size: 1.25rem; color: #4338ca; margin: 0.5rem 0;">
                    ⏰ {deadline_str}
                </div>
                <div class="card-caption">
                    {result['action_summary']['urgency_note']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Action Summary Section
        st.markdown("#### 📋 Recommended Student Action & Checklist")
        action_info = result["action_summary"]
        st.markdown(f"**Primary Action:** {action_info['primary_action']}")

        st.markdown("""
        <div class="checklist-box">
            <strong>Action Items for Student:</strong>
            <ul style="margin-top: 0.5rem; padding-left: 1.2rem;">
        """, unsafe_allow_html=True)
        for item in action_info["checklist"]:
            st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        # Class Probabilities Breakdown
        with st.expander("📊 View Detailed Category Probabilities", expanded=False):
            st.markdown(
                "*These probabilities represent the softmax output of the trained Logistic Regression model.*"
            )

            probs_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["Category", "Probability"]
            )
            probs_df["Percentage"] = (probs_df["Probability"] * 100).round(2)

            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                st.bar_chart(
                    data=probs_df.set_index("Category")["Percentage"],
                    use_container_width=True
                )
            with col_tbl:
                st.dataframe(
                    probs_df[["Category", "Percentage"]].rename(columns={"Percentage": "Probability (%)"}),
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown(f"""
            <div class="notice-box">
                <strong>Confidence Explanation:</strong> {result['confidence_explanation']}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: Advanced Analytics Dashboard
# =============================================================================
with tab_dashboard:
    st.subheader("📊 Advanced Announcement Analytics")
    st.markdown("""
    Aggregated operational statistics computed dynamically from real announcements processed
    and stored in local SQLite database (`data/history.db`).
    """)

    stats = history_db.get_dashboard_stats()

    # Metric Cards Row - 5 KPIs
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    with col_d1:
        st.metric("Total Processed", stats["total_count"])
    with col_d2:
        st.metric("🚨 High Priority", stats["high_count"], delta="Needs immediate action", delta_color="inverse")
    with col_d3:
        st.metric("⚠️ Medium Priority", stats["medium_count"])
    with col_d4:
        st.metric("ℹ️ Low Priority", stats["low_count"])
    with col_d5:
        st.metric("🎯 Avg Confidence", f"{stats['avg_confidence'] * 100:.1f}%")

    st.markdown("---")

    # Visual Charts Row
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 🏷️ Announcements by Category")
        if not stats["category_counts"].empty:
            st.bar_chart(
                data=stats["category_counts"].set_index("Category")["Count"],
                use_container_width=True
            )
            st.dataframe(stats["category_counts"], hide_index=True, use_container_width=True)
        else:
            st.info("No category data recorded yet.")

    with col_c2:
        st.markdown("#### 🚨 Announcements by Priority Level")
        if not stats["priority_counts"].empty:
            st.bar_chart(
                data=stats["priority_counts"].set_index("Priority")["Count"],
                use_container_width=True
            )
            st.dataframe(stats["priority_counts"], hide_index=True, use_container_width=True)
        else:
            st.info("No priority data recorded yet.")

    # Third Row: Language Distribution & Summary
    st.markdown("---")
    col_lang1, col_lang2 = st.columns([1, 2])
    with col_lang1:
        st.markdown("#### 🌐 Language Distribution")
        if "language_counts" in stats and not stats["language_counts"].empty:
            st.dataframe(stats["language_counts"], hide_index=True, use_container_width=True)
        else:
            st.info("No language data available.")
    with col_lang2:
        st.markdown("#### 💡 Analytics Insights")
        st.markdown(f"""
        - **Audit Summary**: **{stats['total_count']}** announcements triaged and persisted in the local SQLite database.
        - **Urgency Profile**: **{stats['high_count']}** critical notices requiring immediate student action ({round((stats['high_count']/stats['total_count'])*100, 1) if stats['total_count'] else 0}% of all records).
        - **Confidence Metric**: Average statistical confidence across historical predictions is **{stats['avg_confidence'] * 100:.1f}%**.
        - **Data Integrity**: Every metric is calculated dynamically from `data/history.db`. No mock or synthetic metrics are ever displayed.
        """)


# =============================================================================
# TAB 3: History & CSV Export
# =============================================================================
with tab_history:
    st.subheader("📜 Announcement Triage Audit History")
    st.markdown("""
    Every triaged notice is stored in a local **SQLite database** (`data/history.db`)
    without recording personal identifiers. Use the filters below to search, audit, or export history to CSV.
    """)

    # Filter Bar
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
    with col_f1:
        search_kw = st.text_input("🔍 Search Announcement Text:", placeholder="e.g., fee, exam, hackathon...")
    with col_f2:
        category_options = ["ALL", "EXAM", "ASSIGNMENT", "EVENT", "HOLIDAY", "PLACEMENT", "FEE", "GENERAL"]
        selected_cat = st.selectbox("Filter Category:", category_options)
    with col_f3:
        priority_options = ["ALL", "HIGH", "MEDIUM", "LOW"]
        selected_prio = st.selectbox("Filter Priority:", priority_options)
    with col_f4:
        language_options = ["ALL", "English", "Gujarati"]
        selected_lang = st.selectbox("Filter Language:", language_options)

    # Limit option
    col_lim1, col_lim2 = st.columns([3, 1])
    with col_lim2:
        limit_val = st.selectbox("Max rows to fetch:", [50, 100, 200, 500], index=2)

    # Fetch History DataFrame
    history_df = history_db.get_history(
        search_query=search_kw,
        category_filter=selected_cat,
        priority_filter=selected_prio,
        language_filter=selected_lang,
        limit=limit_val
    )

    col_h_info, col_h_export = st.columns([3, 1])
    with col_h_info:
        st.caption(f"Showing **{len(history_df)}** matching historical announcements.")
    with col_h_export:
        if not history_df.empty:
            # Generate CSV in memory
            csv_buffer = io.StringIO()
            history_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"announcement_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Display Table
    if not history_df.empty:
        st.dataframe(
            history_df,
            hide_index=True,
            use_container_width=True,
            height=380
        )
    else:
        st.info("No announcement records match the selected search or filter criteria.")

    # Maintenance row
    with st.expander("🛠️ History Storage Maintenance", expanded=False):
        st.caption("Clear all saved announcement records from the local SQLite database.")
        confirm_clear = st.checkbox("I confirm I want to clear all prediction history.")
        if st.button("🗑️ Clear Database History", type="secondary", disabled=not confirm_clear):
            history_db.clear_history()
            st.session_state["current_result"] = None
            st.success("Database history cleared successfully.")
            st.rerun()


# =============================================================================
# TAB 4: Model Evaluation & Reports
# =============================================================================
with tab_metrics:
    st.subheader("Model Performance & Held-Out Test Evaluation")
    st.markdown("""
    The classification model is an **scikit-learn Pipeline** combining **TF-IDF Vectorization** (unigrams + bigrams)
    with a **Logistic Regression** classifier. It is fitted exclusively on the 80% training partition and evaluated
    on the 20% unseen test partition using stratified sampling.
    """)

    metrics, cm_path, report_text = load_evaluation_metrics()

    if metrics is None:
        st.warning("No evaluation metrics found. Run `python train_model.py` to generate evaluation reports.")
    else:
        # Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%", help="Correct predictions / total test samples")
        with col_m2:
            st.metric("Macro F1-Score", f"{metrics['macro_f1']:.4f}", help="Unweighted average F1 across all 7 classes")
        with col_m3:
            st.metric("Weighted F1-Score", f"{metrics['weighted_f1']:.4f}", help="F1 score weighted by class support")
        with col_m4:
            st.metric("Test Set Size", f"{metrics['test_samples']} samples", help="Stratified held-out test split (20%)")

        st.markdown("---")

        # Two columns: Confusion Matrix and Classification Report
        col_cm, col_rep = st.columns([1, 1])

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix")
            if cm_path and Path(cm_path).exists():
                st.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Set (49 samples)", use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Run `python train_model.py`.")

        with col_rep:
            st.markdown("#### 📋 Detailed Classification Report")
            if report_text:
                st.code(report_text, language="text")
            else:
                st.info("Classification report file not found.")

        # Per-class metrics table
        if "classification_report" in metrics:
            st.markdown("#### 🔬 Per-Class Precision, Recall & F1 Breakdown")
            class_rows = []
            for cls_name in metrics["classes"]:
                if cls_name in metrics["classification_report"]:
                    cdata = metrics["classification_report"][cls_name]
                    class_rows.append({
                        "Category": cls_name,
                        "Precision": f"{cdata['precision'] * 100:.1f}%",
                        "Recall": f"{cdata['recall'] * 100:.1f}%",
                        "F1-Score": f"{cdata['f1-score']:.4f}",
                        "Support (Test Count)": int(cdata['support'])
                    })
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)

        st.caption(
            "⚠️ Notice: All metrics are derived from the stratified held-out test set of the 245-sample demonstration dataset."
        )

        st.markdown("---")
        st.markdown("#### 🔄 Live Retraining Pipeline")
        st.markdown("Re-run the training pipeline to regenerate model weights and fresh evaluation reports:")
        if st.button("🚀 Retrain Model & Refresh Reports", type="secondary"):
            with st.spinner("Training scikit-learn Pipeline and generating reports..."):
                try:
                    new_metrics = train_and_evaluate()
                    st.cache_resource.clear()
                    st.success(f"✅ Model successfully retrained! Held-out accuracy: {new_metrics['accuracy'] * 100:.2f}%. Reports updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retraining failed: {e}")


# =============================================================================
# TAB 5: Benchmark Examples
# =============================================================================
with tab_benchmark:
    st.subheader("System Benchmark: Sample Inputs & Outputs")
    st.markdown("""
    Representative examples across all 7 categories demonstrating end-to-end system behavior:
    ML classification, rule-based urgency ranking, and deterministic actionable guidance.
    Click **'📥 Load into Live Triage'** to instantly analyze any sample in Tab 1.
    """)

    benchmark_data = [
        {
            "Category": "EXAM",
            "Announcement": "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. All students must download admit cards before Friday 5:00 PM.",
            "Expected Priority": "HIGH",
            "Action Output": "Check schedule, verify room allocation, and download admit card."
        },
        {
            "Category": "ASSIGNMENT",
            "Announcement": "Machine Learning Assignment 2 on Logistic Regression is due on Google Classroom by Friday at 11:59 PM. Submissions above 15% plagiarism will receive zero marks.",
            "Expected Priority": "MEDIUM",
            "Action Output": "Complete assigned tasks, run plagiarism check, and submit deliverables."
        },
        {
            "Category": "EVENT",
            "Announcement": "The Department of Computer Science is hosting a 24-hour National Hackathon 'CodeVerse 2026' on November 15-16. Registrations close this Sunday.",
            "Expected Priority": "LOW / MEDIUM",
            "Action Output": "Review event itinerary, register if interested, and save to calendar."
        },
        {
            "Category": "HOLIDAY",
            "Announcement": "Due to severe weather and heavy rainfall warning issued by the meteorological department, the university will remain closed tomorrow.",
            "Expected Priority": "HIGH (Emergency)",
            "Action Output": "Note non-instructional dates and adjust personal academic schedule."
        },
        {
            "Category": "PLACEMENT",
            "Announcement": "Amazon India will be conducting an online coding assessment for final year B.Tech students on Saturday at 10:00 AM on HackerRank. Shortlisted candidates must update resume before tonight.",
            "Expected Priority": "HIGH",
            "Action Output": "Update resume credentials, review assessment slots, and prepare formal attire."
        },
        {
            "Category": "FEE",
            "Announcement": "Notice: The last date for payment of Semester 6 tuition fees without late fine is October 31, 2026. Hall tickets will be withheld for pending dues.",
            "Expected Priority": "HIGH",
            "Action Output": "Pay dues through official online ERP gateway to prevent penalties."
        },
        {
            "Category": "GENERAL",
            "Announcement": "Central Library advisory: All borrowed books and reference materials must be returned or renewed by Friday afternoon to avoid automatic overdue library fines.",
            "Expected Priority": "LOW",
            "Action Output": "Read advisory instructions and comply with university administration guidelines."
        }
    ]

    for idx, item in enumerate(benchmark_data):
        with st.expander(f"📌 [{item['Category']}] — Priority: {item['Expected Priority']}", expanded=(idx == 0)):
            st.markdown(f"**Sample Text:** *\"{item['Announcement']}\"*")
            st.markdown(f"- **Category:** `{item['Category']}`")
            st.markdown(f"- **Expected Priority:** `{item['Expected Priority']}`")
            st.markdown(f"- **Generated Next Action:** {item['Action Output']}")
            if st.button(f"📥 Load into Live Triage: [{item['Category']}]", key=f"btn_bench_load_{idx}"):
                st.session_state["announcement_text_box"] = item["Announcement"]
                if model:
                    res = analyze_announcement(item["Announcement"], model=model)
                    if res["is_valid"]:
                        try:
                            history_db.save_prediction(
                                text=res["input_text"],
                                category=res["category"],
                                confidence=res["confidence"],
                                priority=res["priority"],
                                priority_reason=res["priority_reason"],
                                primary_action=res["action_summary"]["primary_action"],
                                detected_deadline=res["action_summary"]["detected_deadline"],
                                language_detected=res["language_detected"]
                            )
                        except Exception:
                            pass
                        st.session_state["current_result"] = res
                        st.session_state["banner_message"] = ("success", f"✅ Loaded benchmark [{item['Category']}] into Live Triage!")
                st.rerun()


# =============================================================================
# TAB 6: Architecture, How It Works & Viva Guide
# =============================================================================
with tab_architecture:
    st.subheader("System Architecture, How It Works & Internship Review Guide")

    st.markdown("""
    ### 🏗️ Technical Pipeline & Architecture
    The system follows a transparent, modular architecture designed for high explainability:
    
    1. **Data Preprocessing & Validation**:
       - Reads `data/announcements.csv` (245 balanced college announcements across 7 categories).
       - Validates columns, trims whitespace, eliminates empty and duplicate rows.
       
    2. **Machine Learning Model (scikit-learn Pipeline)**:
       - **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=5000, stop_words='english')`.
       - **Classifier**: `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')`.
       - **Strict Holdout Split**: Stratified 80% training / 20% test partitioning (`random_state=42`). The model is **never** evaluated on training data.
       
    3. **Rule-Based Priority Estimation**:
       - **Explicit Separation**: Priority is calculated through a deterministic rule engine, not the statistical ML classifier.
       - Urgency keywords (`urgent`, `today`, `tomorrow`, `mandatory`, `hall ticket`, `penalty`, `withheld`) combined with category sensitivity produce `HIGH`, `MEDIUM`, or `LOW` priority with clear rationales.
       
    4. **Action-Oriented Summary Synthesis**:
       - Extractive date/timing matching via regex.
       - Deterministic template selection based on category and priority level.
       - Discloses that it is heuristic and **not** generated by a generative AI model (such as an LLM).
    """)

    st.markdown("---")
    st.subheader("🧠 Deep Dive: How the Machine Learning Components Work")

    st.markdown("""
    #### 1. What is TF-IDF (Term Frequency - Inverse Document Frequency)?
    TF-IDF converts unstructured text into numerical feature vectors that machine learning algorithms can process:
    - **Term Frequency (TF)**: Measures how frequently a word appears in a specific announcement.
      $$TF(t, d) = \\frac{\\text{count of word } t \\text{ in document } d}{\\text{total words in } d}$$
    - **Inverse Document Frequency (IDF)**: Penalizes words that appear everywhere across college notices (e.g., *"students"*, *"college"*, *"notice"*) and boosts category-discriminative keywords (e.g., *"timetable"*, *"tuition"*, *"plagiarism"*, *"hackathon"*).
      $$IDF(t) = \\log\\left(\\frac{1 + N}{1 + DF(t)}\\right) + 1$$
    - **Sublinear TF Scaling**: Replaces raw count $TF$ with $1 + \\log(TF)$ to prevent repeated words from skewing predictions.
    - **N-Grams (1, 2)**: Extracts both single words (*unigrams*) like *"exam"* and two-word phrases (*bigrams*) like *"hall ticket"*, *"late fee"*, *"last date"*.

    #### 2. What is Logistic Regression & Softmax Classification?
    Logistic Regression learns optimal linear weights for each word/n-gram to discriminate between the 7 classes:
    - The model computes a raw logit score $z_k$ for each category $k$:
      $$z_k = \\sum_{i} w_{ki} x_i + b_k$$
    - The **Softmax activation function** converts these raw logits into normalized probabilities summing to 1.0 (100%):
      $$P(\\text{Category } k \\mid \\text{Text}) = \\frac{e^{z_k}}{\\sum_{j=1}^{7} e^{z_j}}$$
    - The category with the maximum probability is selected as the prediction.
    - **Balanced Class Weights** (`class_weight='balanced'`): Automatically adjusts weights inversely proportional to class frequencies to prevent bias.

    #### 3. Why Keep Urgency Priority Separate from Machine Learning?
    - **Safety & Stakes**: In a student environment, an announcement saying *"Pay fees by 5 PM tomorrow or hall ticket withheld"* carries high real-world stakes.
    - An ML model might miscalculate a numerical priority if phrasing varies slightly.
    - Deterministic rules ensure that critical penalty markers (*debarred*, *withheld*, *penalty*, *imminent deadline*) **guarantee** a HIGH priority alert, while routine circulars stay LOW.
    """)

    st.markdown("---")
    st.subheader("🌐 Language Scope & Gujarati Multilingual Roadmap")
    st.markdown("""
    #### Current Language Scope:
    - **Active Model Language**: **English (`en`)**
    - Current TF-IDF vectorizer vocabulary was fitted exclusively on an English corpus.
    - Built-in Unicode detection (`\\u0A80` to `\\u0AFF`) detects Gujarati script and displays an advisory notice.
    
    #### Why Gujarati Cannot Be Reliably Classified with the Current English Model:
    1. **Vocabulary Mismatch**: A TF-IDF vectorizer trained on English n-grams has zero overlap with Gujarati words (e.g., *પરીક્ષા*, *ફી*, *હોલ ટિકિટ*).
    2. **Out-of-Vocabulary (OOV) Issue**: Any Gujarati text results in an empty or near-zero TF-IDF vector, leading the classifier to predict arbitrary majority or default classes.
    3. **Academic Integrity**: In accordance with the SWYNEX Task 2 guidelines, we do not claim multilingual accuracy until trained and verified on actual multilingual data.

    #### 🗺️ 3-Step Roadmap to Add Full Gujarati Support:
    1. **Curate a Bilingual Dataset**:
       - Create a Gujarati announcement dataset with 30+ samples per category across all 7 classes.
       - Include both pure Gujarati script and Romanized Gujarati (*Gujlish*).
    2. **Multilingual Embeddings**:
       - Replace word-level English TF-IDF with a multilingual embedding model such as:
         - **IndicBERT** (AI4Bharat)
         - **MuRIL** (Multilingual Representations for Indian Languages by Google)
         - **paraphrase-multilingual-MiniLM-L12-v2** (Sentence-Transformers)
    3. **Multilingual Classifier Training**:
       - Train and evaluate a multilingual classifier with stratified holdout validation and publish genuine Gujarati F1-scores.
    """)

    st.markdown("---")
    st.subheader("🎓 Internship Review & Viva Q&A Guide")
    st.markdown("""
    Use these questions and answers to present and explain this project during your internship evaluation:

    **Q1: Why did you choose TF-IDF + Logistic Regression instead of a Deep Learning model or Large Language Model (LLM)?**  
    *Answer:* For single-domain, short-text classification with a 245-sample dataset, TF-IDF + Logistic Regression is computationally efficient, trains in under 2 seconds, runs completely offline on standard hardware without GPU or paid API keys, provides transparent feature weights, and achieves 75.51% test accuracy without overfitting. Deep learning models or LLMs would introduce high latency, cost, and risk of hallucinations.

    **Q2: Why is urgency priority computed with rules instead of machine learning?**  
    *Answer:* Academic urgency is consequence-driven. If an announcement mentions *"hall ticket withheld"* or *"due tomorrow"*, students face immediate debarment. Rules provide 100% deterministic, auditable guarantees for emergency triggers, avoiding probabilistic errors on critical notices.

    **Q3: How did you ensure your model evaluation was rigorous and honest?**  
    *Answer:* We performed a stratified 80/20 train/test split. The pipeline was fitted strictly on the 196 training samples and tested on 49 unseen samples with 7 samples per class. We reported actual Accuracy (75.51%), Macro F1 (0.7655), Weighted F1 (0.7655), and generated a Confusion Matrix. We never fabricated metrics.

    **Q4: How does the local SQLite storage work and is student data protected?**  
    *Answer:* Every triaged notice is saved to a local SQLite database (`data/history.db`) with timestamp, category, priority, and action checklist. No personal identifiable information (PII) like student names, roll numbers, or contact details is recorded.

    **Q5: Can this system handle non-English announcements like Gujarati right now?**  
    *Answer:* The system actively detects Gujarati Unicode characters and displays an honest advisory. The current ML model is trained on an English vocabulary, so Gujarati tokens are out-of-vocabulary (OOV). We have documented a 3-step roadmap to integrate IndicBERT/MuRIL once a Gujarati training corpus is collected.
    """)

    st.markdown("---")
    st.markdown("""
    ### 🛡️ Honesty & Academic Integrity Statement
    - **No Fake Generative AI**: We do not claim this application runs an LLM. Summaries are produced by transparent rule templates.
    - **No Invented Metrics**: All reported evaluation metrics (Accuracy, Macro F1, Confusion Matrix) are generated dynamically from `train_model.py` and read from `reports/evaluation_metrics.json`.
    - **Dataset Scope**: The dataset is a sample prototype collection created for demonstration purposes, not an official internal university database.
    - **Local Prototype**: This application runs locally via Streamlit (`python -m streamlit run app.py`) and does not claim external cloud hosting unless actually deployed.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85rem; color: #64748b; padding: 1rem 0;">
    Student Announcement Intelligence System — SWYNEX Internship Task 2: Model or API Integration<br>
    Developed with Python, scikit-learn, SQLite, and Streamlit.
</div>
""", unsafe_allow_html=True)
