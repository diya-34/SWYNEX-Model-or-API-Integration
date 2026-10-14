"""
history_db.py - SQLite Local Storage Module for Announcement History
Project: Student Announcement Intelligence System (SWYNEX Internship Task 2)
Author: Diya Patel

Provides lightweight, robust persistence using Python's built-in sqlite3:
- Stores historical announcement predictions, priorities, and timestamps.
- Supports search, category/priority filtering, and CSV export.
- Computes aggregate metrics for the advanced dashboard.
- Seeds initial benchmark records on first initialization.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


def get_db_path() -> Path:
    """Return absolute path to the SQLite database file."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "history.db"


def get_connection():
    """Create and return a database connection with row factory."""
    conn = sqlite3.connect(get_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the SQLite database schema if not already present.
    Seeds initial realistic records if the history table is empty.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS announcement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                announcement_text TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                priority_reason TEXT,
                primary_action TEXT,
                detected_deadline TEXT,
                language_detected TEXT DEFAULT 'English',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Check if table has any records
        cursor.execute("SELECT COUNT(*) FROM announcement_history")
        count = cursor.fetchone()[0]

        if count == 0:
            # Seed 6 representative historical records for immediate dashboard analytics
            seed_data = [
                (
                    "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. All students must download admit cards before Friday 5:00 PM.",
                    "EXAM", 0.7143, "HIGH",
                    "Examination announcements carry inherently elevated priority for students.",
                    "Check schedule, verify room allocation, and download admit card.",
                    "before Friday 5:00 PM", "English"
                ),
                (
                    "Machine Learning Assignment 2 on Logistic Regression is due on Google Classroom by Friday at 11:59 PM. Turnitin plagiarism cutoff is 15%.",
                    "ASSIGNMENT", 0.8333, "MEDIUM",
                    "Active academic assignment requiring student submission within standard coursework window.",
                    "Complete assigned tasks and submit deliverables on the designated platform.",
                    "at 11:59 PM", "English"
                ),
                (
                    "Urgent Notice: Semester 6 tuition fee payment window closes tomorrow evening. Rs 500 penalty applies afterwards. Hall tickets will be withheld.",
                    "FEE", 0.9231, "HIGH",
                    "Triggered by high-urgency keywords: 'urgent, penalty, withheld' in critical category [FEE].",
                    "Pay dues through official online ERP gateway to prevent penalties.",
                    "tomorrow evening", "English"
                ),
                (
                    "Amazon India on-campus recruitment drive online assessment commences Saturday at 10:00 AM on HackerRank. Shortlisted candidates must update resume.",
                    "PLACEMENT", 0.6316, "HIGH",
                    "Placement recruitment opportunity requiring immediate student action.",
                    "Update resume credentials, review assessment slots, and prepare formal attire.",
                    "on Saturday", "English"
                ),
                (
                    "Due to severe weather and heavy rainfall warning issued by the meteorological department, the university will remain closed tomorrow.",
                    "HOLIDAY", 0.9231, "HIGH",
                    "Emergency campus closure due to severe weather/administrative alerts.",
                    "Note non-instructional dates and plan personal academic schedule.",
                    "tomorrow", "English"
                ),
                (
                    "Department of Computer Science is hosting a 24-hour National Hackathon 'CodeVerse 2026' on November 15-16. Register via Devfolio.",
                    "EVENT", 0.6667, "MEDIUM",
                    "Campus event requiring registration before capacity is reached.",
                    "Review event itinerary, register if interested, and save to calendar.",
                    "on November 15", "English"
                ),
                (
                    "Campus Wi-Fi authentication will be temporarily unavailable on Saturday from 2:00 PM to 6:00 PM due to core router firmware upgrades.",
                    "GENERAL", 0.6667, "LOW",
                    "Routine informational announcement. No urgent deadline or mandatory compliance required.",
                    "Read advisory instructions and comply with university administration guidelines.",
                    "on Saturday", "English"
                )
            ]

            cursor.executemany("""
                INSERT INTO announcement_history (
                    announcement_text, category, confidence, priority,
                    priority_reason, primary_action, detected_deadline, language_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, seed_data)
            conn.commit()


def save_prediction(
    text: str,
    category: str,
    confidence: float,
    priority: str,
    priority_reason: str,
    primary_action: str,
    detected_deadline: Optional[str] = None,
    language_detected: str = "English"
) -> int:
    """Save a newly triaged announcement into SQLite history."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO announcement_history (
                announcement_text, category, confidence, priority,
                priority_reason, primary_action, detected_deadline, language_detected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            text.strip(),
            category.strip().upper(),
            round(float(confidence), 4),
            priority.strip().upper(),
            priority_reason.strip() if priority_reason else "",
            primary_action.strip() if primary_action else "",
            detected_deadline.strip() if detected_deadline else None,
            language_detected.strip()
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(
    search_query: Optional[str] = None,
    category_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
    limit: int = 200
) -> pd.DataFrame:
    """
    Retrieve prediction history with optional search and filters.
    Returns a pandas DataFrame formatted for table display and CSV export.
    """
    init_db()
    query = """
        SELECT 
            id AS "ID",
            created_at AS "Timestamp",
            category AS "Category",
            confidence AS "Confidence",
            priority AS "Priority",
            primary_action AS "Recommended Action",
            detected_deadline AS "Detected Deadline",
            language_detected AS "Language",
            announcement_text AS "Announcement Text",
            priority_reason AS "Priority Reason"
        FROM announcement_history
        WHERE 1=1
    """
    params: List[Any] = []

    if search_query and search_query.strip():
        query += " AND announcement_text LIKE ?"
        params.append(f"%{search_query.strip()}%")

    if category_filter and category_filter != "ALL":
        query += " AND category = ?"
        params.append(category_filter.strip().upper())

    if priority_filter and priority_filter != "ALL":
        query += " AND priority = ?"
        params.append(priority_filter.strip().upper())

    if language_filter and language_filter != "ALL":
        query += " AND language_detected = ?"
        params.append(language_filter.strip())

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def get_dashboard_stats() -> Dict[str, Any]:
    """
    Compute aggregate summary statistics for the Advanced Dashboard tab:
    - Total announcements triaged
    - Category-wise count distribution
    - Priority-wise breakdown (HIGH, MEDIUM, LOW)
    - Average statistical confidence
    - Language distribution (English vs Gujarati)
    """
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()

        # Total announcements
        cursor.execute("SELECT COUNT(*) FROM announcement_history")
        total_count = cursor.fetchone()[0]

        if total_count == 0:
            return {
                "total_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "avg_confidence": 0.0,
                "category_counts": pd.DataFrame(columns=["Category", "Count"]),
                "priority_counts": pd.DataFrame(columns=["Priority", "Count"]),
                "language_counts": pd.DataFrame(columns=["Language", "Count"])
            }

        # Priority breakdown
        cursor.execute("SELECT priority, COUNT(*) FROM announcement_history GROUP BY priority")
        prio_rows = dict(cursor.fetchall())
        high_count = prio_rows.get("HIGH", 0)
        medium_count = prio_rows.get("MEDIUM", 0)
        low_count = prio_rows.get("LOW", 0)

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM announcement_history")
        avg_conf = cursor.fetchone()[0] or 0.0

        # Category distribution
        cursor.execute("""
            SELECT category AS "Category", COUNT(*) AS "Count"
            FROM announcement_history
            GROUP BY category
            ORDER BY "Count" DESC
        """)
        cat_df = pd.DataFrame(cursor.fetchall(), columns=["Category", "Count"])

        # Priority distribution DataFrame
        cursor.execute("""
            SELECT priority AS "Priority", COUNT(*) AS "Count"
            FROM announcement_history
            GROUP BY priority
            ORDER BY "Count" DESC
        """)
        prio_df = pd.DataFrame(cursor.fetchall(), columns=["Priority", "Count"])

        # Language distribution DataFrame
        cursor.execute("""
            SELECT language_detected AS "Language", COUNT(*) AS "Count"
            FROM announcement_history
            GROUP BY language_detected
            ORDER BY "Count" DESC
        """)
        lang_df = pd.DataFrame(cursor.fetchall(), columns=["Language", "Count"])

        return {
            "total_count": total_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "avg_confidence": round(float(avg_conf), 4),
            "category_counts": cat_df,
            "priority_counts": prio_df,
            "language_counts": lang_df
        }


def clear_history():
    """Clear all records from announcement_history table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM announcement_history")
        conn.commit()
