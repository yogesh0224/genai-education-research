# analysis/training_data.py

from __future__ import annotations

from typing import Optional, Any
import pandas as pd

from analysis.models import AnalysisRun, SubmissionFeatures
from rubrics.aggregation import aggregate_rubric_scores


def _extract_submission_text(submission: Any) -> str:
    """
    Best-effort extraction of a submission's answer text.
    We try common field names; if none exist, return empty string.

    Update this list if your Submission model uses a different field name.
    """
    if submission is None:
        return ""

    candidate_fields = [
        "text",
        "answer_text",
        "answer",
        "response",
        "response_text",
        "content",
        "body",
        "submission_text",
        "student_answer",
        "raw_text",
    ]

    for field in candidate_fields:
        if hasattr(submission, field):
            val = getattr(submission, field)
            # Skip callables (methods/properties that need args)
            if callable(val):
                continue
            if val is None:
                return ""
            return str(val)

    # Nothing found
    return ""


def build_training_frame(
    run_id,
    rubric_id: int,
    target_col: str = "depth_score_mean",
    require_target: bool = True,
) -> pd.DataFrame:
    """
    Returns a dataframe joined on submission_id:
      - Targets from aggregate_rubric_scores(...)
      - Feature dict from SubmissionFeatures.features
      - NEW: submission raw text column for TF-IDF baseline

    Columns include:
      - submission_id, student_anon_id, y, group
      - text (string, may be empty)
      - feat_* columns

    Notes:
      - `group` is used for GroupKFold splitting (student-level grouping).
      - Features are flattened from JSON into columns prefixed with `feat_`.
      - Rows with missing target are dropped if require_target=True.
    """
    run = AnalysisRun.objects.get(id=run_id)

    # 1) Load rubric targets (labels)
    targets = aggregate_rubric_scores(assignment_id=run.assignment_id, rubric_id=rubric_id)

    # Map by submission_id (string UUID)
    target_map = {t["submission_id"]: t for t in targets}

    # 2) Load features for this run (+ submission object)
    feats_qs = (
        SubmissionFeatures.objects
        .filter(analysis_run=run)
        .select_related("submission")
        .order_by("created_at")
    )

    rows = []
    text_map = {}  # submission_id -> submission text

    for f in feats_qs:
        sid = str(f.submission_id)

        # Build text map from the related submission (if present)
        if sid not in text_map:
            try:
                text_map[sid] = _extract_submission_text(getattr(f, "submission", None))
            except Exception:
                text_map[sid] = ""

        t = target_map.get(sid)
        if not t:
            # features exist but no labels -> skip
            continue

        # Target value
        y_val = t.get(target_col)

        # If we require target and it's missing -> skip row
        if require_target and (y_val is None or y_val == ""):
            continue

        # Base row
        row = {
            "submission_id": sid,
            "assignment_id": t.get("assignment_id", run.assignment_id),
            "rubric_id": t.get("rubric_id", rubric_id),
            "student_anon_id": t.get("student_anon_id", ""),
            "submitted_at": t.get("submitted_at", ""),
            "self_report_ai_use": t.get("self_report_ai_use", ""),
            "word_count": t.get("word_count", ""),
            "y": float(y_val) if y_val not in (None, "") else None,
        }

        # Group key for GroupKFold (student-level splitting)
        row["group"] = row["student_anon_id"]

        # Flatten features dict into feat_* columns
        feats = f.features or {}
        for k, v in feats.items():
            row[f"feat_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # 2.5) NEW: attach text column using submission_id -> text mapping
    df["text"] = df["submission_id"].map(text_map).fillna("").astype(str)

    # 3) Clean up types
    # Ensure y numeric
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Basic numeric columns if present
    for col in ["word_count", "self_report_ai_use"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert all feat_* columns to numeric where possible (keeps NaN for non-numeric)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing y if require_target
    if require_target:
        df = df.dropna(subset=["y"])

    # Drop rows missing group (shouldn't happen, but just in case)
    df = df[df["group"].notna() & (df["group"].astype(str).str.len() > 0)]

    return df
