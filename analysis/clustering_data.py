# analysis/clustering_data.py

import pandas as pd
from analysis.training_data import build_training_frame


def build_clustering_frame(run_id, rubric_id):
    """
    Returns a dataframe for unsupervised learning:
    - excludes target y
    - keeps student_anon_id for grouping
    """
    df = build_training_frame(
        run_id=run_id,
        rubric_id=rubric_id,
        require_target=False,
    )

    if df.empty:
        return df

    # Remove supervised targets
    drop_cols = {"y"}
    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    keep_cols = [
        "submission_id",
        "student_anon_id",
        "word_count",
        "self_report_ai_use",
    ] + feature_cols

    return df[keep_cols]
