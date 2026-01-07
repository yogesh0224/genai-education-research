from collections import defaultdict
from typing import Dict, Any, List

from django.db.models import Avg, Count
from submissions.models import Submission
from .models import Rubric, RubricCriterion, RubricScore


def aggregate_rubric_scores(assignment_id: int, rubric_id: int) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, each dict represents one submission with aggregated rubric scores.
    """
    rubric = Rubric.objects.select_related("assignment").get(id=rubric_id, assignment_id=assignment_id)
    criteria = list(RubricCriterion.objects.filter(rubric=rubric).order_by("order", "id"))

    # Precompute which criteria are depth
    depth_criteria_ids = {c.id for c in criteria if getattr(c, "is_depth", False)}

    # Aggregate mean score per submission per criterion
    # This produces rows like: submission_id, criterion_id, mean_score, rater_count
    agg = (
        RubricScore.objects
        .filter(submission__assignment_id=assignment_id, criterion__rubric_id=rubric_id)
        .values("submission_id", "criterion_id")
        .annotate(mean_score=Avg("score"), rater_count=Count("id"))
    )

    # Organize into submission -> criterion -> mean
    by_submission = defaultdict(dict)
    for row in agg:
        by_submission[row["submission_id"]][row["criterion_id"]] = float(row["mean_score"])

    # Pull submissions
    submissions = (
        Submission.objects
        .filter(assignment_id=assignment_id)
        .select_related("student")
        .order_by("created_at")
    )

    out = []
    for s in submissions:
        crit_means = by_submission.get(s.id, {})

        # Build per-criterion outputs
        per_criterion = {}
        for c in criteria:
            per_criterion[f"crit_{c.id}_mean"] = crit_means.get(c.id)

        # Compute totals
        # Unweighted mean over available criterion means
        available_scores = [crit_means.get(c.id) for c in criteria if crit_means.get(c.id) is not None]
        total_score_mean = sum(available_scores) / len(available_scores) if available_scores else None

        # Weighted mean
        weighted_sum = 0.0
        weight_total = 0.0
        for c in criteria:
            v = crit_means.get(c.id)
            if v is None:
                continue
            weighted_sum += v * float(c.weight)
            weight_total += float(c.weight)
        total_score_weighted_mean = (weighted_sum / weight_total) if weight_total > 0 else None

        # Depth subscore mean (unweighted over depth criteria)
        depth_vals = [crit_means.get(cid) for cid in depth_criteria_ids if crit_means.get(cid) is not None]
        depth_score_mean = (sum(depth_vals) / len(depth_vals)) if depth_vals else None

        # Minimal covariates
        word_count = len((s.text or "").split())

        out.append({
            "submission_id": str(s.id),
            "assignment_id": s.assignment_id,
            "rubric_id": rubric_id,
            "student_anon_id": s.student.anon_id,
            "submitted_at": s.submitted_at.isoformat() if s.submitted_at else "",
            "self_report_ai_use": s.self_report_ai_use,
            "word_count": word_count,
            "total_score_mean": total_score_mean,
            "total_score_weighted_mean": total_score_weighted_mean,
            "depth_score_mean": depth_score_mean,
            **per_criterion,
        })

    return out
