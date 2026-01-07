
import csv

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from accounts.decorators import role_required
from audit.models import AuditLog
from courses.models import Assignment
from submissions.models import Submission
from .aggregation import aggregate_rubric_scores
from .forms import RubricScoreForm
from .models import Rubric, RubricCriterion, RubricScore


@login_required
@role_required({"ADMIN", "RESEARCHER", "RATER"})
def assignment_select(request):
    assignments = Assignment.objects.all().select_related("course")
    return render(request, "rubrics/assignment_select.html", {"assignments": assignments})


@login_required
@role_required({"ADMIN", "RESEARCHER", "RATER"})
def rubric_select(request, assignment_id: int):
    assignment = get_object_or_404(Assignment, id=assignment_id)
    rubrics = Rubric.objects.filter(assignment=assignment)
    return render(
        request,
        "rubrics/rubric_select.html",
        {"assignment": assignment, "rubrics": rubrics},
    )


@login_required
@role_required({"ADMIN", "RESEARCHER", "RATER"})
def scoring_queue(request, assignment_id: int, rubric_id: int):
    assignment = get_object_or_404(Assignment, id=assignment_id)
    rubric = get_object_or_404(Rubric, id=rubric_id, assignment=assignment)

    criteria_count = RubricCriterion.objects.filter(rubric=rubric).count()

    # Count how many criteria THIS rater has scored for each submission
    scored_counts = (
        RubricScore.objects.filter(
            rater=request.user,
            criterion__rubric=rubric,
            submission__assignment=assignment,
        )
        .values("submission_id")
        .annotate(cnt=Count("id"))
    )
    scored_map = {row["submission_id"]: row["cnt"] for row in scored_counts}

    submissions = (
        Submission.objects.filter(assignment=assignment)
        .select_related("student")
        .order_by("created_at")
    )

    queue = []
    remaining = 0
    for s in submissions:
        cnt = scored_map.get(s.id, 0)
        done = (criteria_count > 0) and (cnt >= criteria_count)
        if not done:
            remaining += 1
        queue.append(
            {
                "submission": s,
                "done": done,
                "count": cnt,
                "total": criteria_count,
            }
        )

    next_submission = next((x["submission"] for x in queue if not x["done"]), None)

    return render(
        request,
        "rubrics/scoring_queue.html",
        {
            "assignment": assignment,
            "rubric": rubric,
            "queue": queue,
            "remaining": remaining,
            "next_submission": next_submission,
        },
    )


@login_required
@role_required({"ADMIN", "RESEARCHER", "RATER"})
def score_submission(request, assignment_id: int, rubric_id: int, submission_id):
    assignment = get_object_or_404(Assignment, id=assignment_id)
    rubric = get_object_or_404(Rubric, id=rubric_id, assignment=assignment)
    submission = get_object_or_404(Submission, id=submission_id, assignment=assignment)

    criteria = list(
        RubricCriterion.objects.filter(rubric=rubric).order_by("order", "id")
    )

    # Existing scores (prefill)
    existing = RubricScore.objects.filter(
        submission=submission, rater=request.user, criterion__rubric=rubric
    ).select_related("criterion")

    existing_scores = {s.criterion_id: s.score for s in existing}
    existing_notes = {s.criterion_id: s.notes for s in existing}

    if request.method == "POST":
        form = RubricScoreForm(request.POST, criteria=criteria, existing_scores=existing_scores)
        if form.is_valid():
            with transaction.atomic():
                for c in criteria:
                    score_val = form.cleaned_data[f"crit_{c.id}"]
                    note_val = form.cleaned_data.get(f"note_{c.id}", "") or ""

                    RubricScore.objects.update_or_create(
                        submission=submission,
                        criterion=c,
                        rater=request.user,
                        defaults={"score": score_val, "notes": note_val},
                    )

            messages.success(request, "Scores saved.")
            return redirect(
                _next_unscored_submission_url(
                    user=request.user,
                    assignment=assignment,
                    rubric=rubric,
                    current_submission_id=submission.id,
                )
            )
    else:
        initial = {}
        for c in criteria:
            initial[f"crit_{c.id}"] = existing_scores.get(c.id)
            initial[f"note_{c.id}"] = existing_notes.get(c.id, "")
        form = RubricScoreForm(criteria=criteria, existing_scores=existing_scores, initial=initial)

    return render(
        request,
        "rubrics/score_submission.html",
        {
            "assignment": assignment,
            "rubric": rubric,
            "submission": submission,
            "criteria": criteria,
            "form": form,
        },
    )


def _next_unscored_submission_url(user, assignment: Assignment, rubric: Rubric, current_submission_id):
    criteria_count = RubricCriterion.objects.filter(rubric=rubric).count()

    scored_counts = (
        RubricScore.objects.filter(
            rater=user,
            criterion__rubric=rubric,
            submission__assignment=assignment,
        )
        .values("submission_id")
        .annotate(cnt=Count("id"))
    )
    scored_map = {row["submission_id"]: row["cnt"] for row in scored_counts}

    submission_ids = list(
        Submission.objects.filter(assignment=assignment)
        .order_by("created_at")
        .values_list("id", flat=True)
    )

    found_current = False
    for sid in submission_ids:
        if sid == current_submission_id:
            found_current = True
            continue
        if not found_current:
            continue

        cnt = scored_map.get(sid, 0)
        if criteria_count > 0 and cnt < criteria_count:
            return reverse(
                "rubrics:score_submission",
                kwargs={
                    "assignment_id": assignment.id,
                    "rubric_id": rubric.id,
                    "submission_id": sid,
                },
            )

    # Wrap-around: find first unscored (excluding current)
    for sid in submission_ids:
        if sid == current_submission_id:
            continue
        cnt = scored_map.get(sid, 0)
        if criteria_count > 0 and cnt < criteria_count:
            return reverse(
                "rubrics:score_submission",
                kwargs={
                    "assignment_id": assignment.id,
                    "rubric_id": rubric.id,
                    "submission_id": sid,
                },
            )

    return reverse(
        "rubrics:scoring_queue",
        kwargs={"assignment_id": assignment.id, "rubric_id": rubric.id},
    )


@login_required
@role_required({"ADMIN", "RESEARCHER"})
def export_ml_dataset(request, assignment_id: int, rubric_id: int):
    """
    Export an ML-ready CSV:
    - one row per submission
    - aggregated rubric targets (means, weighted totals, depth subscore)
    - minimal covariates (word_count, self_report_ai_use, submitted_at)
    - per-criterion mean columns in rubric order
    """
    rows = aggregate_rubric_scores(assignment_id=assignment_id, rubric_id=rubric_id)

    criteria = list(
        RubricCriterion.objects.filter(rubric_id=rubric_id).order_by("order", "id")
    )
    criterion_cols = [f"crit_{c.id}_mean" for c in criteria]

    base_cols = [
        "submission_id",
        "assignment_id",
        "rubric_id",
        "student_anon_id",
        "submitted_at",
        "self_report_ai_use",
        "word_count",
        "total_score_mean",
        "total_score_weighted_mean",
        "depth_score_mean",
    ]
    fieldnames = base_cols + criterion_cols

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="ml_dataset_assignment_{assignment_id}_rubric_{rubric_id}.csv"'
    )

    writer = csv.DictWriter(response, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

    AuditLog.objects.create(
        actor=request.user,
        action=AuditLog.Action.EXPORT,
        entity_type="MLDataset",
        entity_id=f"{assignment_id}:{rubric_id}",
        metadata={"rows": len(rows), "fields": len(fieldnames)},
    )

    return response
