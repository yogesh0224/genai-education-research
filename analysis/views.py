# analysis/views.py

import csv
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.db.models import Prefetch

from accounts.decorators import role_required
from analysis.models import (
    AnalysisRun,
    ClusteringResult,
    ModelArtifact,
    SubmissionFeatures,
)
from analysis.training import train_regression_models
from rubrics.aggregation import aggregate_rubric_scores
from rubrics.models import Rubric, RubricCriterion
from analysis.models import ClusterProfile, ClusterProfile, SubmissionFeatures
from collections import defaultdict

def _to_media_url(file_path: str) -> str:
    """
    Convert an absolute or relative file path to a URL under MEDIA_URL if possible.
    Returns "" if it cannot be mapped.
    """
    if not file_path:
        return ""

    try:
        p = Path(file_path)

        # If already a relative media path like "artifacts/<run>/file.png"
        if not p.is_absolute():
            return settings.MEDIA_URL.rstrip("/") + "/" + str(p).replace("\\", "/")

        media_root = Path(settings.MEDIA_ROOT).resolve()
        rel = p.resolve().relative_to(media_root)
        return settings.MEDIA_URL.rstrip("/") + "/" + str(rel).replace("\\", "/")
    except Exception:
        return ""

@login_required
@role_required({"ADMIN", "RESEARCHER"})
def run_dashboard(request, run_id):
    # ------------------------------------------------------------
    # Load run + rubrics
    # ------------------------------------------------------------
    run = get_object_or_404(AnalysisRun, id=run_id)
    rubrics = Rubric.objects.filter(assignment_id=run.assignment_id).order_by("id")

    # ------------------------------------------------------------
    # Determine rubric to visualize (robust)
    # ------------------------------------------------------------
    viz_rubric_id = None
    rid_param = request.GET.get("rubric_id")

    if rid_param:
        try:
            viz_rubric_id = int(rid_param)
        except ValueError:
            viz_rubric_id = None

    if not viz_rubric_id and rubrics.exists():
        viz_rubric_id = rubrics.first().id

    default_rubric_id = viz_rubric_id

    # ------------------------------------------------------------
    # Artifacts (safe for empty runs)
    # ------------------------------------------------------------
    artifact_rows = []

    artifacts = (
        ModelArtifact.objects
        .filter(analysis_run=run)
        .order_by("-created_at")
    )

    for a in artifacts:
        metrics = a.metrics or {}
        cv = metrics.get("cv_metrics", {})
        paths = metrics.get("paths", {})

        artifact_rows.append({
            "id": a.id,
            "name": a.name,
            "mae": cv.get("mae_mean"),
            "rmse": cv.get("rmse_mean"),
            "r2": cv.get("r2_mean"),
            "model_url": paths.get("model"),
            "pred_url": paths.get("pred_vs_actual"),
            "fi_url": paths.get("feature_importance"),
        })

    # ------------------------------------------------------------
    # Rubric aggregates lookup (depth, ai_use)
    # ------------------------------------------------------------
    label_map = {}
    if viz_rubric_id:
        targets = aggregate_rubric_scores(
            assignment_id=run.assignment_id,
            rubric_id=viz_rubric_id,
        )
        for t in targets:
            label_map[t["submission_id"]] = {
                "depth": t.get("depth_score_mean"),
                "ai_use": t.get("self_report_ai_use"),
            }

    # ------------------------------------------------------------
    # Centroid similarity lookup
    # ------------------------------------------------------------
    sim_lookup = {}
    for f in SubmissionFeatures.objects.filter(analysis_run=run):
        sim = (f.features or {}).get("centroid_similarity")
        if sim is not None:
            sim_lookup[str(f.submission_id)] = float(sim)

    # ------------------------------------------------------------
    # Cluster label lookup (human labels)
    # ------------------------------------------------------------
    label_lookup = {}
    if viz_rubric_id:
        for cp in ClusterProfile.objects.filter(
            analysis_run=run,
            rubric_id=viz_rubric_id
        ):
            label_lookup[int(cp.cluster)] = cp.label

    # ------------------------------------------------------------
    # Build cluster points for UMAP
    # ------------------------------------------------------------
    cluster_points = []

    clusters_qs = ClusteringResult.objects.filter(
        analysis_run=run,
        rubric_id=viz_rubric_id
    ).order_by("cluster", "submission_id")

    for c in clusters_qs:
        sid = str(c.submission_id)
        info = label_map.get(sid, {})

        cluster_points.append({
            "submission_id": sid,
            "student": c.student_anon_id,
            "cluster": int(c.cluster),
            "cluster_label": label_lookup.get(int(c.cluster), f"Cluster {c.cluster}"),
            "x": c.umap_x,
            "y": c.umap_y,
            "depth": info.get("depth"),
            "ai_use": info.get("ai_use"),
            "centroid_similarity": sim_lookup.get(sid),
        })

    # ------------------------------------------------------------
    # Cluster summary (counts + means)
    # ------------------------------------------------------------
    summary = {}

    for p in cluster_points:
        c = p["cluster"]
        summary.setdefault(c, {
            "cluster": c,
            "label": p["cluster_label"],
            "count": 0,
            "students": set(),
            "depth_vals": [],
            "ai_vals": [],
            "sim_vals": [],
        })

        summary[c]["count"] += 1
        summary[c]["students"].add(p["student"])

        if p.get("depth") is not None:
            summary[c]["depth_vals"].append(float(p["depth"]))

        if p.get("ai_use") is not None:
            try:
                summary[c]["ai_vals"].append(float(p["ai_use"]))
            except Exception:
                pass

        if p.get("centroid_similarity") is not None:
            summary[c]["sim_vals"].append(float(p["centroid_similarity"]))

    cluster_summary = []
    for c in sorted(summary.keys()):
        item = summary[c]

        cluster_summary.append({
            "cluster": item["cluster"],
            "label": item["label"],
            "count": item["count"],
            "unique_students": len(item["students"]),
            "depth_mean": (
                sum(item["depth_vals"]) / len(item["depth_vals"])
                if item["depth_vals"] else None
            ),
            "ai_use_mean": (
                sum(item["ai_vals"]) / len(item["ai_vals"])
                if item["ai_vals"] else None
            ),
            "sim_mean": (
                sum(item["sim_vals"]) / len(item["sim_vals"])
                if item["sim_vals"] else None
            ),
        })

    # ------------------------------------------------------------
    # Render (ALL variables ALWAYS defined)
    # ------------------------------------------------------------
    return render(
        request,
        "analysis/run_dashboard.html",
        {
            "run": run,
            "rubrics": rubrics,
            "viz_rubric_id": viz_rubric_id,
            "default_rubric_id": default_rubric_id,
            "artifact_rows": artifact_rows,
            "cluster_points": cluster_points,
            "cluster_summary": cluster_summary,
        }
    )


@login_required
@role_required({"ADMIN", "RESEARCHER"})
def retrain_run(request, run_id):
    if request.method != "POST":
        return redirect("analysis:run_dashboard", run_id=run_id)

    try:
        run = get_object_or_404(
            AnalysisRun.objects.select_related("assignment", "assignment__course"),
            id=run_id
        )
    except Exception:
        run = get_object_or_404(AnalysisRun, id=run_id)

    rubric_id_raw = request.POST.get("rubric_id")
    target = (request.POST.get("target") or "depth_score_mean").strip()

    if not rubric_id_raw:
        messages.error(request, "Missing rubric_id.")
        return redirect("analysis:run_dashboard", run_id=run_id)

    rubric_id = int(rubric_id_raw)

    if not Rubric.objects.filter(id=rubric_id, assignment_id=run.assignment_id).exists():
        messages.error(request, "Selected rubric does not belong to this run's assignment.")
        return redirect("analysis:run_dashboard", run_id=run_id)

    # Optional knobs
    n_splits = int(request.POST.get("n_splits") or 5)
    random_state = int(request.POST.get("random_state") or 42)

    try:
        run.status = AnalysisRun.Status.RUNNING
        run.error_message = ""
        run.save(update_fields=["status", "error_message"])

        train_regression_models(
            run=run,
            rubric_id=rubric_id,
            target=target,
            n_splits=n_splits,
            random_state=random_state,
        )

        run.status = AnalysisRun.Status.DONE
        run.save(update_fields=["status"])
        messages.success(request, f"Retraining complete. Target={target}, Rubric={rubric_id}")

    except Exception as e:
        run.status = AnalysisRun.Status.FAILED
        run.error_message = str(e)
        run.save(update_fields=["status", "error_message"])
        messages.error(request, f"Retraining failed: {e}")

    return redirect("analysis:run_dashboard", run_id=run_id)


@login_required
@role_required({"ADMIN", "RESEARCHER"})
def export_joined_ml_dataset(request, run_id):
    """
    Joined export: rubric aggregated targets + SubmissionFeatures.features for this run.
    URL: /export/joined/<run_uuid>.csv?rubric_id=2
    """
    run = get_object_or_404(AnalysisRun, id=run_id)

    rubric_id_raw = request.GET.get("rubric_id")
    if not rubric_id_raw:
        return HttpResponse("Missing rubric_id query param, e.g. ?rubric_id=2", status=400)
    rubric_id = int(rubric_id_raw)

    if not Rubric.objects.filter(id=rubric_id, assignment_id=run.assignment_id).exists():
        return HttpResponse("Rubric does not belong to this assignment.", status=400)

    # Targets
    targets = aggregate_rubric_scores(assignment_id=run.assignment_id, rubric_id=rubric_id)
    target_map = {t["submission_id"]: t for t in targets}

    criteria = list(RubricCriterion.objects.filter(rubric_id=rubric_id).order_by("order", "id"))
    criterion_cols = [f"crit_{c.id}_mean" for c in criteria]

    base_cols = [
        "submission_id", "assignment_id", "rubric_id", "student_anon_id",
        "submitted_at", "self_report_ai_use", "word_count",
        "total_score_mean", "total_score_weighted_mean", "depth_score_mean",
    ]
    target_cols = base_cols + criterion_cols

    feats_qs = SubmissionFeatures.objects.filter(analysis_run=run).order_by("created_at")

    # Choose feature keys from first row (assumes consistent keys)
    first = feats_qs.first()
    feature_keys = sorted((first.features or {}).keys()) if first else []
    feature_cols = [f"feat_{k}" for k in feature_keys]

    fieldnames = target_cols + feature_cols

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="joined_ml_run_{run_id}_rubric_{rubric_id}.csv"'

    writer = csv.DictWriter(response, fieldnames=fieldnames)
    writer.writeheader()

    written = 0
    for f in feats_qs:
        sid = str(f.submission_id)
        t = target_map.get(sid)
        if not t:
            continue

        row = {k: t.get(k, "") for k in target_cols}
        feats = f.features or {}
        for k in feature_keys:
            row[f"feat_{k}"] = feats.get(k, "")

        writer.writerow(row)
        written += 1

    return response


@login_required
@role_required({"ADMIN", "RESEARCHER"})
def export_cluster_labels_csv(request, run_id):
    """
    Export per-submission clustering output + labels + rubric aggregates.

    URL:
      /export/clusters/<run_uuid>.csv?rubric_id=2
    """
    run = get_object_or_404(AnalysisRun, id=run_id)

    rubric_id_raw = request.GET.get("rubric_id")
    if not rubric_id_raw:
        return HttpResponse("Missing rubric_id query param, e.g. ?rubric_id=2", status=400)
    rubric_id = int(rubric_id_raw)

    # Cluster label lookup
    label_lookup = {}
    for cp in ClusterProfile.objects.filter(analysis_run=run, rubric_id=rubric_id):
        label_lookup[int(cp.cluster)] = cp.label

    # Rubric aggregates lookup (depth, ai_use, totals)
    targets = aggregate_rubric_scores(assignment_id=run.assignment_id, rubric_id=rubric_id)
    target_map = {t["submission_id"]: t for t in targets}

    # Centroid similarity lookup from SubmissionFeatures
    sim_lookup = {}
    for f in SubmissionFeatures.objects.filter(analysis_run=run):
        sim = (f.features or {}).get("centroid_similarity")
        if sim is not None:
            sim_lookup[str(f.submission_id)] = float(sim)

    # Clustering rows
    qs = ClusteringResult.objects.filter(analysis_run=run, rubric_id=rubric_id).order_by("cluster", "submission_id")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="clusters_{run_id}_rubric_{rubric_id}.csv"'

    fieldnames = [
        "submission_id",
        "student_anon_id",
        "cluster",
        "cluster_label",
        "umap_x",
        "umap_y",
        "centroid_similarity",
        "depth_score_mean",
        "total_score_mean",
        "total_score_weighted_mean",
        "self_report_ai_use",
        "word_count",
        "submitted_at",
    ]
    writer = csv.DictWriter(response, fieldnames=fieldnames)
    writer.writeheader()

    for c in qs:
        sid = str(c.submission_id)
        t = target_map.get(sid, {})

        writer.writerow({
            "submission_id": sid,
            "student_anon_id": c.student_anon_id,
            "cluster": c.cluster,
            "cluster_label": label_lookup.get(int(c.cluster), f"Cluster {c.cluster}"),
            "umap_x": c.umap_x,
            "umap_y": c.umap_y,
            "centroid_similarity": sim_lookup.get(sid, ""),
            "depth_score_mean": t.get("depth_score_mean", ""),
            "total_score_mean": t.get("total_score_mean", ""),
            "total_score_weighted_mean": t.get("total_score_weighted_mean", ""),
            "self_report_ai_use": t.get("self_report_ai_use", ""),
            "word_count": t.get("word_count", ""),
            "submitted_at": t.get("submitted_at", ""),
        })

    return response


@login_required
@role_required({"ADMIN", "RESEARCHER"})
def compare_runs(request):
    """
    Compare runs (optionally filtered by assignment_id) and highlight best run
    for each (model, target) by highest CV r2_mean.
    """
    assignment_id = request.GET.get("assignment_id")

    runs_qs = AnalysisRun.objects.all().order_by("-created_at")
    if assignment_id:
        try:
            runs_qs = runs_qs.filter(assignment_id=int(assignment_id))
        except ValueError:
            assignment_id = ""

    runs = list(runs_qs[:30])
    run_ids = [r.id for r in runs]

    artifacts_qs = (
        ModelArtifact.objects
        .filter(analysis_run_id__in=run_ids)
        .order_by("-created_at")
    )

    # Latest artifact per run per (model,target)
    by_run = {rid: {} for rid in run_ids}
    for a in artifacts_qs:
        m = a.metrics or {}
        model = m.get("model")
        target = m.get("target")
        if not model or not target:
            # fallback from "ridge:depth_score_mean"
            if ":" in a.name:
                model = model or a.name.split(":")[0]
                target = target or a.name.split(":")[1]
        if not model or not target:
            continue

        key = (model, target)
        if key not in by_run[a.analysis_run_id]:
            by_run[a.analysis_run_id][key] = a

    # Compute best artifact per (model,target) across runs by max r2_mean
    best_by_key = {}
    for rid, amap in by_run.items():
        for key, a in amap.items():
            cv = (a.metrics or {}).get("cv_metrics") or {}
            r2_mean = cv.get("r2_mean")
            if r2_mean is None:
                continue
            if key not in best_by_key or r2_mean > best_by_key[key]["r2_mean"]:
                best_by_key[key] = {"run_id": rid, "r2_mean": r2_mean}

    # Build rows for template
    rows = []
    for r in runs:
        amap = by_run.get(r.id, {})
        artifacts_view = []
        for (model, target), a in sorted(amap.items(), key=lambda x: (x[0][0], x[0][1])):
            metrics = a.metrics or {}
            cv = metrics.get("cv_metrics") or {}

            key = (model, target)
            is_best = (best_by_key.get(key, {}).get("run_id") == r.id)

            artifacts_view.append({
                "model": model,
                "target": target,
                "r2_mean": cv.get("r2_mean"),
                "r2_std": cv.get("r2_std"),
                "rmse_mean": cv.get("rmse_mean"),
                "rmse_std": cv.get("rmse_std"),
                "mae_mean": cv.get("mae_mean"),
                "mae_std": cv.get("mae_std"),
                "created_at": a.created_at,
                "is_best": is_best,
            })

        assignment_display = str(getattr(r, "assignment", None) or f"Assignment {r.assignment_id}")

        rows.append({
            "run": r,
            "assignment_display": assignment_display,
            "artifacts": artifacts_view,
        })
        # ---- Build plot data for Plotly ----
    plot_points = []
    for row in rows:
        r = row["run"]

        # Prefer created_at; fallback to started_at
        ts = getattr(r, "created_at", None) or getattr(r, "started_at", None)
        if not ts:
            continue

        for a in row["artifacts"]:
            # Only plot if we have r2_mean
            if a.get("r2_mean") is None:
                continue

            plot_points.append({
                "run_id": str(r.id),
                "timestamp": ts.isoformat(),
                "model": a["model"],
                "target": a["target"],
                "series": f'{a["model"]}:{a["target"]}',
                "r2_mean": a["r2_mean"],
                "r2_std": a.get("r2_std"),
            })


    return render(request, "analysis/compare_runs.html", {
        "rows": rows,
        "assignment_id": assignment_id or "",
        "plot_points":plot_points,
    })
