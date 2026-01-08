import json
import csv
from pathlib import Path
from itertools import combinations

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

from analysis.models import (
    AnalysisRun,
    ClusteringResult,
    ClusterProfile,
)
from rubrics.aggregation import aggregate_rubric_scores


class Command(BaseCommand):
    help = "Run statistical tests comparing rubric depth across clusters"

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=str)
        parser.add_argument("--rubric_id", type=int, required=True)
        parser.add_argument("--target", type=str, default="depth_score_mean")

    def handle(self, *args, **opts):
        run_id = opts["run_id"]
        rubric_id = opts["rubric_id"]
        target = opts["target"]

        try:
            run = AnalysisRun.objects.get(id=run_id)
        except AnalysisRun.DoesNotExist:
            raise CommandError(f"AnalysisRun not found: {run_id}")

        # ---- Load cluster labels ----
        label_lookup = {}
        for cp in ClusterProfile.objects.filter(analysis_run=run, rubric_id=rubric_id):
            label_lookup[int(cp.cluster)] = cp.label

        if not label_lookup:
            raise CommandError("No ClusterProfile labels found. Label clusters first.")

        # ---- Load rubric aggregates ----
        rows = aggregate_rubric_scores(
            assignment_id=run.assignment_id,
            rubric_id=rubric_id,
        )

        row_map = {r["submission_id"]: r for r in rows}

        # ---- Collect values by cluster ----
        cluster_values = {}
        for c in ClusteringResult.objects.filter(
            analysis_run=run,
            rubric_id=rubric_id,
        ):
            sid = str(c.submission_id)
            r = row_map.get(sid)
            if not r:
                continue

            val = r.get(target)
            if val is None:
                continue

            cluster_values.setdefault(c.cluster, []).append(float(val))

        if len(cluster_values) < 2:
            raise CommandError("Need at least 2 clusters with data")

        # ---- Descriptive stats ----
        desc = {}
        for k, vals in cluster_values.items():
            desc[k] = {
                "label": label_lookup.get(k, f"Cluster {k}"),
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

        # ---- Kruskal–Wallis ----
        kw_stat, kw_p = kruskal(*cluster_values.values())

        results = {
            "run_id": run_id,
            "rubric_id": rubric_id,
            "target": target,
            "kruskal": {
                "H": float(kw_stat),
                "p_value": float(kw_p),
            },
            "descriptive": desc,
            "pairwise": [],
        }

        # ---- Pairwise Mann–Whitney (if significant) ----
        clusters = list(cluster_values.keys())
        alpha = 0.05
        m = len(list(combinations(clusters, 2)))
        alpha_corr = alpha / m

        for a, b in combinations(clusters, 2):
            va = cluster_values[a]
            vb = cluster_values[b]

            u, p = mannwhitneyu(va, vb, alternative="two-sided")

            results["pairwise"].append({
                "cluster_a": a,
                "label_a": label_lookup.get(a),
                "cluster_b": b,
                "label_b": label_lookup.get(b),
                "u_stat": float(u),
                "p_value": float(p),
                "significant_bonferroni": bool(p < alpha_corr),
            })

        # ---- Write artifacts ----
        out_dir = Path(settings.MEDIA_ROOT) / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"cluster_stats_{run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        csv_path = out_dir / f"cluster_stats_{run_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cluster",
                "label",
                "n",
                "mean",
                "median",
                "std",
            ])
            for k, d in desc.items():
                writer.writerow([
                    k,
                    d["label"],
                    d["n"],
                    d["mean"],
                    d["median"],
                    d["std"],
                ])

        self.stdout.write(self.style.SUCCESS("Cluster statistics computed"))
        self.stdout.write(f"JSON: {json_path}")
        self.stdout.write(f"CSV:  {csv_path}")
