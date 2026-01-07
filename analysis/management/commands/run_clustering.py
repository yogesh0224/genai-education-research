from django.core.management.base import BaseCommand
from analysis.clustering import run_umap_clustering
from analysis.models import AnalysisRun, ClusteringResult


class Command(BaseCommand):
    help = "Run UMAP + clustering for an analysis run"

    def add_arguments(self, parser):
        parser.add_argument("run_id")
        parser.add_argument("--rubric_id", type=int, required=True)
        parser.add_argument("--clusters", type=int, default=3)

    def handle(self, *args, **opts):
        run = AnalysisRun.objects.get(id=opts["run_id"])

        df = run_umap_clustering(
            run_id=run.id,
            rubric_id=opts["rubric_id"],
            n_clusters=opts["clusters"],
        )

        ClusteringResult.objects.filter(
            analysis_run=run,
            rubric_id=opts["rubric_id"]
        ).delete()

        objs = []
        for _, r in df.iterrows():
            objs.append(
                ClusteringResult(
                    analysis_run=run,
                    rubric_id=opts["rubric_id"],
                    submission_id=r["submission_id"],
                    student_anon_id=r["student_anon_id"],
                    umap_x=r["umap_x"],
                    umap_y=r["umap_y"],
                    cluster=int(r["cluster"]),
                )
            )

        ClusteringResult.objects.bulk_create(objs)
        self.stdout.write(self.style.SUCCESS(f"Saved {len(objs)} clustering rows"))
