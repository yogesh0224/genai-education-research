import uuid
from django.conf import settings
from django.db import models
from courses.models import Assignment
from submissions.models import Submission


class AnalysisRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        RUNNING = "RUNNING", "Running"
        DONE = "DONE", "Done"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name="analysis_runs")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="analysis_runs")

    config = models.JSONField(default=dict, blank=True)  # embedding model, thresholds, clustering params, etc.
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    error_message = models.TextField(blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["assignment", "created_at"])]

    def __str__(self) -> str:
        return f"Run {self.id} — {self.assignment.title} ({self.status})"


class SubmissionFeatures(models.Model):
    """
    Stores per-submission computed features for a given analysis run.
    Keep embeddings out of the DB at first if you prefer; store file path.
    """
    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="submission_features")
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE, related_name="features")

    # Store computed feature dicts (lexical, syntactic, similarity stats, etc.)
    features = models.JSONField(default=dict, blank=True)

    # Optional: store embedding as a list (ok for small-scale), or store path in metadata.
    embedding = models.JSONField(null=True, blank=True)  # e.g., list[float]; can be replaced later
    ai_similarity = models.FloatField(null=True, blank=True)

    cluster_id = models.IntegerField(null=True, blank=True)
    predicted_ai_assist_prob = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("analysis_run", "submission")]
        indexes = [
            models.Index(fields=["analysis_run"]),
            models.Index(fields=["cluster_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.analysis_run_id} — {self.submission_id}"


class ModelArtifact(models.Model):
    class ArtifactType(models.TextChoices):
        CLASSIFIER = "CLASSIFIER", "Classifier"
        REGRESSOR = "REGRESSOR", "Regressor"
        CLUSTERING = "CLUSTERING", "Clustering"
        VISUALIZATION = "VISUALIZATION", "Visualization"
        OTHER = "OTHER", "Other"

    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="artifacts")
    type = models.CharField(max_length=20, choices=ArtifactType.choices, default=ArtifactType.OTHER)

    name = models.CharField(max_length=200)
    metrics = models.JSONField(default=dict, blank=True)
    file_path = models.CharField(max_length=500, blank=True)  # store path to pickle/plot/csv
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["analysis_run", "type", "created_at"]

    def __str__(self) -> str:
        return f"{self.analysis_run_id} — {self.type} — {self.name}"


class ClusteringResult(models.Model):
    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE)
    rubric_id = models.IntegerField()
    submission_id = models.UUIDField()
    student_anon_id = models.CharField(max_length=64)

    umap_x = models.FloatField()
    umap_y = models.FloatField()
    cluster = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

class ClusterProfile(models.Model):
    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE)
    rubric_id = models.IntegerField()
    cluster = models.IntegerField()

    label = models.CharField(max_length=64)  # e.g., "AI copier"
    description = models.TextField(blank=True, default="")

    # optional: store summary stats at the time of labeling
    stats = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("analysis_run", "rubric_id", "cluster")
