from django.conf import settings
from django.db import models
from submissions.models import Submission


class ConsentRecord(models.Model):
    """
    Tracks whether a submission is allowed for research use.
    """
    submission = models.OneToOneField(Submission, on_delete=models.CASCADE, related_name="consent")
    consented = models.BooleanField(default=False)
    consent_source = models.CharField(max_length=200, blank=True)  # e.g., "form v1", "IRB protocol X"
    notes = models.TextField(blank=True)

    recorded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="consent_records")
    recorded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.submission_id} — consented={self.consented}"


class AuditLog(models.Model):
    class Action(models.TextChoices):
        IMPORT = "IMPORT", "Import"
        EXPORT = "EXPORT", "Export"
        VIEW = "VIEW", "View"
        UPDATE = "UPDATE", "Update"
        DELETE = "DELETE", "Delete"
        RUN_ANALYSIS = "RUN_ANALYSIS", "Run analysis"

    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="audit_logs")
    action = models.CharField(max_length=20, choices=Action.choices)

    entity_type = models.CharField(max_length=100, blank=True)  # "Submission", "RubricScore", etc.
    entity_id = models.CharField(max_length=100, blank=True)

    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["action", "created_at"])]

    def __str__(self) -> str:
        return f"{self.actor} — {self.action} — {self.entity_type}:{self.entity_id}"


import uuid
from django.conf import settings
from django.db import models
from courses.models import Assignment


class ImportBatch(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name="import_batches")
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="import_batches")

    original_filename = models.CharField(max_length=255, blank=True)
    column_map = models.JSONField(default=dict, blank=True)  # stores user mapping
    stats = models.JSONField(default=dict, blank=True)       # created, skipped, errors
    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Import {self.id} — {self.assignment.title}"
