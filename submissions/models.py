import uuid
from django.db import models
from courses.models import Assignment


class Student(models.Model):
    """
    IMPORTANT: Store only anonymized IDs here. Never store names/emails.
    """
    anon_id = models.CharField(max_length=64, unique=True)  # e.g., "S001", hashed ID, etc.
    cohort = models.CharField(max_length=100, blank=True)   # optional grouping (section, batch)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.anon_id


class Submission(models.Model):
    class AISelfReport(models.IntegerChoices):
        UNKNOWN = -1, "Unknown"
        NONE = 0, "None"
        LOW = 1, "Low"
        MEDIUM = 2, "Medium"
        HIGH = 3, "High"
        FULL = 4, "Full / Heavy use"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name="submissions")
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="submissions")

    text = models.TextField()
    submitted_at = models.DateTimeField(null=True, blank=True)

    self_report_ai_use = models.IntegerField(choices=AISelfReport.choices, default=AISelfReport.UNKNOWN)
    ai_disclosure_text = models.TextField(blank=True)  # if your policy requires disclosure
    prompt_used = models.TextField(blank=True)  # optional, only if collected ethically

    # metadata for logs, revision counts, word counts, etc.
    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["assignment", "created_at"]),
            models.Index(fields=["student"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.student.anon_id} — {self.assignment.title}"
