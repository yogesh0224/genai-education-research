from django.conf import settings
from django.db import models
from courses.models import Assignment
from submissions.models import Submission


class Rubric(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name="rubrics")
    name = models.CharField(max_length=200)

    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("assignment", "name")]
        ordering = ["assignment__course__name", "assignment__title", "name"]

    def __str__(self) -> str:
        return f"{self.assignment.title} — {self.name}"


class RubricCriterion(models.Model):
    rubric = models.ForeignKey(Rubric, on_delete=models.CASCADE, related_name="criteria")
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    max_score = models.PositiveIntegerField(default=4)
    weight = models.FloatField(default=1.0)

    order = models.PositiveIntegerField(default=0)

    is_depth = models.BooleanField(default=False)

    class Meta:
        ordering = ["rubric", "order", "id"]
        unique_together = [("rubric", "name")]

    def __str__(self) -> str:
        return f"{self.rubric.name}: {self.name}"
    

class RubricScore(models.Model):
    """
    Each rater scores each criterion for a submission.
    """
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE, related_name="rubric_scores")
    criterion = models.ForeignKey(RubricCriterion, on_delete=models.CASCADE, related_name="scores")
    rater = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="rubric_scores")

    score = models.FloatField()
    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("submission", "criterion", "rater")]
        indexes = [
            models.Index(fields=["submission"]),
            models.Index(fields=["criterion"]),
            models.Index(fields=["rater"]),
        ]

    def __str__(self) -> str:
        return f"{self.submission} — {self.criterion.name} — {self.score}"
