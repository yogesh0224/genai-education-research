from django.db import models


class Course(models.Model):
    name = models.CharField(max_length=200)
    term = models.CharField(max_length=100, blank=True)  # e.g., "Fall 2025"
    institution = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("name", "term", "institution")]
        ordering = ["-created_at", "name"]

    def __str__(self) -> str:
        parts = [self.name]
        if self.term:
            parts.append(self.term)
        if self.institution:
            parts.append(self.institution)
        return " — ".join(parts)


class Assignment(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="assignments")

    title = models.CharField(max_length=200)
    prompt_text = models.TextField(blank=True)
    due_date = models.DateField(null=True, blank=True)

    requires_sources = models.BooleanField(default=False)
    min_words = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("course", "title")]
        ordering = ["-created_at", "course__name", "title"]

    def __str__(self) -> str:
        return f"{self.course.name}: {self.title}"


class Policy(models.Model):
    class AIPolicy(models.TextChoices):
        NOT_ALLOWED = "NOT_ALLOWED", "Not allowed"
        ASSIST_ALLOWED = "ASSIST_ALLOWED", "Assist allowed (editing, brainstorming)"
        FULL_ALLOWED = "FULL_ALLOWED", "Full allowed (generation permitted)"

    assignment = models.OneToOneField(Assignment, on_delete=models.CASCADE, related_name="policy")

    ai_allowed = models.CharField(max_length=20, choices=AIPolicy.choices, default=AIPolicy.ASSIST_ALLOWED)
    ai_disclosure_required = models.BooleanField(default=True)
    notes = models.TextField(blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Policy for {self.assignment}"
