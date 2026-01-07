from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    class Role(models.TextChoices):
        ADMIN = "ADMIN", "Admin"
        RESEARCHER = "RESEARCHER", "Researcher"
        RATER = "RATER", "Rater"
        VIEWER = "VIEWER", "Viewer"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.VIEWER)

    institution = models.CharField(max_length=200, blank=True)
    department = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.user.username} ({self.role})"
