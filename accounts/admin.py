from django.contrib import admin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "institution", "department", "created_at")
    list_filter = ("role", "institution")
    search_fields = ("user__username", "user__email", "institution", "department")
