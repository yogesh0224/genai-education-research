from django.contrib import admin
from .models import Course, Assignment, Policy


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ("name", "term", "institution", "created_at")
    list_filter = ("term", "institution")
    search_fields = ("name", "term", "institution")


@admin.register(Assignment)
class AssignmentAdmin(admin.ModelAdmin):
    list_display = ("title", "course", "due_date", "requires_sources", "min_words", "created_at")
    list_filter = ("course", "requires_sources")
    search_fields = ("title", "course__name")


@admin.register(Policy)
class PolicyAdmin(admin.ModelAdmin):
    list_display = ("assignment", "ai_allowed", "ai_disclosure_required", "updated_at")
    list_filter = ("ai_allowed", "ai_disclosure_required")
    search_fields = ("assignment__title", "assignment__course__name")
