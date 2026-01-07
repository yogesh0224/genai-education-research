from django.contrib import admin
from .models import Student, Submission


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("anon_id", "cohort", "created_at")
    list_filter = ("cohort",)
    search_fields = ("anon_id",)


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ("id", "assignment", "student", "self_report_ai_use", "submitted_at", "created_at")
    list_filter = ("assignment", "self_report_ai_use")
    search_fields = ("id", "student__anon_id", "assignment__title", "assignment__course__name")
    readonly_fields = ("id", "created_at")
