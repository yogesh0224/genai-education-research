from django.contrib import admin
from .models import AnalysisRun, SubmissionFeatures, ModelArtifact,ClusterProfile


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("id", "assignment", "status", "created_by", "created_at", "started_at", "finished_at")
    list_filter = ("status", "assignment")
    search_fields = ("id", "assignment__title", "assignment__course__name", "created_by__username")
    readonly_fields = ("id", "created_at")


@admin.register(SubmissionFeatures)
class SubmissionFeaturesAdmin(admin.ModelAdmin):
    list_display = ("analysis_run", "submission", "ai_similarity", "cluster_id", "predicted_ai_assist_prob", "created_at")
    list_filter = ("analysis_run", "cluster_id")
    search_fields = ("submission__student__anon_id", "submission__assignment__title")


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = ("analysis_run", "type", "name", "created_at")
    list_filter = ("type", "analysis_run")
    search_fields = ("name", "analysis_run__id")

@admin.register(ClusterProfile)
class ClusterProfileAdmin(admin.ModelAdmin):
    list_display = ("analysis_run", "rubric_id", "cluster", "label", "created_at")
    list_filter = ("rubric_id", "label")
    search_fields = ("analysis_run__id", "label", "description")