from django.contrib import admin
from .models import Rubric, RubricCriterion, RubricScore


class RubricCriterionInline(admin.TabularInline):
    model = RubricCriterion
    extra = 0


@admin.register(Rubric)
class RubricAdmin(admin.ModelAdmin):
    list_display = ("name", "assignment", "created_at")
    search_fields = ("name", "assignment__title", "assignment__course__name")
    inlines = [RubricCriterionInline]


@admin.register(RubricCriterion)
class RubricCriterionAdmin(admin.ModelAdmin):
    list_display = ("name", "rubric", "max_score", "weight", "order")
    list_filter = ("rubric",)
    search_fields = ("name", "rubric__name")


@admin.register(RubricScore)
class RubricScoreAdmin(admin.ModelAdmin):
    list_display = ("submission", "criterion", "rater", "score", "updated_at")
    list_filter = ("criterion", "rater")
    search_fields = ("submission__student__anon_id", "submission__assignment__title", "criterion__name", "rater__username")
