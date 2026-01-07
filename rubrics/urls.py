from django.urls import path
from . import views

app_name = "rubrics"

urlpatterns = [
    path("score/", views.assignment_select, name="assignment_select"),
    path("score/<int:assignment_id>/", views.rubric_select, name="rubric_select"),
    path("score/<int:assignment_id>/<int:rubric_id>/queue/", views.scoring_queue, name="scoring_queue"),
    path("score/<int:assignment_id>/<int:rubric_id>/<uuid:submission_id>/", views.score_submission, name="score_submission"),
    path("export/<int:assignment_id>/<int:rubric_id>/ml.csv", views.export_ml_dataset, name="export_ml_dataset"),
    path("export/<int:assignment_id>/<int:rubric_id>/ml.csv", views.export_ml_dataset, name="export_ml_dataset"),

]
