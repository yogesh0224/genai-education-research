from django.urls import path
from . import views

app_name = "analysis"

urlpatterns = [
    path("analysis/run/<uuid:run_id>/", views.run_dashboard, name="run_dashboard"),
    path("analysis/run/<uuid:run_id>/retrain/", views.retrain_run, name="retrain_run"),
    path("export/joined/<uuid:run_id>.csv", views.export_joined_ml_dataset, name="export_joined_ml_dataset"),
    path("export/clusters/<uuid:run_id>.csv", views.export_cluster_labels_csv, name="export_cluster_labels_csv"),
    path("analysis/compare/", views.compare_runs, name="compare_runs"),
]
