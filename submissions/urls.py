from django.urls import path
from . import views

app_name = "submissions"

urlpatterns = [
    path("", views.home, name="home"),

    path("import/csv/", views.csv_upload, name="csv_upload"),
    path("import/csv/map/", views.csv_map_columns, name="csv_map_columns"),
    path("import/csv/preview/", views.csv_preview, name="csv_preview"),
    path("import/csv/import/", views.csv_import, name="csv_import"),
]
