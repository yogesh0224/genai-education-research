from django.contrib import admin
from .models import ConsentRecord, AuditLog
from .models import ImportBatch
admin.site.register(ImportBatch)

@admin.register(ConsentRecord)
class ConsentRecordAdmin(admin.ModelAdmin):
    list_display = ("submission", "consented", "consent_source", "recorded_by", "recorded_at")
    list_filter = ("consented", "consent_source")
    search_fields = ("submission__student__anon_id", "submission__assignment__title", "recorded_by__username")


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("actor", "action", "entity_type", "entity_id", "created_at")
    list_filter = ("action", "entity_type")
    search_fields = ("actor__username", "entity_type", "entity_id")
