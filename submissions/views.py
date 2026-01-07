import base64
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import render, redirect
from django.urls import reverse

from audit.models import ImportBatch, AuditLog
from submissions.forms import CSVUploadForm, ColumnMappingForm
from submissions.import_utils import read_csv_header, iter_csv_rows, validate_rows
from submissions.models import Student, Submission


SESSION_KEY = "csv_upload_bytes_b64"
SESSION_ASSIGNMENT_ID = "csv_assignment_id"

from django.shortcuts import redirect

def home(request):
    return redirect("submissions:csv_upload")


@login_required
def csv_upload(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            assignment = form.cleaned_data["assignment"]
            f = form.cleaned_data["csv_file"]
            file_bytes = f.read()

            # Save in session (base64)
            request.session[SESSION_KEY] = base64.b64encode(file_bytes).decode("ascii")
            request.session[SESSION_ASSIGNMENT_ID] = assignment.id

            return redirect("submissions:csv_map_columns")
    else:
        form = CSVUploadForm()

    return render(request, "submissions/csv_upload.html", {"form": form})


@login_required
def csv_map_columns(request):
    b64 = request.session.get(SESSION_KEY)
    assignment_id = request.session.get(SESSION_ASSIGNMENT_ID)
    if not b64 or not assignment_id:
        messages.error(request, "No CSV found. Please upload again.")
        return redirect("submissions:csv_upload")

    file_bytes = base64.b64decode(b64)
    columns = read_csv_header(file_bytes)

    if request.method == "POST":
        form = ColumnMappingForm(request.POST, csv_columns=columns)
        if form.is_valid():
            # store mapping in session
            request.session["csv_colmap"] = form.cleaned_data
            return redirect("submissions:csv_preview")
    else:
        form = ColumnMappingForm(csv_columns=columns)

    return render(request, "submissions/csv_map_columns.html", {"form": form, "columns": columns})


@login_required
def csv_preview(request):
    b64 = request.session.get(SESSION_KEY)
    assignment_id = request.session.get(SESSION_ASSIGNMENT_ID)
    colmap = request.session.get("csv_colmap")

    if not b64 or not assignment_id or not colmap:
        messages.error(request, "Missing upload information. Please upload again.")
        return redirect("submissions:csv_upload")

    file_bytes = base64.b64decode(b64)
    rows = iter_csv_rows(file_bytes)

    preview, errors = validate_rows(rows, colmap)

    if request.method == "POST":
        # If errors exist, block import
        if errors:
            messages.error(request, "Fix CSV errors before importing.")
            return render(request, "submissions/csv_preview.html", {"preview": preview, "errors": errors})

        return redirect("submissions:csv_import")

    return render(request, "submissions/csv_preview.html", {"preview": preview, "errors": errors})


@login_required
def csv_import(request):
    """
    Performs the import (POST-only recommended). We'll allow GET to show summary page.
    """
    b64 = request.session.get(SESSION_KEY)
    assignment_id = request.session.get(SESSION_ASSIGNMENT_ID)
    colmap = request.session.get("csv_colmap")

    if not b64 or not assignment_id or not colmap:
        messages.error(request, "Missing upload information. Please upload again.")
        return redirect("submissions:csv_upload")

    file_bytes = base64.b64decode(b64)
    rows = iter_csv_rows(file_bytes)

    # validate again (important: never trust preview step)
    _, errors = validate_rows(rows, colmap, preview_limit=0)
    if errors:
        messages.error(request, "Import blocked due to CSV errors.")
        return redirect("submissions:csv_preview")

    created = 0
    skipped = 0

    with transaction.atomic():
        batch = ImportBatch.objects.create(
            assignment_id=assignment_id,
            uploaded_by=request.user,
            original_filename="uploaded.csv",
            column_map=colmap,
            stats={},
        )

        for row in rows:
            anon_id = row.get(colmap["anon_id_col"], "").strip()
            text = row.get(colmap["text_col"], "").strip()

            if not anon_id or not text:
                skipped += 1
                continue

            student, _ = Student.objects.get_or_create(anon_id=anon_id)

            # Dedup rule (simple, safe baseline):
            # If same student + assignment + identical text exists, skip
            exists = Submission.objects.filter(
                assignment_id=assignment_id,
                student=student,
                text=text,
            ).exists()
            if exists:
                skipped += 1
                continue

            submission = Submission(
                assignment_id=assignment_id,
                student=student,
                text=text,
            )

            if colmap.get("submitted_at_col"):
                submission.submitted_at = row.get(colmap["submitted_at_col"], "") or None

            if colmap.get("self_report_ai_use_col"):
                # import_utils already validates, but we keep it simple here:
                # store raw into metadata too
                submission.metadata["raw_self_report_ai_use"] = row.get(colmap["self_report_ai_use_col"], "")

            if colmap.get("ai_disclosure_text_col"):
                submission.ai_disclosure_text = row.get(colmap["ai_disclosure_text_col"], "")

            if colmap.get("prompt_used_col"):
                submission.prompt_used = row.get(colmap["prompt_used_col"], "")

            submission.metadata["import_batch_id"] = str(batch.id)
            submission.save()
            created += 1

        batch.stats = {"created": created, "skipped": skipped, "total_rows": len(rows)}
        batch.save()

        AuditLog.objects.create(
            actor=request.user,
            action=AuditLog.Action.IMPORT,
            entity_type="ImportBatch",
            entity_id=str(batch.id),
            metadata=batch.stats,
        )

    # Clear session upload
    request.session.pop(SESSION_KEY, None)
    request.session.pop(SESSION_ASSIGNMENT_ID, None)
    request.session.pop("csv_colmap", None)

    messages.success(request, f"Import complete. Created={created}, Skipped={skipped}.")
    return redirect(reverse("admin:submissions_submission_changelist"))
