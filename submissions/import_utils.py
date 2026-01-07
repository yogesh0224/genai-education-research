import csv
import io
from datetime import datetime
from typing import Dict, List, Tuple, Any

from django.utils.dateparse import parse_datetime


ALLOWED_AI = {"UNKNOWN", "NONE", "LOW", "MEDIUM", "HIGH", "FULL", "-1", "0", "1", "2", "3", "4"}


def read_csv_header(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    return [h.strip() for h in header]


def iter_csv_rows(file_bytes: bytes) -> List[Dict[str, str]]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def normalize_ai_self_report(val: str) -> int:
    """
    Map CSV values to Submission.AISelfReport integer codes.
    """
    v = (val or "").strip().upper()
    mapping = {
        "UNKNOWN": -1, "-1": -1,
        "NONE": 0, "0": 0,
        "LOW": 1, "1": 1,
        "MEDIUM": 2, "2": 2,
        "HIGH": 3, "3": 3,
        "FULL": 4, "4": 4,
    }
    if v == "":
        return -1
    if v not in mapping:
        raise ValueError(f"Invalid self_report_ai_use: {val}")
    return mapping[v]


def parse_submitted_at(val: str):
    if not val:
        return None
    # Try Django parser first (handles ISO strings well)
    dt = parse_datetime(val)
    if dt:
        return dt
    # Fallback: common format
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid submitted_at datetime: {val}")


def validate_rows(rows: List[Dict[str, str]], colmap: Dict[str, str], preview_limit: int = 25) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - preview: list of cleaned preview rows (up to preview_limit)
      - errors: list of error strings (all found errors)
    """
    errors = []
    preview = []

    for i, row in enumerate(rows, start=2):  # 1 is header
        anon_id = row.get(colmap["anon_id_col"], "").strip()
        text = row.get(colmap["text_col"], "").strip()

        if not anon_id:
            errors.append(f"Row {i}: anon_id is empty.")
        if not text:
            errors.append(f"Row {i}: text is empty.")

        cleaned = {"anon_id": anon_id, "text": text}

        # Optional fields
        if colmap.get("submitted_at_col"):
            try:
                cleaned["submitted_at"] = parse_submitted_at(row.get(colmap["submitted_at_col"], ""))
            except Exception as e:
                errors.append(f"Row {i}: {e}")

        if colmap.get("self_report_ai_use_col"):
            try:
                cleaned["self_report_ai_use"] = normalize_ai_self_report(row.get(colmap["self_report_ai_use_col"], ""))
            except Exception as e:
                errors.append(f"Row {i}: {e}")

        if colmap.get("ai_disclosure_text_col"):
            cleaned["ai_disclosure_text"] = row.get(colmap["ai_disclosure_text_col"], "")

        if colmap.get("prompt_used_col"):
            cleaned["prompt_used"] = row.get(colmap["prompt_used_col"], "")

        if len(preview) < preview_limit:
            preview.append(cleaned)

    return preview, errors
