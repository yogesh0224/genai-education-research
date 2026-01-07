from django import forms
from courses.models import Assignment


class CSVUploadForm(forms.Form):
    assignment = forms.ModelChoiceField(queryset=Assignment.objects.all())
    csv_file = forms.FileField()


class ColumnMappingForm(forms.Form):
    """
    User chooses which CSV column corresponds to which field.
    We'll populate choices dynamically after reading the header.
    """
    anon_id_col = forms.ChoiceField()
    text_col = forms.ChoiceField()

    submitted_at_col = forms.ChoiceField(required=False)
    self_report_ai_use_col = forms.ChoiceField(required=False)
    ai_disclosure_text_col = forms.ChoiceField(required=False)
    prompt_used_col = forms.ChoiceField(required=False)

    def __init__(self, *args, csv_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [("", "— not provided —")] + [(c, c) for c in (csv_columns or [])]
        for f in self.fields.values():
            f.choices = choices
