from django import forms
from .models import RubricCriterion


class RubricScoreForm(forms.Form):
    """
    Dynamically generated: one field per criterion.
    """
    def __init__(self, *args, criteria=None, existing_scores=None, **kwargs):
        super().__init__(*args, **kwargs)
        criteria = criteria or []
        existing_scores = existing_scores or {}

        for c in criteria:
            field_name = f"crit_{c.id}"
            initial = existing_scores.get(c.id)
            self.fields[field_name] = forms.FloatField(
                label=f"{c.name} (max {c.max_score})",
                min_value=0,
                max_value=float(c.max_score),
                required=True,
                initial=initial,
                widget=forms.NumberInput(attrs={"step": "0.5"})
            )
            # optional notes for each criterion
            self.fields[f"note_{c.id}"] = forms.CharField(
                label=f"Notes for {c.name}",
                required=False,
                widget=forms.Textarea(attrs={"rows": 2})
            )
