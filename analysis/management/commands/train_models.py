# analysis/management/commands/train_models.py

import uuid
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analysis.models import AnalysisRun, ModelArtifact
from analysis.training_data import build_training_frame


class Command(BaseCommand):
    help = "Train baseline regression models using SubmissionFeatures joined with rubric targets."

    def add_arguments(self, parser):
        parser.add_argument(
            "run_id",
            type=str,
            help="UUID of AnalysisRun (e.g., 2b3c1e9a-4f7d-4c77-9e2a-9f8c2a0c3d11)",
        )
        parser.add_argument("--rubric_id", type=int, required=True, help="Rubric ID to aggregate targets from.")
        parser.add_argument(
            "--target",
            type=str,
            default="depth_score_mean",
            help="Target column from rubric aggregation (e.g., depth_score_mean, total_score_weighted_mean).",
        )
        parser.add_argument("--test_size", type=float, default=0.2)
        parser.add_argument("--random_state", type=int, default=42)

        # RF knobs (optional)
        parser.add_argument("--rf_estimators", type=int, default=300)
        parser.add_argument("--rf_max_depth", type=int, default=0, help="0 means None (unlimited).")
        parser.add_argument("--rf_min_samples_leaf", type=int, default=1)

    def handle(self, *args, **opts):
        # ---- Validate run UUID early (avoids Django UUID ValidationError) ----
        run_id_raw = opts["run_id"].strip()
        try:
            run_uuid = uuid.UUID(run_id_raw)
        except ValueError:
            raise CommandError(
                f"run_id must be a valid UUID. You provided: {run_id_raw}\n"
                f"Example: 2b3c1e9a-4f7d-4c77-9e2a-9f8c2a0c3d11"
            )

        rubric_id = int(opts["rubric_id"])
        target = opts["target"].strip()
        test_size = float(opts["test_size"])
        random_state = int(opts["random_state"])

        run = AnalysisRun.objects.filter(id=run_uuid).first()
        if not run:
            raise CommandError(f"AnalysisRun not found: {run_uuid}")

        # ---- Load joined training dataframe ----
        df = build_training_frame(run_id=run.id, rubric_id=rubric_id, target_col=target)

        if df.empty:
            raise CommandError(
                "No training rows found. Make sure:\n"
                "- You have SubmissionFeatures for this run\n"
                "- You have rubric scores for this assignment/rubric\n"
                f"- The target '{target}' exists and is non-empty"
            )

        if len(df) < 10:
            raise CommandError(f"Not enough labeled rows to train (need >= 10, got {len(df)}).")

        # ---- Select feature columns ----
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feature_cols:
            raise CommandError(
                "No feature columns found (expected columns starting with 'feat_').\n"
                "Make sure your feature extraction stored keys in SubmissionFeatures.features."
            )

        X = df[feature_cols]
        y = df["y"].astype(float)

        # ---- Train/test split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # ---- Preprocessing: numeric imputer ----
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, feature_cols)],
            remainder="drop",
        )

        # ---- Define models ----
        rf_max_depth = None if int(opts["rf_max_depth"]) == 0 else int(opts["rf_max_depth"])

        models = {
            "ridge": Ridge(alpha=1.0, random_state=random_state),
            "rf": RandomForestRegressor(
                n_estimators=int(opts["rf_estimators"]),
                random_state=random_state,
                n_jobs=-1,
                max_depth=rf_max_depth,
                min_samples_leaf=int(opts["rf_min_samples_leaf"]),
            ),
        }

        # ---- Prepare artifact directory ----
        media_root = getattr(settings, "MEDIA_ROOT", None)
        if not media_root:
            raise CommandError("MEDIA_ROOT is not set in settings.py. Please configure it to save artifacts.")

        artifact_dir = Path(media_root) / "artifacts" / str(run.id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # ---- Train + persist ----
        created_artifacts = 0

        with transaction.atomic():
            # Update run status
            run.status = AnalysisRun.Status.RUNNING
            run.started_at = run.started_at or timezone.now()
            run.error_message = ""
            run.config = {**(run.config or {}), "training_request": {
                "rubric_id": rubric_id,
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "rf_estimators": int(opts["rf_estimators"]),
                "rf_max_depth": rf_max_depth,
                "rf_min_samples_leaf": int(opts["rf_min_samples_leaf"]),
            }}
            run.save()

            for name, estimator in models.items():
                pipeline = Pipeline(steps=[
                    ("prep", preprocessor),
                    ("model", estimator),
                ])

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                mae = float(mean_absolute_error(y_test, preds))
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = float(r2_score(y_test, preds))

                metrics = {
                    "task": "regression",
                    "model": name,
                    "target": target,
                    "rubric_id": rubric_id,
                    "n_rows": int(len(df)),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "feature_count": int(len(feature_cols)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "random_state": random_state,
                    "test_size": test_size,
                }

                # Save model file
                model_path = artifact_dir / f"{name}__{target}.joblib"
                joblib.dump(pipeline, model_path)

                # Save prediction plot
                pred_plot_path = artifact_dir / f"{name}__{target}__pred_vs_actual.png"
                plt.figure()
                plt.scatter(y_test, preds)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{name} — {target}")
                plt.savefig(pred_plot_path)
                plt.close()

                # Save RF feature importance
                fi_path = ""
                if name == "rf":
                    rf = pipeline.named_steps["model"]
                    importances = getattr(rf, "feature_importances_", None)
                    if importances is not None and len(importances) == len(feature_cols):
                        top_n = min(20, len(feature_cols))
                        top_idx = np.argsort(importances)[::-1][:top_n]
                        top_feats = [feature_cols[i] for i in top_idx]
                        top_vals = importances[top_idx]

                        fi_plot_path = artifact_dir / f"{name}__{target}__feature_importance_top{top_n}.png"
                        plt.figure()
                        plt.bar(range(top_n), top_vals)
                        plt.xticks(range(top_n), top_feats, rotation=90)
                        plt.title(f"{name} — Top {top_n} feature importances")
                        plt.tight_layout()
                        plt.savefig(fi_plot_path)
                        plt.close()
                        fi_path = str(fi_plot_path)

                # Store artifact record
                ModelArtifact.objects.create(
                    analysis_run=run,
                    type=ModelArtifact.ArtifactType.REGRESSOR,
                    name=f"{name}:{target}",
                    metrics={
                        **metrics,
                        "paths": {
                            "model": str(model_path),
                            "pred_vs_actual": str(pred_plot_path),
                            "feature_importance": fi_path,
                        }
                    },
                    file_path=str(model_path),
                )
                created_artifacts += 1

            # Mark run complete
            run.status = AnalysisRun.Status.DONE
            run.finished_at = timezone.now()
            run.save()

        self.stdout.write(self.style.SUCCESS(
            f"Training complete. Created {created_artifacts} artifacts.\n"
            f"Artifacts folder: {artifact_dir}"
        ))
