from pathlib import Path

import numpy as np

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

from analysis.models import AnalysisRun, ModelArtifact
from analysis.training_data import build_training_frame


class Command(BaseCommand):
    help = "Train regression models with group-aware cross-validation and save artifacts."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=str)
        parser.add_argument("--rubric_id", type=int, required=True)
        parser.add_argument("--target", type=str, required=True)
        parser.add_argument("--min_rows", type=int, default=10)
        parser.add_argument("--random_state", type=int, default=42)

    def handle(self, *args, **opts):
        run_id = opts["run_id"]
        rubric_id = opts["rubric_id"]
        target = opts["target"]
        min_rows = opts["min_rows"]
        random_state = opts["random_state"]

        try:
            run = AnalysisRun.objects.get(id=run_id)
        except AnalysisRun.DoesNotExist:
            raise CommandError(f"AnalysisRun not found: {run_id}")

        # ---- Load training data ----
        df = build_training_frame(
            run_id=run_id,
            rubric_id=rubric_id,
            target_col=target,
            require_target=True,
        )

        if len(df) < min_rows:
            raise CommandError(
                f"Not enough labeled rows to train (need >= {min_rows}, got {len(df)})."
            )

        y = df["y"].values
        groups = df["group"].values

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feature_cols]

        # ---- Preprocessing (important for SVR) ----
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, feature_cols)],
            remainder="drop",
        )

        # ---- Models ----
        models = {
            "ridge": Ridge(alpha=1.0),
            "svr": LinearSVR(
                C=0.5,
                epsilon=0.1,
                random_state=random_state,
                max_iter=100_000,
                tol=1e-4,
            ),
            "rf": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        }

        n_splits = min(5, len(np.unique(groups)))
        if n_splits < 2:
            raise CommandError("Need at least 2 unique students/groups for GroupKFold CV.")

        gkf = GroupKFold(n_splits=n_splits)

        artifacts_created = 0
        out_dir = Path(settings.MEDIA_ROOT) / "artifacts" / str(run.id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import (keeps startup clean)
        import joblib

        for model_name, model in models.items():
            self.stdout.write(f"\n=== Training model: {model_name} ===")

            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )

            y_true_all = []
            y_pred_all = []

            try:
                for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)

                    y_true_all.extend(y_test.tolist())
                    y_pred_all.extend(preds.tolist())

                    self.stdout.write(f"  fold {fold}/{n_splits} done")

            except Exception as e:
                self.stderr.write(f"❌ {model_name} failed during CV: {e}")
                continue

            if not y_true_all:
                self.stderr.write(f"❌ {model_name} produced no predictions")
                continue

            # ---- Metrics (compatible with older sklearn) ----
            y_true_all = np.array(y_true_all, dtype=float)
            y_pred_all = np.array(y_pred_all, dtype=float)

            mae = float(mean_absolute_error(y_true_all, y_pred_all))
            mse = float(mean_squared_error(y_true_all, y_pred_all))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true_all, y_pred_all))

            metrics = {
                "model": model_name,
                "target": target,
                "n_rows": int(len(df)),
                "n_groups": int(len(np.unique(groups))),
                "cv_metrics": {
                    "mae_mean": mae,
                    "rmse_mean": rmse,
                    "r2_mean": r2,
                },
            }

            # ---- Fit final model on full data + save ----
            model_path = out_dir / f"{model_name}_{target}.joblib"
            try:
                pipeline.fit(X, y)
                joblib.dump(pipeline, model_path)
                file_path = str(model_path)
            except Exception as e:
                self.stderr.write(f"⚠️ Could not save model file for {model_name}: {e}")
                file_path = ""

            # ---- Save DB artifact (MATCHES YOUR MODEL) ----
            ModelArtifact.objects.create(
                analysis_run=run,
                type=ModelArtifact.ArtifactType.REGRESSOR,
                name=f"{model_name}:{target}",
                metrics=metrics,
                file_path=file_path,
            )

            artifacts_created += 1
            self.stdout.write(f"✅ Saved artifact for {model_name}")

        self.stdout.write(
            self.style.SUCCESS(
                f"\nTraining complete. Created {artifacts_created} artifacts.\n"
                f"Artifacts folder: {out_dir}"
            )
        )
