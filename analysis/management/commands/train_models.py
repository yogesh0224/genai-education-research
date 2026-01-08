# analysis/management/commands/train_models.py

from pathlib import Path

import numpy as np
import pandas as pd

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib

from analysis.models import AnalysisRun, ModelArtifact
from analysis.training_data import build_training_frame


class Command(BaseCommand):
    help = "Train regression models with group-aware CV; saves OOF predictions for error analysis; includes TF-IDF baseline."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=str)
        parser.add_argument("--rubric_id", type=int, required=True)
        parser.add_argument("--target", type=str, required=True)

        parser.add_argument("--min_rows", type=int, default=10)
        parser.add_argument("--random_state", type=int, default=42)
        parser.add_argument("--max_folds", type=int, default=5)

    def handle(self, *args, **opts):
        run_id = opts["run_id"]
        rubric_id = opts["rubric_id"]
        target = opts["target"]
        min_rows = opts["min_rows"]
        random_state = opts["random_state"]
        max_folds = opts["max_folds"]

        try:
            run = AnalysisRun.objects.get(id=run_id)
        except AnalysisRun.DoesNotExist:
            raise CommandError(f"AnalysisRun not found: {run_id}")

        # -------- Load joined training frame --------
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

        if "submission_id" not in df.columns:
            raise CommandError("Training frame missing 'submission_id' column.")
        if "group" not in df.columns:
            raise CommandError("Training frame missing 'group' column.")
        if "y" not in df.columns:
            raise CommandError("Training frame missing 'y' column.")

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feature_cols:
            raise CommandError("No feature columns found (expected columns starting with 'feat_').")

        X = df[feature_cols]
        y = df["y"].astype(float).values
        groups = df["group"].astype(str).values

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        n_splits = min(max_folds, n_groups)
        if n_splits < 2:
            raise CommandError(f"Need at least 2 unique groups for GroupKFold. Got {n_groups}.")

        gkf = GroupKFold(n_splits=n_splits)

        # -------- Output folder --------
        out_dir = Path(settings.MEDIA_ROOT) / "artifacts" / str(run.id)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _media_url_for(filename: str) -> str:
            base = settings.MEDIA_URL.rstrip("/")
            return f"{base}/artifacts/{run.id}/{filename}"

        # -------- Numeric preprocessing --------
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, feature_cols),
            ],
            remainder="drop",
        )

        # -------- Models (numeric feature set) --------
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

        artifacts_created = 0

        # ============================
        # Part 1: numeric-feature models
        # ============================
        for model_name, model in models.items():
            self.stdout.write(f"\n=== Training model: {model_name} ===")

            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )

            oof_rows = []
            fold_metrics = []

            try:
                for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)

                    mae_f = mean_absolute_error(y_test, preds)
                    mse_f = mean_squared_error(y_test, preds)  # sklearn version-safe
                    rmse_f = float(np.sqrt(mse_f))
                    r2_f = r2_score(y_test, preds)

                    fold_metrics.append({
                        "fold": fold_idx,
                        "mae": float(mae_f),
                        "rmse": float(rmse_f),
                        "r2": float(r2_f),
                        "n": int(len(test_idx)),
                    })

                    test_meta = df.iloc[test_idx][["submission_id", "group"]].copy()
                    for i in range(len(test_meta)):
                        actual = float(y_test[i])
                        pred = float(preds[i])
                        oof_rows.append({
                            "submission_id": str(test_meta.iloc[i]["submission_id"]),
                            "student_anon_id": str(test_meta.iloc[i]["group"]),
                            "actual": actual,
                            "predicted": pred,
                            "error": float(pred - actual),
                            "abs_error": float(abs(pred - actual)),
                            "fold": fold_idx,
                        })

            except Exception as e:
                self.stderr.write(f"❌ {model_name} failed during CV: {e}")
                continue

            if not oof_rows:
                self.stderr.write(f"❌ {model_name} produced no OOF predictions; skipping artifact.")
                continue

            mae_vals = [m["mae"] for m in fold_metrics]
            rmse_vals = [m["rmse"] for m in fold_metrics]
            r2_vals = [m["r2"] for m in fold_metrics]

            cv_metrics = {
                "mae_mean": float(np.mean(mae_vals)),
                "mae_std": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0,
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else 0.0,
                "r2_mean": float(np.mean(r2_vals)),
                "r2_std": float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0,
            }

            # Save OOF CSV
            pred_filename = f"{model_name}_{target}_oof_predictions.csv"
            pred_path = out_dir / pred_filename
            pd.DataFrame(oof_rows).to_csv(pred_path, index=False)
            oof_csv_url = _media_url_for(pred_filename)

            # Fit full + save model joblib
            model_joblib_url = None
            try:
                pipeline.fit(X, y)
                model_filename = f"{model_name}_{target}.joblib"
                joblib.dump(pipeline, out_dir / model_filename)
                model_joblib_url = _media_url_for(model_filename)
            except Exception as e:
                self.stderr.write(f"⚠️ {model_name}: failed to save joblib: {e}")

            metrics = {
                "model": model_name,
                "feature_set": "numeric",
                "target": target,
                "rubric_id": int(rubric_id),
                "n_rows": int(len(df)),
                "n_groups": int(n_groups),
                "feature_count": int(len(feature_cols)),
                "cv_folds": int(n_splits),
                "cv_metrics": cv_metrics,
                "fold_metrics": fold_metrics,
                "model_joblib_url": model_joblib_url,
                "oof_predictions_csv_url": oof_csv_url,
            }

            ModelArtifact.objects.create(
                analysis_run=run,
                name=f"{model_name}:{target}",
                metrics=metrics,
            )

            artifacts_created += 1
            self.stdout.write(self.style.SUCCESS(f"✅ Saved artifact: {model_name}:{target}"))

        # ============================
        # Part 2: TF-IDF baseline model
        # ============================
        if "text" in df.columns:
            text = df["text"].fillna("").astype(str).values
            self.stdout.write(f"\n=== Training model: tfidf_ridge ===")

            tfidf_pipeline = Pipeline(steps=[
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    max_features=20000,
                )),
                ("model", Ridge(alpha=1.0)),
            ])

            oof_rows = []
            fold_metrics = []

            try:
                dummy_X = np.zeros((len(text), 1))
                for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dummy_X, y, groups), start=1):
                    X_train = text[train_idx]
                    X_test = text[test_idx]
                    y_train = y[train_idx]
                    y_test = y[test_idx]

                    tfidf_pipeline.fit(X_train, y_train)
                    preds = tfidf_pipeline.predict(X_test)

                    mae_f = mean_absolute_error(y_test, preds)
                    mse_f = mean_squared_error(y_test, preds)
                    rmse_f = float(np.sqrt(mse_f))
                    r2_f = r2_score(y_test, preds)

                    fold_metrics.append({
                        "fold": fold_idx,
                        "mae": float(mae_f),
                        "rmse": float(rmse_f),
                        "r2": float(r2_f),
                        "n": int(len(test_idx)),
                    })

                    test_meta = df.iloc[test_idx][["submission_id", "group"]].copy()
                    for i in range(len(test_meta)):
                        actual = float(y_test[i])
                        pred = float(preds[i])
                        oof_rows.append({
                            "submission_id": str(test_meta.iloc[i]["submission_id"]),
                            "student_anon_id": str(test_meta.iloc[i]["group"]),
                            "actual": actual,
                            "predicted": pred,
                            "error": float(pred - actual),
                            "abs_error": float(abs(pred - actual)),
                            "fold": fold_idx,
                        })

            except Exception as e:
                self.stderr.write(f"❌ tfidf_ridge failed during CV: {e}")
            else:
                mae_vals = [m["mae"] for m in fold_metrics]
                rmse_vals = [m["rmse"] for m in fold_metrics]
                r2_vals = [m["r2"] for m in fold_metrics]

                cv_metrics = {
                    "mae_mean": float(np.mean(mae_vals)),
                    "mae_std": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0,
                    "rmse_mean": float(np.mean(rmse_vals)),
                    "rmse_std": float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else 0.0,
                    "r2_mean": float(np.mean(r2_vals)),
                    "r2_std": float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0,
                }

                pred_filename = f"tfidf_ridge_{target}_oof_predictions.csv"
                pred_path = out_dir / pred_filename
                pd.DataFrame(oof_rows).to_csv(pred_path, index=False)
                oof_csv_url = _media_url_for(pred_filename)

                model_joblib_url = None
                try:
                    tfidf_pipeline.fit(text, y)
                    model_filename = f"tfidf_ridge_{target}.joblib"
                    joblib.dump(tfidf_pipeline, out_dir / model_filename)
                    model_joblib_url = _media_url_for(model_filename)
                except Exception as e:
                    self.stderr.write(f"⚠️ tfidf_ridge: failed to save joblib: {e}")

                metrics = {
                    "model": "tfidf_ridge",
                    "feature_set": "tfidf",
                    "target": target,
                    "rubric_id": int(rubric_id),
                    "n_rows": int(len(df)),
                    "n_groups": int(n_groups),
                    "cv_folds": int(n_splits),
                    "cv_metrics": cv_metrics,
                    "fold_metrics": fold_metrics,
                    "model_joblib_url": model_joblib_url,
                    "oof_predictions_csv_url": oof_csv_url,
                }

                ModelArtifact.objects.create(
                    analysis_run=run,
                    name=f"tfidf_ridge:{target}",
                    metrics=metrics,
                )

                artifacts_created += 1
                self.stdout.write(self.style.SUCCESS(f"✅ Saved artifact: tfidf_ridge:{target}"))
        else:
            self.stderr.write("TF-IDF baseline skipped: training frame missing 'text' column.")

        self.stdout.write(
            self.style.SUCCESS(
                f"\nTraining complete. Created {artifacts_created} artifacts.\n"
                f"Artifacts folder: {out_dir}"
            )
        )
