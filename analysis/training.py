# analysis/training.py

from __future__ import annotations

import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from analysis.models import AnalysisRun, ModelArtifact
from analysis.training_data import build_training_frame


def train_regression_models(
    run: AnalysisRun,
    rubric_id: int,
    target: str = "depth_score_mean",
    n_splits: int = 5,
    random_state: int = 42,
    rf_estimators: int = 300,
    rf_max_depth=None,
    rf_min_samples_leaf: int = 1,
):
    """
    Train regression models using GroupKFold CV (grouped by student_anon_id).

    - Computes CV mean ± std metrics
    - Trains final model on all data
    - Saves models + plots under MEDIA_ROOT/artifacts/<run_id>/
    """

    # ------------------------------------------------------------------
    # 1) Build training dataframe
    # ------------------------------------------------------------------
    df = build_training_frame(
        run_id=run.id,
        rubric_id=rubric_id,
        target_col=target,
        require_target=True,
    )

    if df.empty:
        raise ValueError("No labeled rows available for training.")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected feat_* columns).")

    X = df[feature_cols]
    y = df["y"].astype(float)
    groups = df["group"]

    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError("Need at least 2 distinct students for grouped CV.")

    n_splits = min(n_splits, n_groups)

    # ------------------------------------------------------------------
    # 2) Preprocessing pipeline
    # ------------------------------------------------------------------
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # 3) Models
    # ------------------------------------------------------------------
    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(
            n_estimators=rf_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
        ),
    }

    # ------------------------------------------------------------------
    # 4) Cross-validation (GroupKFold)
    # ------------------------------------------------------------------
    gkf = GroupKFold(n_splits=n_splits)

    artifact_dir = Path(settings.MEDIA_ROOT) / "artifacts" / str(run.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    created_artifacts = []

    for model_name, estimator in models.items():
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups)
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            fold_metrics.append({
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "r2": r2_score(y_test, preds),
                "n_test": int(len(y_test)),
            })

        # ------------------------------------------------------------------
        # 5) Aggregate CV metrics
        # ------------------------------------------------------------------
        mae_vals = [m["mae"] for m in fold_metrics]
        rmse_vals = [m["rmse"] for m in fold_metrics]
        r2_vals = [m["r2"] for m in fold_metrics]

        cv_metrics = {
            "n_splits": n_splits,
            "group_col": "student_anon_id",
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
        }

        # ------------------------------------------------------------------
        # 6) Train final model on ALL data
        # ------------------------------------------------------------------
        final_pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", estimator),
            ]
        )
        final_pipeline.fit(X, y)

        # ------------------------------------------------------------------
        # 7) Save model
        # ------------------------------------------------------------------
        model_path = artifact_dir / f"{model_name}__{target}.joblib"
        joblib.dump(final_pipeline, model_path)

        # ------------------------------------------------------------------
        # 8) Save prediction-vs-actual plot (on full data)
        # ------------------------------------------------------------------
        preds_all = final_pipeline.predict(X)

        pred_plot_path = artifact_dir / f"{model_name}__{target}__pred_vs_actual.png"
        plt.figure()
        plt.scatter(y, preds_all, alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} — {target}")
        plt.tight_layout()
        plt.savefig(pred_plot_path)
        plt.close()

        # ------------------------------------------------------------------
        # 9) Feature importance (RF only)
        # ------------------------------------------------------------------
        fi_path = ""
        if model_name == "rf":
            rf = final_pipeline.named_steps["model"]
            importances = getattr(rf, "feature_importances_", None)

            if importances is not None and len(importances) == len(feature_cols):
                top_n = min(20, len(feature_cols))
                idx = np.argsort(importances)[::-1][:top_n]

                top_feats = [feature_cols[i] for i in idx]
                top_vals = importances[idx]

                fi_plot_path = artifact_dir / f"{model_name}__{target}__feature_importance_top{top_n}.png"
                plt.figure(figsize=(8, 4))
                plt.bar(range(top_n), top_vals)
                plt.xticks(range(top_n), top_feats, rotation=90)
                plt.title(f"{model_name} — Top {top_n} Feature Importances")
                plt.tight_layout()
                plt.savefig(fi_plot_path)
                plt.close()

                fi_path = str(fi_plot_path)

        # ------------------------------------------------------------------
        # 10) Save ModelArtifact
        # ------------------------------------------------------------------
        metrics = {
            "task": "regression",
            "model": model_name,
            "target": target,
            "rubric_id": rubric_id,
            "n_rows": int(len(df)),
            "n_groups": int(n_groups),
            "feature_count": int(len(feature_cols)),
            "cv_metrics": cv_metrics,
            "paths": {
                "model": str(model_path),
                "pred_vs_actual": str(pred_plot_path),
                "feature_importance": fi_path,
            },
        }

        artifact = ModelArtifact.objects.create(
            analysis_run=run,
            type=ModelArtifact.ArtifactType.REGRESSOR,
            name=f"{model_name}:{target}",
            metrics=metrics,
            file_path=str(model_path),
        )

        created_artifacts.append(artifact)

    return created_artifacts
