# analysis/clustering.py

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analysis.clustering_data import build_clustering_frame


def run_umap_clustering(
    run_id,
    rubric_id,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    n_clusters=3,
    random_state=42,
):
    """
    Returns dataframe with:
    - umap_x, umap_y
    - cluster label
    """
    df = build_clustering_frame(run_id, rubric_id)

    if df.empty:
        raise ValueError("No data for clustering.")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].fillna(0)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_scaled)

    # Clustering in embedding space
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = km.fit_predict(embedding)

    out = df.copy()
    out["umap_x"] = embedding[:, 0]
    out["umap_y"] = embedding[:, 1]
    out["cluster"] = clusters

    return out
