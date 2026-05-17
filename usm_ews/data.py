from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from usm_ews.model import CustomRandomForest


@dataclass
class TrainingArtifacts:
    model: CustomRandomForest
    accuracy: float
    feature_cols: list[str]
    df_raw: pd.DataFrame
    class_dist: dict[int, int]


@st.cache_resource
def prepare_and_train() -> TrainingArtifacts:
    try:
        df = pd.read_csv("data.csv", sep=";")
        df.columns = df.columns.str.strip()
        if "Target" not in df.columns:
            st.error("Column 'Target' not found.")
            st.stop()

        feature_cols = [col for col in df.columns if col != "Target"]
        target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
        df["Target"] = df["Target"].str.strip().map(target_mapping)
        df = df.dropna()
        y = df["Target"].astype(int).values
        X = df[feature_cols].values

        if len(y) > 2000:
            np.random.seed(42)
            sample_idx = np.random.choice(len(y), 2000, replace=False)
            X, y = X[sample_idx], y[sample_idx]
    except Exception as exc:
        st.error(f"Error loading data.csv: {exc}")
        st.stop()

    model = CustomRandomForest(n_trees=11, max_depth=6)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = np.sum(y == y_pred) / len(y)
    class_dist = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1)), 2: int(np.sum(y == 2))}
    return TrainingArtifacts(
        model=model,
        accuracy=accuracy,
        feature_cols=feature_cols,
        df_raw=df,
        class_dist=class_dist,
    )
