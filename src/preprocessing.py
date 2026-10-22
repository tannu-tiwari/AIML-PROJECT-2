"""Data preprocessing pipeline using Scikit-Learn ColumnTransformer.

Provides modular transformation pipelines for categorical and numerical features,
ensuring strict separation of train/test statistics to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    """Constructs a Scikit-Learn ColumnTransformer for mixed feature types.

    Categorical attributes are one-hot encoded with handle_unknown='ignore'
    to ensure robust inference on new or unseen student attributes.
    Numeric attributes are standardized using StandardScaler.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fits the preprocessor exclusively on training data to prevent leakage."""
    logger.info(f"Fitting preprocessor on training data ({len(X_train)} samples)...")
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Transforms raw DataFrame into processed NumPy array and extracts clean feature names."""
    transformed_matrix = preprocessor.transform(X)
    feature_names = get_clean_feature_names(preprocessor)
    return transformed_matrix, feature_names


def get_clean_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieves human-readable feature names post one-hot encoding and scaling."""
    raw_names = preprocessor.get_feature_names_out()
    clean_names = []
    for name in raw_names:
        # Strip transformer prefixes like 'cat__' or 'num__'
        if "__" in name:
            clean_name = name.split("__", 1)[1]
        else:
            clean_name = name
        clean_names.append(clean_name)
    return clean_names


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str | Path) -> Path:
    """Serializes fitted ColumnTransformer artifact using joblib."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_path)
    logger.info(f"Saved fitted preprocessor to: {out_path}")
    return out_path


def load_preprocessor(filepath: str | Path) -> ColumnTransformer:
    """Loads serialized ColumnTransformer artifact."""
    in_path = Path(filepath)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {in_path}")
    preprocessor = joblib.load(in_path)
    logger.info(f"Loaded preprocessor from: {in_path}")
    return preprocessor


if __name__ == "__main__":
    from data_loader import prepare_dataset_bundle

    bundle = prepare_dataset_bundle()
    preprocessor = build_preprocessor(bundle.categorical_cols, bundle.numeric_cols)
    preprocessor = fit_preprocessor(preprocessor, bundle.X_train)

    X_train_trans, names = transform_data(preprocessor, bundle.X_train)
    X_test_trans, _ = transform_data(preprocessor, bundle.X_test)

    print(f"X_train shape post-transform: {X_train_trans.shape}")
    print(f"X_test shape post-transform: {X_test_trans.shape}")
    print(f"Total encoded features ({len(names)}): {names[:10]} ...")
