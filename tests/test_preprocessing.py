"""Unit tests for Scikit-Learn preprocessing pipeline."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import prepare_dataset_bundle
from preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    get_clean_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_data,
)


def test_build_and_fit_preprocessor(tmp_path):
    """Tests building, fitting, and transforming via ColumnTransformer."""
    bundle = prepare_dataset_bundle()
    prep = build_preprocessor(bundle.categorical_cols, bundle.numeric_cols)

    # Must fit strictly on X_train
    fitted_prep = fit_preprocessor(prep, bundle.X_train)

    # Transform train and test
    X_tr_trans, names_tr = transform_data(fitted_prep, bundle.X_train)
    X_te_trans, names_te = transform_data(fitted_prep, bundle.X_test)

    assert isinstance(X_tr_trans, np.ndarray)
    assert isinstance(X_te_trans, np.ndarray)
    assert X_tr_trans.shape[0] == 316
    assert X_te_trans.shape[0] == 79
    assert X_tr_trans.shape[1] == X_te_trans.shape[1]
    assert names_tr == names_te

    # Confirm zero NaN values after transformation
    assert not np.isnan(X_tr_trans).any()
    assert not np.isnan(X_te_trans).any()


def test_unseen_categories_gracefully_handled():
    """Confirms OneHotEncoder handles unseen categorical levels without crashing."""
    bundle = prepare_dataset_bundle()
    prep = build_preprocessor(bundle.categorical_cols, bundle.numeric_cols)
    fitted_prep = fit_preprocessor(prep, bundle.X_train)

    unseen_row = bundle.X_test.iloc[[0]].copy()
    unseen_row.loc[unseen_row.index[0], "Mjob"] = "astronaut"  # Unseen category

    # Should not raise exception because handle_unknown='ignore'
    transformed, _ = transform_data(fitted_prep, unseen_row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_preprocessor_serialization(tmp_path):
    """Verifies preprocessor persistence and identical transformation upon reload."""
    bundle = prepare_dataset_bundle()
    prep = build_preprocessor(bundle.categorical_cols, bundle.numeric_cols)
    fitted_prep = fit_preprocessor(prep, bundle.X_train)

    save_file = tmp_path / "test_preprocessor.joblib"
    save_preprocessor(fitted_prep, save_file)
    assert save_file.exists()

    reloaded_prep = load_preprocessor(save_file)
    X_orig, _ = transform_data(fitted_prep, bundle.X_test)
    X_reloaded, _ = transform_data(reloaded_prep, bundle.X_test)

    np.testing.assert_allclose(X_orig, X_reloaded)
