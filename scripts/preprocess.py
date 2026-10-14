"""Fungsi dan transformer untuk pembacaan, pembersihan, serta preprocessing data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)

from scripts.config import (ID_COL, TARGET, INVALID_VALUES, NUM_COLS, ENGINEERED_COLS,
                            CAT_COLS, GEO_OHE_COLS, GEO_TE_COLS, LOG_COLS,
                            OHE_MIN_FREQUENCY, RANDOM_STATE)


def load_data(path):
    """
    Data CSV dibaca menjadi DataFrame.

    Parameters
    ----------
    path : str or Path
        Lokasi file CSV.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def convert_target(y):
    """
    Target "Grade X" dikonversi menjadi integer X.

    Parameters
    ----------
    y : pd.Series

    Returns
    -------
    pd.Series
    """
    return y.str.replace("Grade ", "", regex=False).astype(int)


def split_xy(df):
    """
    Data train dipisahkan menjadi fitur (X) dan target (y).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[ID_COL, TARGET])
    y = convert_target(df[TARGET])
    return X, y


class InvalidValueReplacer(BaseEstimator, TransformerMixin):
    """
    Kode nilai tidak valid diganti menjadi NaN.

    Parameters
    ----------
    invalid_values : dict, optional
        Pasangan {kolom: nilai}. Jika None, digunakan INVALID_VALUES dari config.
    """

    def __init__(self, invalid_values=None):
        self.invalid_values = invalid_values

    def fit(self, X, y=None):
        """Tidak ada parameter yang dipelajari."""
        return self

    def transform(self, X):
        """
        Nilai tidak valid pada kolom terkait diganti NaN.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        X = X.copy()
        for col, val in (self.invalid_values or INVALID_VALUES).items():
            X[col] = X[col].replace(val, np.nan)
        return X


def build_preprocessor(kind="linear"):
    """
    ColumnTransformer dibangun sesuai jenis model.

    Parameters
    ----------
    kind : {"linear", "tree"}
        "linear" untuk model berbasis koefisien (imputasi, log, scaling, one-hot).
        "tree" untuk model berbasis pohon (imputasi dan ordinal encoding); kolom
        tambahan di luar daftar config (misalnya agregat wilayah) diteruskan apa adanya.

    Returns
    -------
    ColumnTransformer
    """
    num_cols = NUM_COLS + ENGINEERED_COLS
    target_enc = TargetEncoder(target_type="multiclass", random_state=RANDOM_STATE)

    if kind == "linear":
        other_num = [c for c in num_cols if c not in LOG_COLS]
        log_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        onehot = OneHotEncoder(handle_unknown="infrequent_if_exist",
                               min_frequency=OHE_MIN_FREQUENCY, sparse_output=False)
        return ColumnTransformer([
            ("log_num", log_pipe, LOG_COLS),
            ("num", num_pipe, other_num),
            ("cat", onehot, CAT_COLS + GEO_OHE_COLS),
            ("geo_te", target_enc, GEO_TE_COLS),
        ])

    if kind == "tree":
        ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", ordinal, CAT_COLS),
            ("geo", "passthrough", GEO_OHE_COLS),
            ("geo_te", target_enc, GEO_TE_COLS),
        ], remainder="passthrough")   # kolom agregat wilayah diteruskan

    raise ValueError("kind harus 'linear' atau 'tree'")