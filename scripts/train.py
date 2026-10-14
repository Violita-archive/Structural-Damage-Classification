"""Fungsi pembangunan pipeline, model final, dan pelatihan."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from scripts.preprocess import InvalidValueReplacer, build_preprocessor
from scripts.feature_engineering import FeatureEngineer, GroupAggregator


def build_pipeline(model, kind="tree", aggregate=False):
    """
    Pipeline lengkap disusun: pembersihan → fitur baru → (agregat wilayah) → preprocessing → model.

    Parameters
    ----------
    model : estimator sklearn atau "passthrough"
        Model klasifikasi pada langkah terakhir.
    kind : {"linear", "tree"}
        Jenis preprocessor yang digunakan.
    aggregate : bool
        Jika True, fitur statistik per wilayah ditambahkan (hanya untuk kind="tree").

    Returns
    -------
    Pipeline
    """
    steps = [("replace", InvalidValueReplacer()), ("features", FeatureEngineer())]
    if aggregate:
        steps.append(("aggregate", GroupAggregator()))
    steps += [("preprocess", build_preprocessor(kind)), ("model", model)]
    return Pipeline(steps)


class WeightedProbaClassifier(BaseEstimator, ClassifierMixin):
    """
    Probabilitas estimator dikalikan bobot per kelas sebelum argmax.

    Parameters
    ----------
    estimator : estimator sklearn
        Model dengan method predict_proba.
    weights : array-like, optional
        Bobot per kelas sesuai urutan classes_. Jika None, seluruh bobot bernilai 1.
    """

    def __init__(self, estimator, weights=None):
        self.estimator = estimator
        self.weights = weights

    def fit(self, X, y):
        """
        Estimator dilatih pada data.

        Parameters
        ----------
        X : pd.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        """Probabilitas asli estimator dikembalikan tanpa pembobotan."""
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """
        Kelas diprediksi dari argmax probabilitas yang telah dibobot.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        w = np.ones(len(self.classes_)) if self.weights is None else np.asarray(self.weights)
        return self.classes_[np.argmax(self.predict_proba(X) * w, axis=1)]


def build_final_model(rf, lgbm, weights, aggregate_lgbm=False):
    """
    Model final disusun: ensemble soft voting RF + LightGBM dengan bobot kelas.

    Parameters
    ----------
    rf : RandomForestClassifier
    lgbm : LGBMClassifier
    weights : array-like
        Bobot per kelas hasil optimasi.
    aggregate_lgbm : bool
        Jika True, pipeline LightGBM menggunakan fitur agregat wilayah.

    Returns
    -------
    WeightedProbaClassifier
    """
    voting = VotingClassifier([
        ("rf", build_pipeline(rf, kind="tree")),
        ("lgbm", build_pipeline(lgbm, kind="tree", aggregate=aggregate_lgbm)),
    ], voting="soft")
    return WeightedProbaClassifier(voting, weights)