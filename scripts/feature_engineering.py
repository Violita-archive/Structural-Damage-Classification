"""Transformer pembuatan fitur baru berdasarkan temuan EDA."""

from sklearn.base import BaseEstimator, TransformerMixin

from scripts.config import RC_COLS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Fitur `height_per_floor` dan `rc_count` ditambahkan ke data.

    Parameters
    ----------
    rc_cols : list of str, optional
        Kolom yang diperiksa kandungan "RC". Jika None, digunakan RC_COLS dari config.
    """

    def __init__(self, rc_cols=None):
        self.rc_cols = rc_cols

    def fit(self, X, y=None):
        """Tidak ada parameter yang dipelajari."""
        return self

    def transform(self, X):
        """
        Fitur baru dibuat pada salinan data.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        X = X.copy()
        X["height_per_floor"] = X["height_ft_pre_eq"] / X["count_floors_pre_eq"]

        rc_material = sum(X[c].str.contains("RC", na=False).astype(int) for c in (self.rc_cols or RC_COLS))
        rc_super = X["superstructure"].str.contains("rc_", na=False).astype(int)
        X["rc_count"] = rc_material + rc_super   # jumlah komponen berbahan RC (0–5)
        return X

class GroupAggregator(BaseEstimator, TransformerMixin):
    """
    Statistik bangunan per wilayah ditambahkan sebagai fitur konteks.

    Rata-rata fitur bangunan dan jumlah bangunan dihitung per wilayah pada data latih,
    kemudian digabungkan ke setiap baris berdasarkan kode wilayahnya.

    Parameters
    ----------
    group_cols : list of str, optional
        Kolom wilayah. Default ["ward_id", "vdcmun_id"].
    agg_cols : list of str, optional
        Kolom yang dirata-rata. Default umur, rc_count, jumlah lantai, dan luas.
    """

    def __init__(self, group_cols=None, agg_cols=None):
        self.group_cols = group_cols
        self.agg_cols = agg_cols

    def fit(self, X, y=None):
        """
        Statistik per wilayah dihitung dari data latih.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        self
        """
        group_cols = self.group_cols or ["ward_id", "vdcmun_id"]
        agg_cols = self.agg_cols or ["age_building", "rc_count", "count_floors_pre_eq", "plinth_area_sq_ft"]

        self.stats_ = {}
        for g in group_cols:
            stats = X.groupby(g)[agg_cols].mean().add_prefix(f"{g}_mean_")
            stats[f"{g}_count"] = X.groupby(g).size()
            self.stats_[g] = stats
        return self

    def transform(self, X):
        """
        Statistik wilayah digabungkan ke data.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        X = X.copy()
        for g, stats in self.stats_.items():
            X = X.join(stats, on=g)
        return X


    