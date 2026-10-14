"""Fungsi pemuatan model dan pembuatan file submission."""

import joblib
import pandas as pd

from scripts.config import ID_COL, TARGET, MODEL_PATH, TEST_PATH, SUBMISSION_PATH
from scripts.preprocess import load_data


def load_model(path=MODEL_PATH):
    """
    Model tersimpan dimuat dari file .pkl.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    estimator sklearn
    """
    return joblib.load(path)


def make_submission(model, test_df, path=SUBMISSION_PATH):
    """
    model : estimator sklearn
        Model final yang telah dilatih.
    test_df : pd.DataFrame
        Data test lengkap termasuk kolom ID.
    path : str or Path
        Lokasi penyimpanan file submission.

    Returns
    -------
    pd.DataFrame
        Tabel submission berisi building_id dan damage_grade.
    """
    X_test = test_df.drop(columns=[ID_COL])
    pred = model.predict(X_test)
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: pred.astype(int)})
    submission.to_csv(path, index=False)
    return submission


if __name__ == "__main__":
    model = load_model()
    sub = make_submission(model, load_data(TEST_PATH))
    print(f"Submission disimpan: {SUBMISSION_PATH} ({len(sub)} baris)")