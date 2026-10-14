import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)


def evaluate_model(y_true, y_pred, title="Model", save_path=None):
    """
    Model dievaluasi menggunakan Macro F1, akurasi, laporan per kelas, dan confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Label sebenarnya.
    y_pred : array-like
        Label hasil prediksi.
    title : str
        Nama model untuk judul output.
    save_path : str or Path, optional
        Lokasi penyimpanan gambar confusion matrix.

    Returns
    -------
    dict
        Berisi nama model, Macro F1, dan akurasi.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    print(f"{title} | Macro F1: {macro_f1:.4f} | Akurasi: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, digits=3))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (proporsi per kelas asli) - {title}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return {"model": title, "macro_f1": macro_f1, "accuracy": acc}


def optimize_class_weights(proba, y_true, classes, grid=None, n_rounds=3):
    """
    Bobot per kelas pada probabilitas dicari untuk memaksimalkan Macro F1.

    Prediksi akhir ditentukan oleh argmax(proba × bobot). Pencarian dilakukan
    bergantian per kelas (coordinate search) sebanyak n_rounds putaran.

    Parameters
    ----------
    proba : np.ndarray, shape (n_sampel, n_kelas)
        Probabilitas prediksi.
    y_true : array-like
        Label sebenarnya.
    classes : np.ndarray
        Urutan label kelas sesuai kolom proba.
    grid : array-like, optional
        Kandidat nilai bobot. Default 0,5–2,0.
    n_rounds : int
        Jumlah putaran pencarian.

    Returns
    -------
    weights : np.ndarray
        Bobot terbaik per kelas.
    best_score : float
        Macro F1 dengan bobot terbaik.
    """
    grid = np.linspace(0.5, 2.0, 31) if grid is None else grid
    weights = np.ones(proba.shape[1])
    best_score = f1_score(y_true, classes[np.argmax(proba * weights, axis=1)], average="macro")

    for _ in range(n_rounds):
        for k in range(len(weights)):
            for g in grid:
                w_try = weights.copy()
                w_try[k] = g
                score = f1_score(y_true, classes[np.argmax(proba * w_try, axis=1)], average="macro")
                if score > best_score:
                    best_score, weights = score, w_try
    return weights, best_score