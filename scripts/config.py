"""
Konfigurasi global proyek klasifikasi damage grade gempa Nepal 2015.

Berisi path folder, parameter umum, serta pengelompokan kolom
berdasarkan hasil EDA (notebooks/01_eda.ipynb).
"""

from pathlib import Path

# Path folder
ROOT_DIR = Path(__file__).resolve().parent.parent   # root repo ditentukan dari lokasi file ini
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
SUBMISSION_DIR = ROOT_DIR / "submissions"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

# Path file
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
MODEL_PATH = MODEL_DIR / "pipeline_model_v2.pkl"          # model final terpilih (v2)
SUBMISSION_PATH = SUBMISSION_DIR / "submission_v2.csv"   # submission final terpilih (v2)

# Parameter umum
RANDOM_STATE = 42
N_SPLITS = 5    # jumlah fold StratifiedKFold

# Kolom identitas dan target
ID_COL = "building_id"
TARGET = "damage_grade"

# Kolom lokasi
GEO_OHE_COLS = ["district_id"]                 # 11 kategori, di-OneHot
GEO_TE_COLS = ["vdcmun_id", "ward_id"]         # 110 & 945 kategori, di-TargetEncode

# Kolom numerik asli
NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]

# Fitur baru hasil feature engineering
ENGINEERED_COLS = ["height_per_floor", "rc_count"]

# Kolom kategorik
CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]

# Kolom penyusun rc_count
RC_COLS = ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type"]

# Kode nilai tidak valid yang diubah menjadi NaN
INVALID_VALUES = {
    "age_building": 999,
    "height_ft_pre_eq": 99,
}

# Kolom yang ditransformasi log1p (khusus model linear)
LOG_COLS = ["age_building", "plinth_area_sq_ft"]

# Kategori dengan proporsi di bawah nilai ini digabung oleh OneHotEncoder
OHE_MIN_FREQUENCY = 0.001   # 0,1% ≈ 533 bangunan