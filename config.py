# config.py
# =========
# Central Configuration for EMG_5ML_PIPELINE_40
#
# FROZEN: Final version aligned with all 18 blocks.
# WSL compatible: Uses /mnt/ paths for Linux/WSL execution.
#

from pathlib import Path
from typing import List, Dict


#
# PATHS - WSL COMPATIBLE (Change if running on native Windows)
#

# For WSL/Linux: Use /mnt/ paths
INPUT_PATH = Path("/mnt/f/Faruk/EEG_EMG_Fusion_Project/EMG_7Gesture_Raw_Dataset")
OUTPUT_ROOT = Path("/mnt/f/Faruk/EEG_EMG_Fusion_Project/EMG_7Gesture_5ML_Pipeline_40/outputs")

# For native Windows, uncomment these instead:
# INPUT_PATH = Path("F:/Faruk/EEG_EMG_Fusion_Project/EMG_7Gesture_Raw_Dataset")
# OUTPUT_ROOT = Path("F:/Faruk/EEG_EMG_Fusion_Project/EMG_7Gesture_5ML_Pipeline_40/outputs")

#
# PIPELINE METADATA
#

PIPELINE_NAME = "EMG_5ML_PIPELINE_40"
PIPELINE_VERSION = "1.0.0-90FEATURE"  # Updated to reflect feature count
PIPELINE_STAGE = "90_FEATURE_FROZEN"
FEATURE_LIBRARY_VERSION = "90_FEATURE_FINAL"
#
# GLOBAL RANDOM SEED
#

SEED = 42
#
# CHANNEL CONFIGURATION
#

N_CHANNELS = 3
EMG_CHANNEL_COLUMNS = ["EMG_ch-01", "EMG_ch-02", "EMG_ch-03"]

#
# SIGNAL PROCESSING
#

SAMPLING_RATE = 125.0
MIN_RMS_THRESHOLD = 0.012

BANDPASS_LOW = 10.0
BANDPASS_HIGH = 45.0
FILTER_ORDER = 4
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

#
# SEGMENTATION
#

WINDOW_SIZE = 256
OVERLAP = 0.5
STEP_SIZE = max(1, int(WINDOW_SIZE * (1 - OVERLAP)))

WINDOW_SIZES_GRID = [200, 256, 300, 400, 500]
OVERLAPS_GRID = [0.0, 0.25, 0.5, 0.75]

#
# FEATURE ENGINEERING
#

FEATURE_COUNT = 90  # Single source of truth for feature count
K_FEATURES = 50
VARIANCE_THRESHOLD = 1e-4
AR_ORDER = 4
WAVELET = "db4"
WAVELET_MAX_LEVEL = 4

#
# SMOTE
#

SMOTE_K_NEIGHBORS = 5
SMOTE_MIN_SAMPLES = 6

#
# CROSS-VALIDATION
#

NUM_FOLDS_MAX = 5
CALIB_CV_FOLDS = 3

#
# GESTURES
#

GESTURE_MAP: Dict[str, int] = {
    "Hand_Open_Close": 0,
    "Thumb_Finger_Movement": 1,
    "Index_Finger_Movement": 2,
    "Middle_Finger_Movement": 3,
    "Ring_Finger_Movement": 4,
    "Little_Finger_Movement": 5,
    "Pen_Holding": 6,
}
GESTURE_SHORT = ["HOC", "TFM", "IFM", "MFM", "RFM", "LFM", "PH"]
GESTURE_NAMES = list(GESTURE_MAP.keys())
N_CLASSES = len(GESTURE_MAP)
#
# MODELS
#

# Canonical model order used throughout the pipeline
MODEL_NAMES = ["LDA", "KNN", "SVM", "RF", "XGB"]
MODEL_INDEX = {
    model: idx
    for idx, model in enumerate(MODEL_NAMES)
}

N_MODELS = len(MODEL_NAMES)
#
# MODEL COLORS
# (Single source of truth for all figures)
#

MODEL_COLORS = {
    "LDA": "#4E79A7",   # Blue
    "KNN": "#F28E2B",   # Orange
    "SVM": "#59A14F",   # Green
    "RF":  "#E15759",   # Red
    "XGB": "#B07AA1",   # Purple
}

#
# MODEL HYPERPARAMETERS
# (Single source of truth)
#

LDA_PARAMS = {}

KNN_PARAMS = {
    "n_neighbors": 7,
    "n_jobs": 1,
}

SVM_PARAMS = {
    "C": 50,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "random_state": SEED,
}

RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": 1,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.1,
    "reg_lambda": 1.5,
    "reg_alpha": 0.2,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "random_state": SEED,
    "n_jobs": 1,
    "tree_method": "hist",
}

#
# MODEL PARAMETER DICTIONARY
# (For consistency across all pipeline blocks)
#

MODEL_PARAM_DICT = {
    "LDA": LDA_PARAMS,
    "KNN": KNN_PARAMS,
    "SVM": SVM_PARAMS,
    "RF": RF_PARAMS,
    "XGB": XGB_PARAMS,
}
METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro_F1",
    "Weighted_F1",
    "Balanced_Accuracy",
    "MCC",
]
#
# ROBUSTNESS PARAMETERS
#

# Gaussian Noise - Noise Standard Deviation levels (distinct from SNR)
NOISE_STD_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# SNR levels (synchronized with BLOCK_13, includes 0 dB)
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30]

# Baseline Drift levels
DRIFT_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7]

# Amplitude Scaling factors (excluding 1.0, handled as baseline)
SCALING_FACTORS = [0.5, 0.75, 1.25, 1.5, 2.0]

# Feature Dropout conditions
DROPOUT_CONDITIONS = ["1_Feature", "2_Features", "Random"]

# Signal Corruption levels
CORRUPTION_LEVELS = [5, 10, 20, 30, 40]

#
# OPT1 CONFIGURATION
#

OPT_MAX_SUBJECTS_CACHE = 6
OPT_MAX_SUBJECTS_EVAL = 4
OPT_USE_ALL_SUBJECTS = True

#
# STATISTICAL TESTS
#

N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
MAX_CALIBRATION_FOLDS = 18

#
# CALIBRATION LEVELS
#

CALIBRATION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]

#
# FEATURE NAMES - 90 Features (30 per channel x 3 channels)
#

def _generate_feature_names() -> List[str]:
    """Generate exactly 90 feature names."""
    names = []
    per_ch = [
        "MAV", "RMS", "WL", "DASDV", "WAMP", "LogVar", "TKEO", "IEMG", "ZC",
        "Hjorth_Act", "Hjorth_Mob", "Hjorth_Cplx",
        "AR1", "AR2", "AR3", "AR4",
        "WAV_E_L1", "WAV_E_L2", "WAV_E_L3", "WAV_E_L4",
        "PSD_low", "PSD_mid", "PSD_high",
        "MDF", "MNF", "Spec_Entropy",
        "Shannon_Hilb_Ent", "SampEn", "Skew", "Kurt",
    ]
    for ch in range(1, 4):
        names.extend([f"{f}_ch{ch}" for f in per_ch])
    return names

FEATURE_NAMES = _generate_feature_names()
FEATURES_PER_CHANNEL = len(FEATURE_NAMES) // N_CHANNELS

#
# FEATURE GROUPS - For Ablation Analysis (AB1)
#

FEATURE_GROUPS: Dict[str, List[str]] = {
    "TimeDomain": [f for f in FEATURE_NAMES if any(
        f.startswith(p) for p in
        ["MAV", "RMS", "WL", "DASDV", "WAMP", "LogVar", "TKEO", "IEMG", "ZC",
         "Hjorth", "AR", "Skew", "Kurt"])],
    "FreqDomain": [f for f in FEATURE_NAMES if any(
        f.startswith(p) for p in ["PSD", "MDF", "MNF", "Spec_Entropy"])],
    "TimeFreq": [f for f in FEATURE_NAMES if f.startswith("WAV_E")],
    "Complexity": [f for f in FEATURE_NAMES if any(
        f.startswith(p) for p in ["Shannon_Hilb_Ent", "SampEn"])],
}
FEATURE_GROUP_ORDER = [
    "TimeDomain",
    "FreqDomain",
    "TimeFreq",
    "Complexity",
]

FEATURE_GROUP_NAMES = FEATURE_GROUP_ORDER
#
# FIX: Feature count assertion using single FEATURE_COUNT
#

assert len(FEATURE_NAMES) == FEATURE_COUNT, \
    f"Expected {FEATURE_COUNT} features, got {len(FEATURE_NAMES)}"

#
# ARTIFACTS - Single source of truth for FE1 outputs
#

ARTIFACTS = {
    "X_features_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "X_features.npy"),
    "y_labels_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "y_labels.npy"),
    "subjects_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "subjects.npy"),
    "feature_names_json": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "feature_names.json"),
}

#
#DIRECTORY  PATHS
#

BLOCK_DIRS = {
    "PRE_A": OUTPUT_ROOT / "PRE_A_DATA_INTEGRITY",
    "PRE_B": OUTPUT_ROOT / "PRE_B_MEMORY",
    "PRE_C": OUTPUT_ROOT / "PRE_C_LOG",
    "PRE_D": OUTPUT_ROOT / "PRE_D_PREPROCESSING",
    "BLOCK_0": OUTPUT_ROOT / "BLOCK0_OPTIMIZATION",
    "BLOCK_1": OUTPUT_ROOT / "BLOCK1_LEAKAGE",
    "BLOCK_2": OUTPUT_ROOT / "BLOCK2_FEATURES",
    "BLOCK_3": OUTPUT_ROOT / "BLOCK3_LOSO",
    "BLOCK_4": OUTPUT_ROOT / "BLOCK4_SD",
    "BLOCK_5": OUTPUT_ROOT / "BLOCK5_ABLATION",
    "BLOCK_6": OUTPUT_ROOT / "BLOCK6_XAI",
    "BLOCK_7": OUTPUT_ROOT / "BLOCK7_STATS",
    "BLOCK_8": OUTPUT_ROOT / "BLOCK8_BENCHMARK",
    "BLOCK_9": OUTPUT_ROOT / "BLOCK9_CONFUSION",
    "BLOCK_10": OUTPUT_ROOT / "BLOCK10_DEPLOYMENT",
    "BLOCK_11": OUTPUT_ROOT / "BLOCK11_VISUALIZATION",
    "BLOCK_12": OUTPUT_ROOT / "BLOCK12_ERROR",
    "BLOCK_13": OUTPUT_ROOT / "BLOCK13_ROBUSTNESS",
    "LOGS": OUTPUT_ROOT / "logs",
    "TABLES": OUTPUT_ROOT / "TABLES",
    "SUPPLEMENTARY": OUTPUT_ROOT / "SUPPLEMENTARY",
}


#
# AESTHETICS
#

PALETTE = "viridis"
DPI = 300

#
# EXPECTED VALUES - For DATA1 Validation
#

EXPECTED_SUBJECTS = 40
EXPECTED_GESTURES = len(GESTURE_MAP)
EXPECTED_FILES_PER_GESTURE = None
RMS_VALIDATION_FACTOR = 0.5
REQUIRED_CHANNELS = EMG_CHANNEL_COLUMNS

#
# SUMMARY - Print configuration status on import
#

if __name__ == "__main__":
    print("=" * 60)
    print("EMG_5ML_PIPELINE_40 - Configuration Summary")
    print("=" * 60)
    print(f"Pipeline Name:    {PIPELINE_NAME}")
    print(f"Pipeline Version: {PIPELINE_VERSION}")
    print(f"Pipeline Stage:   {PIPELINE_STAGE}")
    print(f"Features:         {FEATURE_COUNT}")
    print(f"Channels:         {N_CHANNELS}")
    print(f"Subjects:         {EXPECTED_SUBJECTS}")
    print(f"Gestures:         {EXPECTED_GESTURES}")
    print(f"Models:           {MODEL_NAMES}")
    print(f"Window Size:      {WINDOW_SIZE}")
    print(f"Overlap:          {OVERLAP * 100}%")
    print(f"K Features:       {K_FEATURES}")
    print(f"Calibration Levs: {CALIBRATION_LEVELS}")
    print(f"SNR Levels:       {SNR_LEVELS}")
    print("=" * 60)