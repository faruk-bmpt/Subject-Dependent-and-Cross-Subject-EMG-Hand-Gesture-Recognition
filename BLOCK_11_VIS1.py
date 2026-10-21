# BLOCK_11_VIS1.py - COMPLETE FINAL VERSION (ALL FIXES APPLIED)
# =================
# FILE PURPOSE:
#     BLOCK 11 — VIS1: Feature Space Separability and Representation Analysis Framework.
#     Analyzes the geometric structure, separability, clustering behavior, feature relationships,
#     calibration effects, and subject variability within the EMG feature space.
#
#     REPRESENTATION ANALYSIS MODULES:
#     - VIS-01: Principal Component Analysis (PCA)
#     - VIS-02: t-SNE Representation Analysis
#     - VIS-03: UMAP Representation Analysis
#     - VIS-04: LOSO Subject Representation Analysis
#     - VIS-05: Feature Correlation Analysis
#     - VIS-06: Gesture Distance and Separability Analysis
#     - VIS-07: Subject Representation Variability Analysis
#     - VIS-08: Calibration Representation Analysis
#     - VIS-09: Feature Group Representation Analysis
#     - VIS-10: Best Performing Subject Visualization
#     - VIS-11: Worst Performing Subject Visualization
#     - VIS-12: Gesture Centroid Representation
#     - VIS-13: Correct vs Misclassified Sample Visualization (FIXED)
#     - VIS-14: Per-Gesture Accuracy Heatmap
#     - VIS-15: Gesture Centroid Distance Network
#     - VIS-16: Class Distribution Before and After Calibration
#     - VIS-17: Best Subject Summary Figure (Strongest Figure)
#
# INPUT_ARTIFACTS  :
#     - outputs/BLOCK_2/X_features.npy (REQUIRED)
#     - outputs/BLOCK_2/y_labels.npy (REQUIRED)
#     - outputs/BLOCK_2/subjects.npy (REQUIRED)
#     - outputs/BLOCK_2/feature_names.json (REQUIRED)
#     - outputs/BLOCK_2/feature_groups.json (optional)
#     - outputs/BLOCK_3/Table4_SubjectWise_Metrics.csv (for heatmap)
#     - outputs/BLOCK_3/Table_S5_PerFold_LOSO.csv (for LOSO analysis)
#     - outputs/BLOCK_3/PredictionArchive_LOSO.csv (for best/worst subject)
#     - outputs/BLOCK_3/PredictionArchive_Calibration.csv (for calibration analysis)
#     - outputs/BLOCK_7/Table_S44_Calibration_Statistics_by_Model.csv (optional)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_11/FigS2_PCA_Projection.png
#     - outputs/BLOCK_11/FigS3_tSNE_Projection.png
#     - outputs/BLOCK_11/FigS4_UMAP_Projection.png
#     - outputs/BLOCK_11/FigS5_LOSO_Subject_Heatmap.png
#     - outputs/BLOCK_11/FigS6_Feature_Correlation_Matrix.png
#     - outputs/BLOCK_11/FigS7_PCA_Explained_Variance.png
#     - outputs/BLOCK_11/FigS13_Class_Distance_Matrix.png
#     - outputs/BLOCK_11/FigS59_Subject_Similarity_Matrix.png
#     - outputs/BLOCK_11/FigS60_Subject_Representation_Projection.png
#     - outputs/BLOCK_11/FigS61_Feature_Space_Before_Calibration.png
#     - outputs/BLOCK_11/FigS62_Feature_Space_After_Calibration.png
#     - outputs/BLOCK_11/FigS63_Calibration_Performance_Improvement.png
#     - outputs/BLOCK_11/FigS64_Feature_Group_Representation.png
#     - outputs/BLOCK_11/FigS65_Best_Subject_Confusion_Matrix.png
#     - outputs/BLOCK_11/FigS66_Worst_Subject_Confusion_Matrix.png
#     - outputs/BLOCK_11/FigS67_Gesture_Centroid_Representation.png
#     - outputs/BLOCK_11/FigS68_Correct_vs_Misclassified_Samples.png
#     - outputs/BLOCK_11/FigS69_PerGesture_Accuracy_Heatmap.png
#     - outputs/BLOCK_11/FigS70_Gesture_Centroid_Network.png
#     - outputs/BLOCK_11/FigS71_Calibration_Class_Distribution.png
#     - outputs/BLOCK_11/FigS72_Best_Subject_Summary.png
#     - outputs/BLOCK_11/environment_versions.json
#     - outputs/BLOCK_11/Table_S59_Highly_Correlated_Feature_Pairs.csv
#     - outputs/BLOCK_11/Table_S60_Gesture_Distance_Statistics.csv
#     - outputs/BLOCK_11/Table_S61_Subject_Similarity_Statistics.csv
#     - outputs/BLOCK_11/Table_S62_Feature_Group_Separability_Statistics.csv
#     - outputs/BLOCK_11/Table_S63_Best_Subject_Statistics.csv
#     - outputs/BLOCK_11/Table_S64_Worst_Subject_Statistics.csv
#     - outputs/BLOCK_11/Table_S65_PerGesture_Accuracy_Summary.csv
#     - outputs/BLOCK_11/Table_S66_Gesture_Centroid_Distances.csv
#     - outputs/BLOCK_11/Table_S67_Calibration_Class_Improvement.csv
#     - outputs/BLOCK_11/Table_S68_Best_Subject_Summary.csv
#     - outputs/BLOCK_11/pca_archive/ (PCA coordinates)
#     - outputs/BLOCK_11/tsne_archive/ (t-SNE coordinates)
#     - outputs/BLOCK_11/umap_archive/ (UMAP coordinates)
#     - outputs/BLOCK_11/distance_archive/ (Distance matrices)
#     - outputs/BLOCK_11/correlation_archive/ (Correlation matrices)
#     - outputs/BLOCK_11/subject_similarity_archive/ (Subject similarity)
#     - outputs/BLOCK_11/calibration_archive/ (Calibration representation)
#     - outputs/BLOCK_11/best_subject_archive/ (Best subject confusion)
#     - outputs/BLOCK_11/worst_subject_archive/ (Worst subject confusion)
#     - outputs/BLOCK_11/centroid_archive/ (Gesture centroid data)
#     - outputs/BLOCK_11/misclassification_archive/ (Correct vs misclassified)
#     - outputs/BLOCK_11/per_gesture_archive/ (Per-gesture accuracy data)
#     - outputs/BLOCK_11/centroid_network_archive/ (Centroid network data)
#     - outputs/BLOCK_11/calibration_class_archive/ (Calibration class data)
#     - outputs/BLOCK_11/artifacts.json
#     - state/BLOCK_11_VIS1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py, common.py
# STANDALONE RUN   : python BLOCK_11_VIS1.py

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

# ── CRITICAL FIX: Disable Numba BEFORE any imports ──────────────────────────
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

# ── XGBoost Memory Safety ──────────────────────────────────────────────────
os.environ["XGBOOST_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XGBOOST_USE_CUDA"] = "0"
os.environ["XGBOOST_VERBOSITY"] = "0"

# ── FIX 1: Move classifier imports to global section ──────────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── UMAP ENABLE/DISABLE SWITCH ──────────────────────────────────────────────
# UMAP is disabled by default to prevent segmentation faults in WSL/Numba.
# Set to True only if you have verified UMAP works on your system.
ENABLE_UMAP = False

import gc
import glob
import logging
import sys
import warnings
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import scipy
import imblearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.stats import chi2
import networkx as nx

# ── FIX: Import from config ──────────────────────────────────────────────────
from config import (
    BLOCK_DIRS,
    INPUT_PATH,
    GESTURE_MAP,
    GESTURE_SHORT,
    SAMPLING_RATE,
    MIN_RMS_THRESHOLD,
    WINDOW_SIZE,
    STEP_SIZE,
    K_FEATURES,
    SEED,
    PALETTE,
    PIPELINE_NAME,
    N_CHANNELS,
    EMG_CHANNEL_COLUMNS,
    MODEL_NAMES,
    N_CLASSES,
    FEATURE_NAMES,
    REQUIRED_CHANNELS,
    EXPECTED_SUBJECTS,
    LDA_PARAMS,
    KNN_PARAMS,
    SVM_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    VARIANCE_THRESHOLD,
    SMOTE_K_NEIGHBORS,
    CALIBRATION_LEVELS,
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_feature_matrix,
    save_fig,
    save_json,
    save_csv,
    load_npy,
    load_csv,
    load_json
)

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")

# FIX: Use deterministic random generator
RNG = np.random.default_rng(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_11_VIS1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_11")

BLOCK_NAME = "BLOCK_11_VIS1"
OUT_DIR    = BLOCK_DIRS["BLOCK_11"]

# ── Archive directories ──────────────────────────────────────────────────────
PCA_ARCHIVE = OUT_DIR / "pca_archive"
TSNE_ARCHIVE = OUT_DIR / "tsne_archive"
UMAP_ARCHIVE = OUT_DIR / "umap_archive"
DISTANCE_ARCHIVE = OUT_DIR / "distance_archive"
CORRELATION_ARCHIVE = OUT_DIR / "correlation_archive"
SUBJECT_SIMILARITY_ARCHIVE = OUT_DIR / "subject_similarity_archive"
CALIBRATION_ARCHIVE = OUT_DIR / "calibration_archive"
BEST_SUBJECT_ARCHIVE = OUT_DIR / "best_subject_archive"
WORST_SUBJECT_ARCHIVE = OUT_DIR / "worst_subject_archive"
CENTROID_ARCHIVE = OUT_DIR / "centroid_archive"
MISCLASSIFICATION_ARCHIVE = OUT_DIR / "misclassification_archive"
PER_GESTURE_ARCHIVE = OUT_DIR / "per_gesture_archive"
CENTROID_NETWORK_ARCHIVE = OUT_DIR / "centroid_network_archive"
CALIBRATION_CLASS_ARCHIVE = OUT_DIR / "calibration_class_archive"

for d in [PCA_ARCHIVE, TSNE_ARCHIVE, UMAP_ARCHIVE, DISTANCE_ARCHIVE,
          CORRELATION_ARCHIVE, SUBJECT_SIMILARITY_ARCHIVE, CALIBRATION_ARCHIVE,
          BEST_SUBJECT_ARCHIVE, WORST_SUBJECT_ARCHIVE, CENTROID_ARCHIVE,
          MISCLASSIFICATION_ARCHIVE, PER_GESTURE_ARCHIVE, CENTROID_NETWORK_ARCHIVE,
          CALIBRATION_CLASS_ARCHIVE]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FIX 1: Centralized model builder
# ============================================================================

def _build_model(model_name: str, use_probability: bool = False):
    """
    Centralized model builder using config.py parameters.
    """
    if model_name == "LDA":
        return LDAClf(**LDA_PARAMS)
    elif model_name == "KNN":
        return KNeighborsClassifier(**KNN_PARAMS)
    elif model_name == "SVM":
        params = SVM_PARAMS.copy()
        if use_probability:
            params["probability"] = True
        return SVC(**params)
    elif model_name == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    elif model_name == "XGB":
        return XGBClassifier(**XGB_PARAMS)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_models() -> Dict[str, Any]:
    """Return fresh (unfitted) model instances."""
    return {
        name: _build_model(name) for name in MODEL_NAMES
    }


# ── Try to import from common with fallback ──────────────────────────────────
try:
    from common import (
        preprocess_emg,
        is_good_window,
        extract_ml_features,
        build_feature_pipeline,
        apply_feature_pipeline,
        apply_smote,
    )
    logger.info("Using centralized feature extraction from common.py")
    USING_COMMON = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from common import (
            preprocess_emg,
            is_good_window,
            extract_ml_features,
            build_feature_pipeline,
            apply_feature_pipeline,
            apply_smote,
        )
        logger.info("Using centralized feature extraction from common.py (absolute path)")
        USING_COMMON = True
    except ImportError:
        # ── Fallback: Define locally if common.py not available ────────────────
        from scipy.signal import butter, filtfilt
        from scipy import signal as scipy_signal
        from imblearn.over_sampling import SMOTE
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

        logger.warning("common.py not found - using fallback feature extraction")
        USING_COMMON = False

        # Filter parameters
        try:
            from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, NOTCH_FREQ, NOTCH_Q
        except ImportError:
            BANDPASS_LOW = 10.0
            BANDPASS_HIGH = 45.0
            FILTER_ORDER = 4
            NOTCH_FREQ = 50.0
            NOTCH_Q = 30.0

        _B_BP, _A_BP = butter(
            FILTER_ORDER,
            [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
            btype="band",
        )
        _B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)

        def preprocess_emg(data):
            data = filtfilt(_B_BP, _A_BP, data, axis=0)
            data = filtfilt(_B_NOTCH, _A_NOTCH, data, axis=0)
            return data

        def is_good_window(win):
            return bool(np.all(np.sqrt(np.mean(win ** 2, axis=0)) > MIN_RMS_THRESHOLD))

        def _detect_emg_columns(df):
            for req in REQUIRED_CHANNELS:
                if req in df.columns:
                    return REQUIRED_CHANNELS
            for req in REQUIRED_CHANNELS:
                matches = [c for c in df.columns if req.lower() in c.lower()]
                if matches:
                    return matches[:3]
            emg_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if "emg" in col_lower or "ch" in col_lower:
                    emg_cols.append(col)
            if len(emg_cols) >= 3:
                return emg_cols[:3]
            if len(df.columns) >= 3:
                return list(df.columns[-3:])
            if len(df.columns) >= 4:
                return list(df.columns[1:4])
            return []

        def extract_ml_features(w):
            """90-feature extraction."""
            from scipy.stats import kurtosis
            from scipy.signal import hilbert, welch
            import pywt
            from statsmodels.tsa.ar_model import AutoReg

            feats = []
            fs = SAMPLING_RATE
            for ch in range(N_CHANNELS):
                sig = w[:, ch].astype(np.float64)
                sig -= sig.mean()
                if np.std(sig) < 1e-8:
                    feats.extend([0.0] * 30)
                    continue

                mav = float(np.mean(np.abs(sig)))
                rms = float(np.sqrt(np.mean(sig ** 2)))
                wl = float(np.sum(np.abs(np.diff(sig))))
                dasdv = float(np.sqrt(np.mean(np.diff(sig) ** 2)))
                thr = 0.02 * np.std(sig)
                wamp = float(np.sum(np.abs(np.diff(sig)) > thr))
                logvar = float(np.log10(np.var(sig) + 1e-10))
                tkeo = float(np.mean(sig[1:-1] ** 2 - sig[:-2] * sig[2:])) if len(sig) > 2 else 0.0
                iemg = float(np.sum(np.abs(sig)))
                zc = float(np.sum((sig[:-1] * sig[1:] < 0) & (np.abs(sig[:-1] - sig[1:]) > 1e-5)) / max(len(sig) - 1, 1))

                act = float(np.var(sig))
                mob = float(np.sqrt(np.mean(np.diff(sig) ** 2) / (act + 1e-10)))
                ds = np.diff(sig)
                if len(ds) > 1:
                    mob_d1 = float(np.sqrt(np.mean(np.diff(ds) ** 2) / (np.var(ds) + 1e-10)))
                    cplx = mob_d1 / (mob + 1e-10)
                else:
                    cplx = 0.0

                skew = float(np.mean(((sig - sig.mean()) / (np.std(sig) + 1e-10)) ** 3))
                kurt = float(np.nan_to_num(kurtosis(sig, fisher=True, bias=False)))

                ar = [0.0] * 4
                try:
                    if np.std(sig) > 1e-4:
                        res = AutoReg(sig, lags=4, old_names=False).fit()
                        ar = list(res.params[1:5])
                        if len(ar) < 4:
                            ar = (ar + [0.0] * 4)[:4]
                except Exception:
                    ar = [0.0] * 4

                try:
                    max_level = min(3, pywt.dwt_max_level(len(sig), "db4"))
                    coeffs = pywt.wavedec(sig, "db4", level=max_level)
                    wav = [float(np.sum(np.abs(c) ** 2)) for c in coeffs[1:]]
                    wav = (wav + [0.0] * 4)[:4]
                except Exception:
                    wav = [0.0] * 4

                nperseg = min(len(sig), max(64, len(sig) // 2))
                fa, px = welch(sig, fs=fs, nperseg=nperseg)

                if np.sum(px) == 0:
                    low = mid = high = mdf_v = mnf_v = spec_entropy = 0.0
                else:
                    low = float(np.sum(px[(fa >= 10) & (fa < 25)]))
                    mid = float(np.sum(px[(fa >= 25) & (fa < 45)]))
                    high = float(np.sum(px[fa >= 45]))
                    cum = np.cumsum(px)
                    idx = np.where(cum >= cum[-1] / 2)[0]
                    mdf_v = float(fa[idx[0]]) if len(idx) > 0 else 0.0
                    mnf_v = float(np.sum(fa * px) / (np.sum(px) + 1e-10))
                    pn = px / (np.sum(px) + 1e-10)
                    spec_entropy = float(-np.sum(pn * np.log2(pn + 1e-10)))

                hist, _ = np.histogram(np.abs(hilbert(sig)), bins=10, density=True)
                sh_h = float(-np.sum(hist * np.log2(hist + 1e-10)))

                def sample_entropy(sig, m=2, r=None):
                    if r is None:
                        r = 0.2 * np.std(sig)
                    if r < 1e-8:
                        return 0.0
                    N = len(sig)

                    def _phi(ml):
                        x = np.array([sig[i:i + ml] for i in range(N - ml + 1)])
                        if len(x) < 2:
                            return 0.0
                        diff = np.abs(x[:, None, :] - x[None, :, :])
                        cnt = np.sum(np.max(diff, axis=2) <= r) - len(x)
                        return cnt / (len(x) * (len(x) - 1) + 1e-10)

                    try:
                        p1, p2 = _phi(m), _phi(m + 1)
                        if p1 <= 0 or p2 <= 0:
                            return 0.0
                        return float(-np.log(p2 / p1))
                    except Exception:
                        return 0.0

                sampen = sample_entropy(sig)

                feats.extend([
                    mav, rms, wl, dasdv, wamp, logvar, tkeo, iemg, zc,
                    act, mob, cplx,
                    *ar,
                    *wav,
                    low, mid, high, mdf_v, mnf_v, spec_entropy,
                    sh_h, sampen, skew, kurt
                ])

            if len(feats) != 90:
                raise ValueError(
                    f"Expected 90 features, got {len(feats)}. Feature extraction failed."
                )
            return feats

        def build_feature_pipeline(X_tr, y_tr):
            k_safe = min(K_FEATURES, X_tr.shape[1])
            scaler = RobustScaler().fit(X_tr)
            X_s = scaler.transform(X_tr)
            sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD).fit(X_s)
            X_v = sel_var.transform(X_s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel_kb = SelectKBest(f_classif, k=k_safe).fit(X_v, y_tr)
            X_sel = sel_kb.transform(X_v)
            return X_sel, scaler, sel_var, sel_kb

        def apply_feature_pipeline(X, scaler, sel_var, sel_kb):
            return sel_kb.transform(sel_var.transform(scaler.transform(X)))

        def apply_smote(X, y, tag=""):
            min_samples = min(Counter(y).values())
            if min_samples > SMOTE_K_NEIGHBORS:
                try:
                    sm = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
                    X, y = sm.fit_resample(X, y)
                except Exception:
                    pass
            return X, y

        VARIANCE_THRESHOLD

        def load_fallback_data():
            logger.info("  Loading raw data from INPUT_PATH (fallback mode)...")
            X_list, y_list, sub_list = [], [], []
            subjects = sorted(
                s for s in os.listdir(INPUT_PATH)
                if os.path.isdir(INPUT_PATH / s)
            )

            for sub in subjects:
                sub_path = INPUT_PATH / sub
                for gest, lbl in GESTURE_MAP.items():
                    files = sorted(glob.glob(str(sub_path / gest / "*.csv")))
                    for fpath in files[:3]:
                        try:
                            df_raw = pd.read_csv(fpath)
                            emg_cols = _detect_emg_columns(df_raw)
                            if len(emg_cols) < 3:
                                continue
                            df = df_raw[emg_cols].astype(float)
                            data = preprocess_emg(df.values)
                            for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                                win = data[i:i + WINDOW_SIZE]
                                if is_good_window(win):
                                    feats = extract_ml_features(win)
                                    X_list.append(feats)
                                    y_list.append(lbl)
                                    sub_list.append(sub)
                        except Exception:
                            continue

            if not X_list:
                raise RuntimeError("No data could be loaded from INPUT_PATH")

            return (
                np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.int32),
                np.array(sub_list),
            )

# ── END OF FALLBACK IMPORTS ──────────────────────────────────────────────────

# ── Try to import UMAP with graceful fallback ──────────────────────────────
try:
    import umap
    UMAP_AVAILABLE = True
    logger.info("UMAP available for visualization")
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed — UMAP projection will be skipped")


# ============================================================================
# Save environment versions and artifact manifest
# ============================================================================

def _save_environment_versions():
    """Save package versions for reproducibility."""
    env_info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "imblearn": imblearn.__version__,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    try:
        import shap
        env_info["shap"] = shap.__version__
    except ImportError:
        env_info["shap"] = "not_installed"
    
    save_json(env_info, OUT_DIR / "environment_versions.json")
    logger.info("  Environment versions saved.")


def _save_artifact_manifest(artifact_map: Dict[str, Path], out_dir: Path) -> None:
    """Save artifacts.json manifest with file verification."""
    verified_artifacts = {}
    missing_files = []
    
    for name, path in artifact_map.items():
        if path.exists():
            verified_artifacts[name] = str(path)
        else:
            missing_files.append(str(path))
            verified_artifacts[name] = f"MISSING: {str(path)}"
    
    if missing_files:
        logger.warning(f"  WARNING: {len(missing_files)} artifacts missing before manifest save")
        for mf in missing_files[:5]:
            logger.warning(f"    Missing: {mf}")
    
    manifest = {
        "block": BLOCK_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pipeline": PIPELINE_NAME,
        "artifacts": verified_artifacts,
        "missing_artifacts": missing_files if missing_files else [],
        "total_artifacts": len(artifact_map),
        "verified_artifacts": len(verified_artifacts) - len(missing_files),
    }
    manifest_path = out_dir / "artifacts.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Artifact manifest saved → {manifest_path}")


# ============================================================================
# Helper: Get best model from BLOCK_3
# ============================================================================

def _get_best_model_from_block3() -> str:
    """
    FIX 6: Automatically determine the best model from BLOCK_3.
    FIX 2: Robust column detection for accuracy column names.
    """
    try:
        loso_path = BLOCK_DIRS["BLOCK_3"] / "Table3_LOSO_Summary.csv"
        if loso_path.exists():
            df = pd.read_csv(loso_path)
            # Check several possible accuracy column names
            acc_col = None
            possible_names = ["Accuracy_Mean", "Accuracy", "Acc_Mean", "Acc", "Mean_Accuracy"]
            for col in df.columns:
                for name in possible_names:
                    if col == name or col.endswith(name):
                        acc_col = col
                        break
                if acc_col:
                    break
            
            if acc_col:
                best_model = df.loc[df[acc_col].idxmax(), "Model"]
                logger.info(f"  Best model from BLOCK_3: {best_model} (from {acc_col})")
                return best_model
            else:
                logger.warning(f"  No accuracy column found. Available columns: {df.columns.tolist()}")
    except Exception as e:
        logger.warning(f"  Could not determine best model from BLOCK_3: {e}")
    
    # Fallback to RF if detection fails
    logger.warning("  Using fallback model: RF")
    return "RF"


def _load_loso_accuracy_from_block3() -> Dict[str, float]:
    """Load LOSO accuracy from BLOCK_3 Table3_LOSO_Summary.csv."""
    loso_path = BLOCK_DIRS["BLOCK_3"] / "Table3_LOSO_Summary.csv"
    if not loso_path.exists():
        logger.warning("  LOSO summary not found - using placeholder accuracy")
        return {m: 0.7 for m in MODEL_NAMES}
    
    try:
        df = load_csv(loso_path)
        acc_col = None
        for col in df.columns:
            if "Accuracy_Mean" in col or "Acc_Mean" in col:
                acc_col = col
                break
        
        if acc_col is None:
            return {m: 0.7 for m in MODEL_NAMES}
        
        result = {}
        for _, row in df.iterrows():
            model = row["Model"]
            acc = float(row[acc_col])
            result[model] = acc
        
        return result
    except Exception as e:
        logger.warning(f"  Failed to load LOSO summary: {e}")
        return {m: 0.7 for m in MODEL_NAMES}


def _compute_gesture_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute per-gesture metrics using one-vs-rest formulation."""
    metrics = {}
    for gi, gs in enumerate(GESTURE_SHORT):
        y_true_binary = (y_true == gi).astype(int)
        y_pred_binary = (y_pred == gi).astype(int)
        
        metrics[gs] = {
            "Precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "Recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "F1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "MCC": matthews_corrcoef(y_true_binary, y_pred_binary),
            "Support": int(np.sum(y_true_binary)),
        }
    return metrics


def _plot_cm(cm: np.ndarray, title: str, path: Path, annot: bool = True) -> None:
    """Plot a single confusion matrix with 'Blues' colormap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=annot, fmt=".2f", cmap="Blues",
                xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT, ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    save_fig(fig, path)
    plt.close(fig)
    gc.collect()


# ============================================================================
# VIS-01: PCA Projection
# ============================================================================

def _plot_pca_projection(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """Figure S2 – PCA Projection."""
    logger.info("  VIS-01: PCA Projection...")
    
    n_samples = min(5000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Gesture": y_sub,
        "Gesture_Label": [GESTURE_SHORT[i] for i in y_sub]
    })
    pca_df.to_csv(PCA_ARCHIVE / "pca_coordinates.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("tab10", N_CLASSES)
    
    for i, gesture in enumerate(GESTURE_SHORT):
        mask = y_sub == i
        if np.sum(mask) > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      s=15, alpha=0.6, color=colors[i], label=gesture)
    
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% explained)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% explained)")
    ax.set_title("Figure S2 – PCA Projection of EMG Feature Space")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  PCA projection saved to {out_path}")


def _plot_pca_explained_variance(X: np.ndarray, out_path: Path) -> None:
    """Figure S7 – PCA Explained Variance."""
    logger.info("  VIS-01: PCA Explained Variance...")
    
    n_samples = min(5000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=SEED)
    pca.fit(X_scaled)
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cum_var) + 1), cum_var, 'o-', color='#1976D2', lw=2)
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    ax.axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Figure S7 – PCA Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    n_95 = np.argmax(cum_var >= 0.95) + 1
    n_99 = np.argmax(cum_var >= 0.99) + 1
    ax.axvline(x=n_95, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=n_99, color='blue', linestyle=':', alpha=0.5)
    ax.text(n_95, 0.1, f'n=95%: {n_95}', ha='center')
    ax.text(n_99, 0.05, f'n=99%: {n_99}', ha='center')
    
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  PCA explained variance saved to {out_path}")


# ============================================================================
# VIS-02: t-SNE Projection
# ============================================================================

def _plot_tsne_projection(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """Figure S3 – t-SNE Projection. max_iter=1000 for stability."""
    logger.info("  VIS-02: t-SNE Projection...")
    
    n_samples = min(1000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    try:
        tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=30,
            max_iter=1000,
            learning_rate="auto",
            init="pca",
            n_jobs=1,
            method="barnes_hut"
        )
        X_tsne = tsne.fit_transform(X_scaled)
        
        tsne_df = pd.DataFrame({
            "TSNE1": X_tsne[:, 0],
            "TSNE2": X_tsne[:, 1],
            "Gesture": y_sub,
            "Gesture_Label": [GESTURE_SHORT[i] for i in y_sub]
        })
        tsne_df.to_csv(TSNE_ARCHIVE / "tsne_coordinates.csv", index=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("tab10", N_CLASSES)
        
        for i, gesture in enumerate(GESTURE_SHORT):
            mask = y_sub == i
            if np.sum(mask) > 0:
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          s=15, alpha=0.6, color=colors[i], label=gesture)
        
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Figure S3 – t-SNE Projection of EMG Feature Space")
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()
        logger.info(f"  t-SNE projection saved to {out_path}")
    except Exception as e:
        logger.warning(f"  t-SNE projection failed: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "t-SNE projection failed", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Figure S3 – t-SNE Projection (Failed)")
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()


# ============================================================================
# VIS-03: UMAP Projection
# ============================================================================

def _plot_umap_projection(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """Figure S4 – UMAP Projection (disabled by default)."""
    if not ENABLE_UMAP:
        logger.info("  VIS-03: UMAP Projection disabled (ENABLE_UMAP=False)")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "UMAP Disabled\n(Set ENABLE_UMAP=True to enable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("Figure S4 – UMAP Projection (Disabled)")
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()
        return
    
    if not UMAP_AVAILABLE:
        logger.warning("  VIS-03: UMAP not installed")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "UMAP not installed\nInstall: pip install umap-learn",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Figure S4 – UMAP Projection (Not Available)")
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()
        return
    
    logger.info("  VIS-03: UMAP Projection...")
    
    n_samples = min(3000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    try:
        reducer = umap.UMAP(
            n_components=2,
            random_state=SEED,
            n_neighbors=15,
            low_memory=True,
            n_jobs=1,
            verbose=False
        )
        X_umap = reducer.fit_transform(X_scaled)
        
        umap_df = pd.DataFrame({
            "UMAP1": X_umap[:, 0],
            "UMAP2": X_umap[:, 1],
            "Gesture": y_sub,
            "Gesture_Label": [GESTURE_SHORT[i] for i in y_sub]
        })
        umap_df.to_csv(UMAP_ARCHIVE / "umap_coordinates.csv", index=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("tab10", N_CLASSES)
        
        for i, gesture in enumerate(GESTURE_SHORT):
            mask = y_sub == i
            if np.sum(mask) > 0:
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                          s=15, alpha=0.6, color=colors[i], label=gesture)
        
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Figure S4 – UMAP Projection of EMG Feature Space")
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()
        logger.info(f"  UMAP projection saved to {out_path}")
    except Exception as e:
        logger.warning(f"  UMAP projection failed: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"UMAP projection failed: {str(e)[:50]}...",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Figure S4 – UMAP Projection (Failed)")
        plt.tight_layout()
        save_fig(fig, out_path)
        plt.close(fig)
        gc.collect()


# ============================================================================
# VIS-04: LOSO Subject Heatmap
# ============================================================================

def _plot_loso_subject_heatmap(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, out_path: Path) -> None:
    """Figure S5 – Subject-Wise LOSO Heatmap."""
    logger.info("  VIS-04: LOSO Subject Representation...")
    
    table4_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    if table4_path.exists():
        try:
            df_subject = load_csv(table4_path)
            logger.info("  Loaded subject-wise metrics from BLOCK_3")
            heatmap_df = df_subject.pivot(index="Subject", columns="Model", values="Accuracy")
            
            fig, ax = plt.subplots(figsize=(max(8, len(heatmap_df) * 0.5), 5))
            sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu",
                        ax=ax, linewidths=0.3, vmin=0, vmax=1,
                        cbar_kws={"label": "Accuracy"})
            ax.set_title("Figure S5 – Subject-Wise LOSO Accuracy Heatmap")
            ax.set_xlabel("Model")
            ax.set_ylabel("Subject")
            plt.tight_layout()
            save_fig(fig, out_path)
            plt.close(fig)
            gc.collect()
            logger.info(f"  Subject-wise heatmap saved to {out_path}")
            return
        except Exception as e:
            logger.warning(f"  Failed to load from BLOCK_3: {e}")
    
    # Fallback: Compute from scratch
    unique_subs = np.unique(subjects)
    n_subs = len(unique_subs)
    heatmap_df = pd.DataFrame(index=unique_subs, columns=MODEL_NAMES)
    
    models = build_models()
    loso = LeaveOneGroupOut()
    
    for fold_idx, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects)):
        test_sub = subjects[te_idx[0]]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        
        X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
        X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
        X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="heatmap")
        
        for name, clf in models.items():
            try:
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                acc = accuracy_score(y_te, preds)
                heatmap_df.loc[test_sub, name] = acc
            except Exception:
                heatmap_df.loc[test_sub, name] = 0.0
        gc.collect()
    
    fig, ax = plt.subplots(figsize=(max(8, n_subs * 0.5), 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu",
                ax=ax, linewidths=0.3, vmin=0, vmax=1,
                cbar_kws={"label": "Accuracy"})
    ax.set_title("Figure S5 – Subject-Wise LOSO Accuracy Heatmap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Subject")
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Subject-wise heatmap saved to {out_path}")


# ============================================================================
# VIS-05: Feature Correlation Matrix
# ============================================================================

def _plot_feature_correlation_matrix(X: np.ndarray, feature_names: List[str], out_path: Path, n_features: int = 30) -> None:
    """Figure S6 – Feature Correlation Matrix."""
    logger.info("  VIS-05: Feature Correlation Analysis...")
    
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-n_features:]
    X_sub = X[:, top_idx]
    
    if len(feature_names) >= n_features:
        feature_labels = [feature_names[i] for i in top_idx]
    else:
        feature_labels = [f"F{i}" for i in top_idx]
    
    corr_matrix = np.corrcoef(X_sub.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    corr_df = pd.DataFrame(corr_matrix, index=feature_labels, columns=feature_labels)
    corr_df.to_csv(CORRELATION_ARCHIVE / "correlation_matrix.csv")
    
    # Find highly correlated pairs
    corr_pairs = []
    for i in range(len(feature_labels)):
        for j in range(i+1, len(feature_labels)):
            r = corr_matrix[i, j]
            if abs(r) > 0.9:
                corr_pairs.append({
                    "Feature_A": feature_labels[i],
                    "Feature_B": feature_labels[j],
                    "Correlation": round(r, 4)
                })
    
    df_pairs = pd.DataFrame(corr_pairs).sort_values("Correlation", ascending=False)
    save_csv(df_pairs, OUT_DIR / "Table_S59_Highly_Correlated_Feature_Pairs.csv")
    logger.info("  Table S59 saved.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0,
                xticklabels=feature_labels, yticklabels=feature_labels,
                ax=ax, linewidths=0.1, cbar_kws={"label": "Correlation"})
    ax.set_title(f"Figure S6 – EMG Feature Correlation Matrix (Top {n_features} Features)")
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Correlation matrix saved to {out_path}")


# ============================================================================
# VIS-06: Gesture Distance and Separability
# ============================================================================

def _compute_gesture_distances(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """VIS-06: Compute class-level distances between gestures."""
    logger.info("  VIS-06: Gesture Distance and Separability Analysis...")
    
    class_centers = {}
    for i in range(N_CLASSES):
        mask = y == i
        if np.sum(mask) > 0:
            class_centers[i] = np.mean(X[mask], axis=0)
    
    class_list = list(class_centers.keys())
    n_classes = len(class_list)
    
    distance_matrix = np.zeros((n_classes, n_classes))
    for i, ci in enumerate(class_list):
        for j, cj in enumerate(class_list):
            if i < j:
                dist = np.linalg.norm(class_centers[ci] - class_centers[cj])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    dist_df = pd.DataFrame(distance_matrix, 
                           index=[GESTURE_SHORT[i] for i in class_list],
                           columns=[GESTURE_SHORT[i] for i in class_list])
    dist_df.to_csv(DISTANCE_ARCHIVE / "euclidean_distance_matrix.csv")
    
    distance_pairs = []
    pair_id = 1
    for i, ci in enumerate(class_list):
        for j, cj in enumerate(class_list):
            if i < j:
                distance_pairs.append({
                    "Gesture_A": GESTURE_SHORT[ci],
                    "Gesture_B": GESTURE_SHORT[cj],
                    "Euclidean_Distance": round(distance_matrix[i, j], 3),
                    "Gesture_Pair_ID": f"Pair_{pair_id:02d}"
                })
                pair_id += 1
    
    df_dist_pairs = pd.DataFrame(distance_pairs).sort_values("Euclidean_Distance", ascending=False)
    save_csv(df_dist_pairs, OUT_DIR / "Table_S60_Gesture_Distance_Statistics.csv")
    logger.info("  Table S60 saved.")
    
    return dist_df


def _plot_class_distance_matrix(dist_df: pd.DataFrame, out_path: Path) -> None:
    """Figure S13 – Class Distance Matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=dist_df.columns, yticklabels=dist_df.index,
                ax=ax, cbar_kws={"label": "Euclidean Distance"})
    ax.set_title("Figure S13 – Gesture Class Distance Matrix")
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Class distance matrix saved to {out_path}")


# ============================================================================
# VIS-07: Subject Representation Variability
# ============================================================================

def _plot_subject_representation(X: np.ndarray, subjects: np.ndarray, out_path: Path) -> None:
    """Figure S60 – Subject Representation Projection."""
    logger.info("  VIS-07: Subject Representation Variability...")
    
    n_samples = min(5000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        sub_sub = subjects[idx]
    else:
        X_sub = X
        sub_sub = subjects
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    unique_subs = np.unique(sub_sub)
    n_subs = min(10, len(unique_subs))
    sub_colors = sns.color_palette("tab10", n_subs)
    sub_map = {s: sub_colors[i] for i, s in enumerate(unique_subs[:n_subs])}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for sub in unique_subs[:n_subs]:
        mask = sub_sub == sub
        if np.sum(mask) > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      s=15, alpha=0.5, color=sub_map[sub], label=str(sub))
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Figure S60 – Subject Representation Projection (PCA)")
    ax.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Subject representation projection saved to {out_path}")


# ============================================================================
# VIS-08: Calibration Representation
# ============================================================================

def _plot_calibration_representation(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """VIS-08: Calibration Representation Analysis. Figures S61, S62, S63."""
    logger.info("  VIS-08: Calibration Representation Analysis...")
    
    loso = LeaveOneGroupOut()
    tr0, te0 = next(iter(loso.split(X, y, subjects)))
    
    X_tr_raw, y_tr = X[tr0], y[tr0]
    X_te_raw, y_te = X[te0], y[te0]
    
    X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
    X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
    X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="calib_vis")
    
    # Fit PCA once on training data
    scaler_pca = RobustScaler().fit(X_tr_final)
    X_train_scaled = scaler_pca.transform(X_tr_final)
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(X_train_scaled)
    
    calib_projections = {}
    
    for cal_level in CALIBRATION_LEVELS:
        n_cal = int(len(X_te_raw) * cal_level)
        
        if n_cal > 0 and cal_level > 0:
            idx = RNG.permutation(len(X_te_raw))
            cal_idx = idx[:n_cal]
            eval_idx = idx[n_cal:]
            
            X_cal = apply_feature_pipeline(X_te_raw[cal_idx], sc, sv, sk)
            y_cal = y_te[cal_idx]
            X_eval = apply_feature_pipeline(X_te_raw[eval_idx], sc, sv, sk)
            y_eval = y_te[eval_idx]
            
            clf = _build_model("RF")
            clf.fit(np.vstack([X_tr_final, X_cal]), np.concatenate([y_tr_final, y_cal]))
            preds = clf.predict(X_eval)
        else:
            clf = _build_model("RF")
            clf.fit(X_tr_final, y_tr_final)
            preds = clf.predict(X_te_sel)
            X_eval = X_te_sel
            y_eval = y_te
        
        X_eval_scaled = scaler_pca.transform(X_eval)
        X_pca = pca.transform(X_eval_scaled)
        
        calib_projections[cal_level] = {
            "coords": X_pca,
            "labels": y_eval,
            "accuracy": accuracy_score(y_eval, preds)
        }
    
    colors = sns.color_palette("tab10", N_CLASSES)
    
    # Figure S61 – Before Calibration (0%)
    fig, ax = plt.subplots(figsize=(10, 8))
    X_pca = calib_projections[0.0]["coords"]
    y_plot = calib_projections[0.0]["labels"]
    
    for i, gesture in enumerate(GESTURE_SHORT):
        mask = y_plot == i
        if np.sum(mask) > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      s=15, alpha=0.6, color=colors[i], label=gesture)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Figure S61 – Feature Space Before Calibration (0%)")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS61_Feature_Space_Before_Calibration.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S61 saved.")
    
    # Figure S62 – After Calibration (90%)
    fig, ax = plt.subplots(figsize=(10, 8))
    X_pca = calib_projections[0.9]["coords"]
    y_plot = calib_projections[0.9]["labels"]
    
    for i, gesture in enumerate(GESTURE_SHORT):
        mask = y_plot == i
        if np.sum(mask) > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      s=15, alpha=0.6, color=colors[i], label=gesture)
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Figure S62 – Feature Space After Calibration (90%)")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS62_Feature_Space_After_Calibration.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S62 saved.")
    
    # Figure S63 – Calibration Performance Improvement
    fig, ax = plt.subplots(figsize=(10, 6))
    cal_levels = sorted(calib_projections.keys())
    accuracies = [calib_projections[level]["accuracy"] for level in cal_levels]
    cal_labels = [f"{int(level*100)}%" for level in cal_levels]
    
    ax.bar(cal_labels, accuracies, color=sns.color_palette("Blues", len(cal_levels)))
    ax.set_xlabel("Calibration Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure S63 – Calibration Performance Improvement")
    ax.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS63_Calibration_Performance_Improvement.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S63 saved.")
    
    # Save calibration archive
    for level, data in calib_projections.items():
        cal_df = pd.DataFrame({
            "PC1": data["coords"][:, 0],
            "PC2": data["coords"][:, 1],
            "Gesture": data["labels"],
            "Gesture_Label": [GESTURE_SHORT[i] for i in data["labels"]]
        })
        cal_df.to_csv(CALIBRATION_ARCHIVE / f"calibration_{int(level*100)}_coords.csv", index=False)


# ============================================================================
# VIS-09: Feature Group Representation
# ============================================================================

def _plot_feature_group_representation(X: np.ndarray, y: np.ndarray, feature_groups: Dict[str, List[str]], out_path: Path) -> None:
    """Figure S64 – Feature Group Representation Analysis."""
    logger.info("  VIS-09: Feature Group Representation Analysis...")
    
    group_masks = {}
    for group_name, group_features in feature_groups.items():
        indices = []
        for i, fname in enumerate(FEATURE_NAMES):
            if any(f in fname for f in group_features):
                indices.append(i)
        if indices:
            group_masks[group_name] = indices
    
    n_groups = len(group_masks)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols
    n_rows = max(2, n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    colors = sns.color_palette("tab10", N_CLASSES)
    
    for idx, (group_name, indices) in enumerate(group_masks.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        X_group = X[:, indices]
        
        n_samples = min(3000, len(X_group))
        if len(X_group) > n_samples:
            sample_idx = RNG.choice(len(X_group), n_samples, replace=False)
            X_sub = X_group[sample_idx]
            y_sub = y[sample_idx]
        else:
            X_sub = X_group
            y_sub = y
        
        scaler = RobustScaler().fit(X_sub)
        X_scaled = scaler.transform(X_sub)
        pca = PCA(n_components=2, random_state=SEED)
        X_pca = pca.fit_transform(X_scaled)
        
        for gi, gesture in enumerate(GESTURE_SHORT):
            mask = y_sub == gi
            if np.sum(mask) > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          s=10, alpha=0.5, color=colors[gi], label=gesture if idx == 0 else "")
        
        var_ratio = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.set_title(group_name.replace("_", " "))
        ax.grid(True, alpha=0.2)
    
    for idx in range(n_groups, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Figure S64 – Feature Group Representation Analysis (PCA)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Feature group representation saved to {out_path}")
    
    # Table S62: Feature Group Separability Statistics
    sep_stats = []
    for group_name, indices in group_masks.items():
        X_group = X[:, indices]
        overall_mean = np.mean(X_group, axis=0)
        between_var = 0
        within_var = 0
        for i in range(N_CLASSES):
            mask = y == i
            if np.sum(mask) > 0:
                class_mean = np.mean(X_group[mask], axis=0)
                between_var += np.sum(mask) * np.sum((class_mean - overall_mean) ** 2)
                within_var += np.sum(np.sum((X_group[mask] - class_mean) ** 2, axis=1))
        
        sep_score = between_var / (within_var + 1e-10)
        
        sep_stats.append({
            "Feature_Group": group_name.replace("_", " "),
            "Fisher_Separability_Index": round(sep_score, 4),
            "Variance_Explained": round(np.mean(np.var(X_group, axis=0)), 4)
        })
    
    df_sep = pd.DataFrame(sep_stats)

    if df_sep.empty:
        logger.warning("No feature group separability statistics were generated.")
        return

    required_col = "Fisher_Separability_Index"

    if required_col not in df_sep.columns:
        raise KeyError(
            f"Expected column '{required_col}' not found. "
            f"Available columns: {list(df_sep.columns)}"
        )

    df_sep = df_sep.sort_values(required_col, ascending=False)
    df_sep["Rank"] = range(1, len(df_sep) + 1)
    save_csv(df_sep, OUT_DIR / "Table_S62_Feature_Group_Separability_Statistics.csv")
    logger.info("  Table S62 saved.")


# ============================================================================
# Subject Similarity Analysis
# ============================================================================

def _compute_subject_similarity(X: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """Compute subject similarity matrix using correlation."""
    logger.info("  Computing subject similarity matrix...")
    
    unique_subs = np.unique(subjects)
    n_subs = len(unique_subs)
    
    subject_centers = {}
    for sub in unique_subs:
        mask = subjects == sub
        if np.sum(mask) > 0:
            subject_centers[sub] = np.mean(X[mask], axis=0)
    
    similarity_matrix = np.zeros((n_subs, n_subs))
    sub_list = list(subject_centers.keys())
    
    for i, sub_i in enumerate(sub_list):
        for j, sub_j in enumerate(sub_list):
            if i < j:
                corr = np.corrcoef(subject_centers[sub_i], subject_centers[sub_j])[0, 1]
                corr = np.nan_to_num(corr, nan=0.0)
                similarity = (corr + 1.0) / 2.0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    sim_df = pd.DataFrame(similarity_matrix, index=sub_list, columns=sub_list)
    sim_df.to_csv(SUBJECT_SIMILARITY_ARCHIVE / "subject_similarity_matrix.csv")
    
    sim_pairs = []
    for i, sub_i in enumerate(sub_list):
        for j, sub_j in enumerate(sub_list):
            if i < j:
                sim_pairs.append({
                    "Subject_A": str(sub_i),
                    "Subject_B": str(sub_j),
                    "Similarity_Score": round(similarity_matrix[i, j], 3)
                })
    
    df_sim_pairs = pd.DataFrame(sim_pairs).sort_values("Similarity_Score", ascending=False)
    save_csv(df_sim_pairs, OUT_DIR / "Table_S61_Subject_Similarity_Statistics.csv")
    logger.info("  Table S61 saved.")
    
    return sim_df


def _plot_subject_similarity_matrix(sim_df: pd.DataFrame, out_path: Path) -> None:
    """Figure S59 – Subject Similarity Matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sim_df.columns, yticklabels=sim_df.index,
                ax=ax, cbar_kws={"label": "Similarity Score"})
    ax.set_title("Figure S59 – Subject Similarity Matrix")
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Subject similarity matrix saved to {out_path}")


# ============================================================================
# VIS-10: Best Subject Confusion
# ============================================================================

def _plot_best_subject_confusion(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[Dict[str, float], str]:
    """VIS-10: Best Performing Subject Visualization. Figure S65 + Table S63."""
    logger.info("  VIS-10: Best Performing Subject Visualization...")
    
    table4_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    best_subject = None
    best_accuracy = 0.0
    subject_accuracies = {}
    
    if table4_path.exists():
        try:
            df_subject = load_csv(table4_path)
            for sub in df_subject["Subject"].unique():
                sub_data = df_subject[df_subject["Subject"] == sub]
                max_acc = sub_data["Accuracy"].max()
                subject_accuracies[sub] = max_acc
                if max_acc > best_accuracy:
                    best_accuracy = max_acc
                    best_subject = sub
            logger.info(f"  Best subject from BLOCK_3: {best_subject} (accuracy: {best_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load from BLOCK_3: {e}")
    
    if best_subject is None:
        logger.info("  Computing best subject from scratch...")
        loso = LeaveOneGroupOut()
        subject_accs = {}
        models = build_models()
        
        for tr_idx, te_idx in loso.split(X, y, subjects):
            test_sub = subjects[te_idx[0]]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
            X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="best_subject")
            
            for name, clf in models.items():
                try:
                    clf.fit(X_tr_final, y_tr_final)
                    preds = clf.predict(X_te_sel)
                    acc = accuracy_score(y_te, preds)
                    if test_sub not in subject_accs or acc > subject_accs[test_sub]:
                        subject_accs[test_sub] = acc
                except Exception:
                    pass
            gc.collect()
        
        best_subject = max(subject_accs, key=subject_accs.get)
        best_accuracy = subject_accs[best_subject]
        logger.info(f"  Best subject (computed): {best_subject} (accuracy: {best_accuracy:.4f})")
    
    pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
    y_true = None
    y_pred = None
    best_model = "RF"
    
    if pred_archive_path.exists():
        try:
            df_pred = pd.read_csv(pred_archive_path)
            df_sub = df_pred[df_pred["Subject"] == best_subject]
            if not df_sub.empty:
                model_accs = {}
                for model in MODEL_NAMES:
                    model_data = df_sub[df_sub["Model"] == model]
                    if not model_data.empty:
                        acc = accuracy_score(model_data["True_Label"], model_data["Pred_Label"])
                        model_accs[model] = acc
                best_model = max(model_accs, key=model_accs.get) if model_accs else "RF"
                df_best = df_sub[df_sub["Model"] == best_model]
                if not df_best.empty:
                    y_true = df_best["True_Label"].values
                    y_pred = df_best["Pred_Label"].values
                    best_accuracy = accuracy_score(y_true, y_pred)
                    logger.info(f"  Using {best_model} for subject {best_subject} (acc: {best_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load prediction archive: {e}")
    
    if y_true is None:
        logger.info(f"  Computing confusion matrix for subject {best_subject}...")
        loso = LeaveOneGroupOut()
        for tr_idx, te_idx in loso.split(X, y, subjects):
            if subjects[te_idx[0]] == best_subject:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_te, y_te = X[te_idx], y[te_idx]
                
                X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
                X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
                X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="best_subject_cm")
                
                clf = _build_model("RF")
                clf.fit(X_tr_final, y_tr_final)
                y_pred = clf.predict(X_te_sel)
                y_true = y_te
                best_accuracy = accuracy_score(y_true, y_pred)
                break
    
    if y_true is None or len(y_true) == 0:
        logger.warning("  Could not generate confusion matrix for best subject")
        return {}, best_subject
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    cm_df = pd.DataFrame(cm_norm, index=GESTURE_SHORT, columns=GESTURE_SHORT)
    cm_df.to_csv(BEST_SUBJECT_ARCHIVE / f"best_subject_{best_subject}_cm.csv")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                ax=ax, cbar_kws={"label": "Proportion (Row-wise)"},
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted Gesture", fontsize=12)
    ax.set_ylabel("True Gesture", fontsize=12)
    ax.set_title(f"Figure S65\nBest Performing Subject ({best_subject})\nNormalized Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS65_Best_Subject_Confusion_Matrix.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S65 saved.")
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    stats = {
        "Subject": best_subject,
        "Accuracy": best_accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1,
        "Balanced_Accuracy": bal_acc,
    }
    
    df_stats = pd.DataFrame([stats])
    save_csv(df_stats, OUT_DIR / "Table_S63_Best_Subject_Statistics.csv")
    logger.info("  Table S63 saved.")
    
    return stats, best_subject


# ============================================================================
# VIS-11: Worst Subject Confusion
# ============================================================================

def _plot_worst_subject_confusion(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[Dict[str, float], str]:
    """VIS-11: Worst Performing Subject Visualization. Figure S66 + Table S64."""
    logger.info("  VIS-11: Worst Performing Subject Visualization...")
    
    table4_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    worst_subject = None
    worst_accuracy = 1.0
    
    if table4_path.exists():
        try:
            df_subject = load_csv(table4_path)
            for sub in df_subject["Subject"].unique():
                sub_data = df_subject[df_subject["Subject"] == sub]
                max_acc = sub_data["Accuracy"].max()
                if max_acc < worst_accuracy:
                    worst_accuracy = max_acc
                    worst_subject = sub
            logger.info(f"  Worst subject from BLOCK_3: {worst_subject} (accuracy: {worst_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load from BLOCK_3: {e}")
    
    if worst_subject is None:
        logger.info("  Computing worst subject from scratch...")
        loso = LeaveOneGroupOut()
        subject_accs = {}
        models = build_models()
        
        for tr_idx, te_idx in loso.split(X, y, subjects):
            test_sub = subjects[te_idx[0]]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
            X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="worst_subject")
            
            for name, clf in models.items():
                try:
                    clf.fit(X_tr_final, y_tr_final)
                    preds = clf.predict(X_te_sel)
                    acc = accuracy_score(y_te, preds)
                    if test_sub not in subject_accs or acc > subject_accs[test_sub]:
                        subject_accs[test_sub] = acc
                except Exception:
                    pass
            gc.collect()
        
        worst_subject = min(subject_accs, key=subject_accs.get)
        worst_accuracy = subject_accs[worst_subject]
        logger.info(f"  Worst subject (computed): {worst_subject} (accuracy: {worst_accuracy:.4f})")
    
    pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
    y_true = None
    y_pred = None
    best_model = "RF"
    
    if pred_archive_path.exists():
        try:
            df_pred = pd.read_csv(pred_archive_path)
            df_sub = df_pred[df_pred["Subject"] == worst_subject]
            if not df_sub.empty:
                model_accs = {}
                for model in MODEL_NAMES:
                    model_data = df_sub[df_sub["Model"] == model]
                    if not model_data.empty:
                        acc = accuracy_score(model_data["True_Label"], model_data["Pred_Label"])
                        model_accs[model] = acc
                best_model = max(model_accs, key=model_accs.get) if model_accs else "RF"
                df_best = df_sub[df_sub["Model"] == best_model]
                if not df_best.empty:
                    y_true = df_best["True_Label"].values
                    y_pred = df_best["Pred_Label"].values
                    worst_accuracy = accuracy_score(y_true, y_pred)
                    logger.info(f"  Using {best_model} for subject {worst_subject} (acc: {worst_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load prediction archive: {e}")
    
    if y_true is None:
        logger.info(f"  Computing confusion matrix for subject {worst_subject}...")
        loso = LeaveOneGroupOut()
        for tr_idx, te_idx in loso.split(X, y, subjects):
            if subjects[te_idx[0]] == worst_subject:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_te, y_te = X[te_idx], y[te_idx]
                
                X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
                X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
                X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="worst_subject_cm")
                
                clf = _build_model("RF")
                clf.fit(X_tr_final, y_tr_final)
                y_pred = clf.predict(X_te_sel)
                y_true = y_te
                worst_accuracy = accuracy_score(y_true, y_pred)
                break
    
    if y_true is None or len(y_true) == 0:
        logger.warning("  Could not generate confusion matrix for worst subject")
        return {}, worst_subject
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    cm_df = pd.DataFrame(cm_norm, index=GESTURE_SHORT, columns=GESTURE_SHORT)
    cm_df.to_csv(WORST_SUBJECT_ARCHIVE / f"worst_subject_{worst_subject}_cm.csv")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                ax=ax, cbar_kws={"label": "Proportion (Row-wise)"},
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted Gesture", fontsize=12)
    ax.set_ylabel("True Gesture", fontsize=12)
    ax.set_title(f"Figure S66\nWorst Performing Subject ({worst_subject})\nNormalized Confusion Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS66_Worst_Subject_Confusion_Matrix.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S66 saved.")
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    stats = {
        "Subject": worst_subject,
        "Accuracy": worst_accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1,
        "Balanced_Accuracy": bal_acc,
    }
    
    df_stats = pd.DataFrame([stats])
    save_csv(df_stats, OUT_DIR / "Table_S64_Worst_Subject_Statistics.csv")
    logger.info("  Table S64 saved.")
    
    return stats, worst_subject


# ============================================================================
# VIS-12: Gesture Centroid Representation
# ============================================================================

def _plot_gesture_centroid_representation(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """VIS-12: Gesture Centroid Representation. Figure S67."""
    logger.info("  VIS-12: Gesture Centroid Representation...")
    
    n_samples = min(5000, len(X))
    if len(X) > n_samples:
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
    
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    
    colors = sns.color_palette("tab10", N_CLASSES)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    centroids = {}
    
    for i, gesture in enumerate(GESTURE_SHORT):
        mask = y_sub == i
        if np.sum(mask) > 0:
            points = X_pca[mask]
            centroid = np.mean(points, axis=0)
            centroids[gesture] = centroid
            
            ax.scatter(points[:, 0], points[:, 1],
                      s=15, alpha=0.3, color=colors[i], label=gesture)
            
            ax.scatter(centroid[0], centroid[1],
                      s=200, marker='X', color=colors[i],
                      edgecolor='black', linewidth=1.5, zorder=5)
            
            if len(points) > 2:
                try:
                    cov = np.cov(points.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    chi2_val = chi2.ppf(0.95, 2)
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    width = 2 * np.sqrt(chi2_val * eigvals[0])
                    height = 2 * np.sqrt(chi2_val * eigvals[1])
                    
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse(xy=centroid, width=width, height=height,
                                      angle=angle, edgecolor=colors[i],
                                      facecolor='none', linewidth=2, linestyle='--')
                    ax.add_patch(ellipse)
                except Exception:
                    pass
    
    for gesture, centroid in centroids.items():
        ax.annotate(gesture, xy=(centroid[0], centroid[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% explained)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% explained)")
    ax.set_title("Figure S67 – Gesture Centroid Representation\nwith 95% Confidence Ellipses", fontsize=14, fontweight="bold")
    ax.legend(markerscale=2, fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Gesture centroid representation saved to {out_path}")
    
    centroid_df = pd.DataFrame([
        {"Gesture": g, "PC1": c[0], "PC2": c[1]} 
        for g, c in centroids.items()
    ])
    centroid_df.to_csv(CENTROID_ARCHIVE / "gesture_centroids.csv", index=False)


# ============================================================================
# VIS-13: Correct vs Misclassified Samples (FIXED - PUBLICATION GRADE)
# ============================================================================

def _plot_correct_vs_misclassified(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, out_path: Path) -> None:
    """
    VIS-13: Correct vs Misclassified Samples. Figure S68.
    
    FIXED (Publication Grade):
    1. Samples BOTH feature matrix and prediction vectors with same index
    2. Validates alignment before proceeding
    3. Handles mismatched lengths gracefully
    4. Properly aligned PCA and predictions
    5. Fixed DataFrame column assignment
    """
    logger.info("  VIS-13: Correct vs Misclassified Sample Visualization...")
    
    best_model_name = "RF"
    pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
    y_true_global = None
    y_pred_global = None
    
    if pred_archive_path.exists():
        try:
            df_pred = pd.read_csv(pred_archive_path)
            df_model = df_pred[df_pred["Model"] == best_model_name]
            if not df_model.empty:
                y_true_global = df_model["True_Label"].values
                y_pred_global = df_model["Pred_Label"].values
                logger.info(f"  Loaded predictions for {best_model_name} from BLOCK_3")
                
                # Check if predictions length matches X
                if len(y_true_global) != len(X):
                    logger.warning(
                        f"  Prediction archive length ({len(y_true_global)}) "
                        f"!= feature matrix ({len(X)}). Will recompute aligned predictions."
                    )
                    y_true_global = None
                    y_pred_global = None
        except Exception as e:
            logger.warning(f"  Failed to load prediction archive: {e}")
            y_true_global = None
            y_pred_global = None
    
    if y_true_global is None or len(y_true_global) == 0:
        logger.info("  Computing aligned predictions across all LOSO folds...")
        loso = LeaveOneGroupOut()
        all_y_true = []
        all_y_pred = []
        all_indices = []
        fold_count = 0
        
        for tr_idx, te_idx in loso.split(X, y, subjects):
            fold_count += 1
            if fold_count % 5 == 0 or fold_count == 1:
                logger.info(f"    Processing fold {fold_count}...")
            
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
            X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"misclass_fold_{fold_count}")
            
            clf = _build_model(best_model_name)
            clf.fit(X_tr_final, y_tr_final)
            preds = clf.predict(X_te_sel)
            
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(preds.tolist())
            all_indices.extend(te_idx.tolist())
            gc.collect()
        
        # Convert to numpy arrays
        y_true_global = np.array(all_y_true)
        y_pred_global = np.array(all_y_pred)
        all_indices = np.array(all_indices)
        
        logger.info(f"  Computed predictions for {len(all_indices)} samples across {fold_count} folds")
        
        # Reorder predictions to match X order
        y_true_aligned = np.zeros(len(X), dtype=np.int32)
        y_pred_aligned = np.zeros(len(X), dtype=np.int32)
        for idx, true_lbl, pred_lbl in zip(all_indices, y_true_global, y_pred_global):
            y_true_aligned[idx] = true_lbl
            y_pred_aligned[idx] = pred_lbl
        
        y_true_global = y_true_aligned
        y_pred_global = y_pred_aligned
        
        # Verify alignment
        if len(y_true_global) != len(X):
            logger.error(f"  Alignment failed: {len(y_true_global)} vs {len(X)}")
            return
    
    # Now we have aligned predictions and features
    logger.info(f"  Using {len(y_true_global)} aligned predictions with feature matrix")
    
    # ------------------------------------------------------------
    # SAMPLE BOTH feature matrix AND prediction vectors together
    # ------------------------------------------------------------
    n_samples = min(5000, len(X))
    
    if len(X) > n_samples:
        # Sample indices once and apply to both X and predictions
        idx = RNG.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]
        y_true_sub = y_true_global[idx]
        y_pred_sub = y_pred_global[idx]
        logger.info(f"  Sampled {n_samples} samples for visualization (from {len(X)} total)")
    else:
        X_sub = X
        y_true_sub = y_true_global
        y_pred_sub = y_pred_global
    
    # ------------------------------------------------------------
    # SAFETY CHECK: Verify alignment before proceeding
    # ------------------------------------------------------------
    if len(X_sub) != len(y_true_sub):
        raise RuntimeError(
            f"Mismatch after sampling: PCA samples={len(X_sub)}, "
            f"Predictions={len(y_true_sub)}. This should never happen."
        )
    
    # PCA on the sampled data
    scaler = RobustScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    
    # Now correct_mask matches X_pca exactly
    correct_mask = (y_true_sub == y_pred_sub)
    
    # ------------------------------------------------------------
    # Create the visualization
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mis_idx = ~correct_mask
    if np.sum(mis_idx) > 0:
        ax.scatter(X_pca[mis_idx, 0], X_pca[mis_idx, 1],
                  s=30, alpha=0.7, color='red', 
                  label=f'Misclassified ({np.sum(mis_idx)})', zorder=2)
    
    corr_idx = correct_mask
    if np.sum(corr_idx) > 0:
        ax.scatter(X_pca[corr_idx, 0], X_pca[corr_idx, 1],
                  s=20, alpha=0.5, color='green', 
                  label=f'Correct ({np.sum(corr_idx)})', zorder=3)
    
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% explained)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% explained)")
    ax.set_title("Figure S68 – Correct vs Misclassified Samples in Feature Space\n"
                 f"{best_model_name} LOSO Predictions (Aligned)",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=2, fontsize=10, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    gc.collect()
    logger.info(f"  Correct vs misclassified visualization saved to {out_path}")
    
    # ------------------------------------------------------------
    # Save the data with correct column assignment
    # ------------------------------------------------------------
    mis_df = pd.DataFrame({
        "PC1": X_pca[:, 0],          # FIXED: Use X_pca[:,0] not X_pca
        "PC2": X_pca[:, 1],          # FIXED: Use X_pca[:,1] not X_pca[:,1]
        "True_Label": y_true_sub,
        "Pred_Label": y_pred_sub,
        "Correct": correct_mask
    })
    mis_df.to_csv(MISCLASSIFICATION_ARCHIVE / "correct_vs_misclassified.csv", index=False)
    
    n_correct = np.sum(correct_mask)
    n_total = len(correct_mask)
    logger.info(f"  Correct: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")


# ============================================================================
# VIS-14: Per-Gesture Accuracy Heatmap
# ============================================================================

def _plot_per_gesture_accuracy_heatmap(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """VIS-14: Per-Gesture Accuracy Heatmap. Figure S69 + Table S65."""
    logger.info("  VIS-14: Per-Gesture Accuracy Heatmap...")
    
    table4_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    
    if table4_path.exists():
        try:
            df_subject = load_csv(table4_path)
            logger.info("  Loaded subject-wise metrics from BLOCK_3")
            
            pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
            
            if pred_archive_path.exists():
                df_pred = pd.read_csv(pred_archive_path)
                
                heatmap_data = []
                for sub in df_subject["Subject"].unique():
                    sub_pred = df_pred[df_pred["Subject"] == sub]
                    if sub_pred.empty:
                        continue
                    
                    sub_metrics = df_subject[df_subject["Subject"] == sub]
                    best_model = sub_metrics.loc[sub_metrics["Accuracy"].idxmax(), "Model"]
                    
                    sub_best = sub_pred[sub_pred["Model"] == best_model]
                    if sub_best.empty:
                        continue
                    
                    y_true = sub_best["True_Label"].values
                    y_pred = sub_best["Pred_Label"].values
                    
                    row = {"Subject": sub}
                    for gi, gest in enumerate(GESTURE_SHORT):
                        mask = y_true == gi
                        if np.sum(mask) > 0:
                            acc = np.mean(y_pred[mask] == y_true[mask])
                        else:
                            acc = np.nan
                        row[gest] = acc
                    heatmap_data.append(row)
                
                if heatmap_data:
                    df_heatmap = pd.DataFrame(heatmap_data)
                    df_heatmap = df_heatmap.set_index("Subject")
                    
                    df_heatmap["Avg_Acc"] = df_heatmap.mean(axis=1)
                    df_heatmap = df_heatmap.sort_values("Avg_Acc", ascending=False)
                    df_heatmap = df_heatmap.drop(columns=["Avg_Acc"])
                    
                    df_heatmap.to_csv(PER_GESTURE_ARCHIVE / "per_gesture_accuracy_heatmap.csv")
                    
                    summary_rows = []
                    for gest in GESTURE_SHORT:
                        gest_data = df_heatmap[gest].dropna()
                        summary_rows.append({
                            "Gesture": gest,
                            "Mean_Accuracy": gest_data.mean(),
                            "Std_Accuracy": gest_data.std(),
                            "Min_Accuracy": gest_data.min(),
                            "Max_Accuracy": gest_data.max(),
                            "N_Subjects": len(gest_data)
                        })
                    df_summary = pd.DataFrame(summary_rows)
                    save_csv(df_summary, OUT_DIR / "Table_S65_PerGesture_Accuracy_Summary.csv")
                    logger.info("  Table S65 saved.")
                    
                    fig, ax = plt.subplots(figsize=(12, max(6, len(df_heatmap) * 0.4)))
                    sns.heatmap(
                        df_heatmap,
                        annot=True,
                        fmt=".2f",
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=1,
                        linewidths=0.3,
                        linecolor="white",
                        ax=ax,
                        cbar_kws={"label": "Accuracy", "shrink": 0.8}
                    )
                    ax.set_title(
                        "Figure S69 – Per-Gesture Accuracy Heatmap\n"
                        "(Green = High Accuracy, Red = Poor Accuracy)",
                        fontsize=14,
                        fontweight="bold"
                    )
                    ax.set_xlabel("Gesture")
                    ax.set_ylabel("Subject")
                    ax.set_xticklabels(GESTURE_SHORT, rotation=45, ha="right")
                    
                    plt.tight_layout()
                    save_fig(fig, OUT_DIR / "FigS69_PerGesture_Accuracy_Heatmap.png")
                    plt.close(fig)
                    gc.collect()
                    logger.info("  Figure S69 saved.")
                    return
        
        except Exception as e:
            logger.warning(f"  Failed to load from BLOCK_3: {e}")
    
    logger.info("  Computing per-gesture accuracy heatmap from scratch...")
    loso = LeaveOneGroupOut()
    models = build_models()
    
    heatmap_data = []
    
    for tr_idx, te_idx in loso.split(X, y, subjects):
        test_sub = subjects[te_idx[0]]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        
        X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
        X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
        X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="per_gesture")
        
        best_acc = -1
        best_pred = None
        
        for name, clf in models.items():
            try:
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                acc = accuracy_score(y_te, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_pred = preds
            except Exception:
                pass
        
        if best_pred is not None:
            row = {"Subject": test_sub}
            for gi, gest in enumerate(GESTURE_SHORT):
                mask = y_te == gi
                if np.sum(mask) > 0:
                    acc = np.mean(best_pred[mask] == y_te[mask])
                else:
                    acc = np.nan
                row[gest] = acc
            heatmap_data.append(row)
        gc.collect()
    
    if not heatmap_data:
        logger.warning("  Could not generate per-gesture accuracy heatmap")
        return
    
    df_heatmap = pd.DataFrame(heatmap_data)
    df_heatmap = df_heatmap.set_index("Subject")
    
    df_heatmap["Avg_Acc"] = df_heatmap.mean(axis=1)
    df_heatmap = df_heatmap.sort_values("Avg_Acc", ascending=False)
    df_heatmap = df_heatmap.drop(columns=["Avg_Acc"])
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_heatmap) * 0.4)))
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Accuracy", "shrink": 0.8}
    )
    ax.set_title(
        "Figure S69 – Per-Gesture Accuracy Heatmap\n"
        "(Green = High Accuracy, Red = Poor Accuracy)",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Subject")
    ax.set_xticklabels(GESTURE_SHORT, rotation=45, ha="right")
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS69_PerGesture_Accuracy_Heatmap.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S69 saved.")


# ============================================================================
# VIS-15: Gesture Centroid Distance Network (IMPROVED)
# ============================================================================

def _plot_gesture_centroid_network(X: np.ndarray, y: np.ndarray) -> None:
    """
    VIS-15: Gesture Centroid Distance Network. Figure S70 + Table S66.
    
    FIX 1: Removed duplicate edge labels
    FIX 2: Consistent node size (700)
    FIX 3: Only connect top 3 nearest neighbors
    """
    logger.info("  VIS-15: Gesture Centroid Distance Network...")
    
    class_centers = {}
    for i in range(N_CLASSES):
        mask = y == i
        if np.sum(mask) > 0:
            class_centers[i] = np.mean(X[mask], axis=0)
    
    class_list = list(class_centers.keys())
    n_classes = len(class_list)
    
    distance_matrix = np.zeros((n_classes, n_classes))
    for i, ci in enumerate(class_list):
        for j, cj in enumerate(class_list):
            if i < j:
                dist = np.linalg.norm(class_centers[ci] - class_centers[cj])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    G = nx.Graph()
    node_labels = {}
    for i, ci in enumerate(class_list):
        label = GESTURE_SHORT[ci]
        G.add_node(i, label=label)
        node_labels[i] = label
    
    # Only connect top 3 nearest neighbors
    for i in range(n_classes):
        distances = [(j, distance_matrix[i, j]) for j in range(n_classes) if j != i]
        distances.sort(key=lambda x: x[1])
        top_k = distances[:3]
        for j, dist in top_k:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=dist)
    
    edge_weights = [data['weight'] for (u, v, data) in G.edges(data=True)]
    max_dist = max(edge_weights) if edge_weights else 1.0
    min_dist = min(edge_weights) if edge_weights else 0.0
    
    edge_rows = []
    for (u, v, data) in G.edges(data=True):
        edge_rows.append({
            "Gesture_A": node_labels[u],
            "Gesture_B": node_labels[v],
            "Distance": round(data["weight"], 3),
            "Similarity": round(1 / (1 + data["weight"]), 3)
        })
    df_edges = pd.DataFrame(edge_rows).sort_values("Distance", ascending=True)
    save_csv(df_edges, OUT_DIR / "Table_S66_Gesture_Centroid_Distances.csv")
    logger.info("  Table S66 saved.")
    
    nx.write_gml(G, CENTROID_NETWORK_ARCHIVE / "gesture_centroid_network.gml")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, seed=SEED, iterations=100)
    
    colors = sns.color_palette("tab10", n_classes)
    node_colors = [colors[i] for i in range(n_classes)]
    
    node_size = 700
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    edges = G.edges(data=True)
    edge_widths = []
    for (u, v, data) in edges:
        dist = data['weight']
        if max_dist > min_dist:
            normalized = (dist - min_dist) / (max_dist - min_dist + 0.001)
            width = 0.5 + 3.5 * (1 - normalized)
        else:
            width = 2.0
        edge_widths.append(width)
    
    for i, ((u, v, data), width) in enumerate(zip(edges, edge_widths)):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=0.6,
            edge_color='#666666',
            ax=ax
        )
    
    labels_dict = {i: node_labels[i] for i in range(n_classes)}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels_dict,
        font_size=12,
        font_weight='bold',
        ax=ax
    )
    
    # Only store edge labels in one direction (u, v)
    edge_labels = {}
    for (u, v, data) in G.edges(data=True):
        if (u, v) not in edge_labels and (v, u) not in edge_labels:
            edge_labels[(u, v)] = f"{data['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )
    
    ax.set_title(
        "Figure S70 – Gesture Centroid Distance Network\n"
        "(Top 3 Nearest Neighbors Only, Edge Thickness ∝ Similarity)",
        fontsize=14,
        fontweight="bold"
    )
    ax.axis('off')
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS70_Gesture_Centroid_Network.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S70 saved.")


# ============================================================================
# VIS-16: Calibration Class Distribution (IMPROVED)
# ============================================================================

def _plot_calibration_class_distribution(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """
    VIS-16: Class Distribution Before and After Calibration. Figure S71 + Table S67.
    
    FIX 5: Plot Accuracy (%) instead of raw counts.
    FIX 6: Automatically determine best model from BLOCK_3.
    FIX 7: Average across all LOSO folds.
    FIX 1: Use fold-specific RNG seed.
    """
    logger.info("  VIS-16: Calibration Class Distribution Analysis...")
    
    best_model = _get_best_model_from_block3()
    
    pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_Calibration.csv"
    
    cal_0_acc = {}
    cal_90_acc = {}
    cal_0_counts = {}
    cal_90_counts = {}
    cal_0_total = {}
    cal_90_total = {}
    
    if pred_archive_path.exists():
        try:
            df_pred = pd.read_csv(pred_archive_path)
            logger.info(f"  Loaded calibration predictions from BLOCK_3 (using {best_model})")
            
            df_0 = df_pred[(df_pred["Calibration_Level"] == 0.0) & (df_pred["Model"] == best_model)]
            if not df_0.empty:
                for gi, gest in enumerate(GESTURE_SHORT):
                    mask = df_0["True_Label"] == gi
                    if np.sum(mask) > 0:
                        sub_df = df_0[mask]
                        correct = np.sum(sub_df["True_Label"] == sub_df["Pred_Label"])
                        total = len(sub_df)
                        cal_0_acc[gest] = correct / total if total > 0 else 0.0
                        cal_0_counts[gest] = correct
                        cal_0_total[gest] = total
            
            df_90 = df_pred[(df_pred["Calibration_Level"] == 0.9) & (df_pred["Model"] == best_model)]
            if not df_90.empty:
                for gi, gest in enumerate(GESTURE_SHORT):
                    mask = df_90["True_Label"] == gi
                    if np.sum(mask) > 0:
                        sub_df = df_90[mask]
                        correct = np.sum(sub_df["True_Label"] == sub_df["Pred_Label"])
                        total = len(sub_df)
                        cal_90_acc[gest] = correct / total if total > 0 else 0.0
                        cal_90_counts[gest] = correct
                        cal_90_total[gest] = total
        
        except Exception as e:
            logger.warning(f"  Failed to load calibration archive: {e}")
    
    if not cal_0_acc:
        logger.info(f"  Computing calibration class distribution from scratch using {best_model}...")
        loso = LeaveOneGroupOut()
        
        all_y_true_0 = []
        all_y_pred_0 = []
        all_y_true_90 = []
        all_y_pred_90 = []
        
        fold_count = 0
        total_folds = len(list(loso.split(X, y, subjects)))
        
        for tr_idx, te_idx in loso.split(X, y, subjects):
            fold_count += 1
            logger.info(f"    Processing fold {fold_count}/{total_folds}")
            
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"calib_class_fold{fold_count}")
            
            clf = _build_model(best_model)
            clf.fit(X_tr_final, y_tr_final)
            X_te_sel_0 = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            preds_0 = clf.predict(X_te_sel_0)
            
            all_y_true_0.extend(y_te.tolist())
            all_y_pred_0.extend(preds_0.tolist())
            
            n_cal = int(len(X_te_raw) * 0.9)
            if n_cal > 0:
                rng = np.random.default_rng(SEED + fold_count)
                idx = rng.permutation(len(X_te_raw))
                cal_idx = idx[:n_cal]
                eval_idx = idx[n_cal:]
                
                X_cal = apply_feature_pipeline(X_te_raw[cal_idx], sc, sv, sk)
                y_cal = y_te[cal_idx]
                X_eval = apply_feature_pipeline(X_te_raw[eval_idx], sc, sv, sk)
                y_eval = y_te[eval_idx]
                
                clf_cal = _build_model(best_model)
                clf_cal.fit(np.vstack([X_tr_final, X_cal]), np.concatenate([y_tr_final, y_cal]))
                preds_90 = clf_cal.predict(X_eval)
                
                all_y_true_90.extend(y_eval.tolist())
                all_y_pred_90.extend(preds_90.tolist())
            else:
                all_y_true_90.extend(y_te.tolist())
                all_y_pred_90.extend(preds_0.tolist())
            
            gc.collect()
        
        y_true_0 = np.array(all_y_true_0)
        y_pred_0 = np.array(all_y_pred_0)
        y_true_90 = np.array(all_y_true_90)
        y_pred_90 = np.array(all_y_pred_90)
        
        for gi, gest in enumerate(GESTURE_SHORT):
            mask_0 = y_true_0 == gi
            if np.sum(mask_0) > 0:
                correct_0 = np.sum(y_pred_0[mask_0] == y_true_0[mask_0])
                total_0 = np.sum(mask_0)
                cal_0_acc[gest] = correct_0 / total_0 if total_0 > 0 else 0.0
                cal_0_counts[gest] = correct_0
                cal_0_total[gest] = total_0
            else:
                cal_0_acc[gest] = 0.0
                cal_0_counts[gest] = 0
                cal_0_total[gest] = 0
            
            mask_90 = y_true_90 == gi
            if np.sum(mask_90) > 0:
                correct_90 = np.sum(y_pred_90[mask_90] == y_true_90[mask_90])
                total_90 = np.sum(mask_90)
                cal_90_acc[gest] = correct_90 / total_90 if total_90 > 0 else 0.0
                cal_90_counts[gest] = correct_90
                cal_90_total[gest] = total_90
            else:
                cal_90_acc[gest] = 0.0
                cal_90_counts[gest] = 0
                cal_90_total[gest] = 0
    
    # Table S67
    improvement_rows = []
    for gest in GESTURE_SHORT:
        improvement_rows.append({
            "Gesture": gest,
            "Correct_0%": cal_0_counts.get(gest, 0),
            "Total_0%": cal_0_total.get(gest, 0),
            "Accuracy_0%": round(cal_0_acc.get(gest, 0) * 100, 1),
            "Correct_90%": cal_90_counts.get(gest, 0),
            "Total_90%": cal_90_total.get(gest, 0),
            "Accuracy_90%": round(cal_90_acc.get(gest, 0) * 100, 1),
            "Acc_Improvement": round((cal_90_acc.get(gest, 0) - cal_0_acc.get(gest, 0)) * 100, 1)
        })
    df_improvement = pd.DataFrame(improvement_rows)
    save_csv(df_improvement, OUT_DIR / "Table_S67_Calibration_Class_Improvement.csv")
    logger.info("  Table S67 saved.")
    
    # Figure S71
    gestures = GESTURE_SHORT
    acc_0 = [cal_0_acc.get(g, 0.0) * 100 for g in gestures]
    acc_90 = [cal_90_acc.get(g, 0.0) * 100 for g in gestures]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(gestures))
    width = 0.35
    
    improvements = [acc_90[i] - acc_0[i] for i in range(len(gestures))]
    sorted_idx = np.argsort(improvements)[::-1]
    
    sorted_gestures = [gestures[i] for i in sorted_idx]
    sorted_acc_0 = [acc_0[i] for i in sorted_idx]
    sorted_acc_90 = [acc_90[i] for i in sorted_idx]
    
    bars1 = ax.bar(x - width/2, sorted_acc_0, width, label='0% Calibration', 
                   color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x + width/2, sorted_acc_90, width, label='90% Calibration', 
                   color='#4ECDC4', edgecolor='black')
    
    ax.set_xlabel('Gesture')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Figure S71 – Per-Gesture Accuracy Before and After Calibration\n'
                 f'(Using {best_model}, 0% vs 90% Calibration)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_gestures, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2, height1 + 1.5,
                f'{height1:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.text(bar2.get_x() + bar2.get_width()/2, height2 + 1.5,
                f'{height2:.1f}%', ha='center', va='bottom', fontsize=8)
        
        if height2 > height1:
            gain = height2 - height1
            ax.annotate(f'+{gain:.1f}%', 
                       xy=(bar2.get_x() + bar2.get_width()/2, height2),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        elif height1 > height2:
            loss = height1 - height2
            ax.annotate(f'-{loss:.1f}%', 
                       xy=(bar1.get_x() + bar1.get_width()/2, height1),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS71_Calibration_Class_Distribution.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S71 saved.")


# ============================================================================
# VIS-17: Best Subject Summary Figure (STRONGEST FIGURE)
# ============================================================================

def _plot_best_subject_summary(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """
    VIS-17: Best Subject Summary Figure. Figure S72 + Table S68.
    
    Shows: Raw CM, Normalized CM, PCA Projection, Misclassified Samples.
    This is the strongest figure in the visualization block.
    """
    logger.info("  VIS-17: Best Subject Summary Figure...")
    
    table4_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    best_subject = None
    best_accuracy = 0.0
    
    if table4_path.exists():
        try:
            df_subject = load_csv(table4_path)
            for sub in df_subject["Subject"].unique():
                sub_data = df_subject[df_subject["Subject"] == sub]
                max_acc = sub_data["Accuracy"].max()
                if max_acc > best_accuracy:
                    best_accuracy = max_acc
                    best_subject = sub
            logger.info(f"  Best subject from BLOCK_3: {best_subject} (accuracy: {best_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load from BLOCK_3: {e}")
    
    if best_subject is None:
        logger.info("  Computing best subject from scratch...")
        loso = LeaveOneGroupOut()
        subject_accs = {}
        models = build_models()
        
        for tr_idx, te_idx in loso.split(X, y, subjects):
            test_sub = subjects[te_idx[0]]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
            X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="best_subject")
            
            for name, clf in models.items():
                try:
                    clf.fit(X_tr_final, y_tr_final)
                    preds = clf.predict(X_te_sel)
                    acc = accuracy_score(y_te, preds)
                    if test_sub not in subject_accs or acc > subject_accs[test_sub]:
                        subject_accs[test_sub] = acc
                except Exception:
                    pass
            gc.collect()
        
        best_subject = max(subject_accs, key=subject_accs.get)
        best_accuracy = subject_accs[best_subject]
        logger.info(f"  Best subject (computed): {best_subject} (accuracy: {best_accuracy:.4f})")
    
    pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
    y_true = None
    y_pred = None
    best_model = "RF"
    
    if pred_archive_path.exists():
        try:
            df_pred = pd.read_csv(pred_archive_path)
            df_sub = df_pred[df_pred["Subject"] == best_subject]
            if not df_sub.empty:
                model_accs = {}
                for model in MODEL_NAMES:
                    model_data = df_sub[df_sub["Model"] == model]
                    if not model_data.empty:
                        acc = accuracy_score(model_data["True_Label"], model_data["Pred_Label"])
                        model_accs[model] = acc
                best_model = max(model_accs, key=model_accs.get) if model_accs else "RF"
                df_best = df_sub[df_sub["Model"] == best_model]
                if not df_best.empty:
                    y_true = df_best["True_Label"].values
                    y_pred = df_best["Pred_Label"].values
                    best_accuracy = accuracy_score(y_true, y_pred)
                    logger.info(f"  Using {best_model} for subject {best_subject} (acc: {best_accuracy:.4f})")
        except Exception as e:
            logger.warning(f"  Failed to load prediction archive: {e}")
    
    if y_true is None:
        logger.info(f"  Computing predictions for subject {best_subject}...")
        loso = LeaveOneGroupOut()
        for tr_idx, te_idx in loso.split(X, y, subjects):
            if subjects[te_idx[0]] == best_subject:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_te, y_te = X[te_idx], y[te_idx]
                
                X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr, y_tr)
                X_te_sel = apply_feature_pipeline(X_te, sc, sv, sk)
                X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag="best_subject_summary")
                
                clf = _build_model("RF")
                clf.fit(X_tr_final, y_tr_final)
                y_pred = clf.predict(X_te_sel)
                y_true = y_te
                best_accuracy = accuracy_score(y_true, y_pred)
                break
    
    if y_true is None or len(y_true) == 0:
        logger.warning("  Could not generate best subject summary")
        return
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    misclassified = y_true != y_pred
    
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Raw Confusion Matrix
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                ax=ax1, cbar_kws={"label": "Count"})
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"Raw Confusion Matrix\nTotal: {len(y_true)} samples", fontsize=11, fontweight="bold")
    
    # Subplot 2: Normalized Confusion Matrix
    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                ax=ax2, cbar_kws={"label": "Proportion"})
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Normalized Confusion Matrix\n(Proportions by True Gesture)", fontsize=11, fontweight="bold")
    
    # Subplot 3: PCA Projection with Highlights
    ax3 = plt.subplot(2, 2, 3)
    colors = sns.color_palette("tab10", N_CLASSES)
    
    for gi, gesture in enumerate(GESTURE_SHORT):
        mask = y == gi
        if np.sum(mask) > 0:
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=10, alpha=0.15, color=colors[gi], label=gesture)
    
    sub_indices = np.where(subjects == best_subject)[0]
    if len(sub_indices) > 0:
        sub_pca = X_pca[sub_indices]
        sub_true = y[sub_indices]
        
        pred_archive_path = BLOCK_DIRS["BLOCK_3"] / "PredictionArchive_LOSO.csv"
        sub_y_pred = None
        if pred_archive_path.exists():
            try:
                df_pred = pd.read_csv(pred_archive_path)
                df_sub = df_pred[(df_pred["Subject"] == best_subject) & (df_pred["Model"] == best_model)]
                if not df_sub.empty:
                    sub_y_pred = df_sub["Pred_Label"].values
            except Exception:
                pass
        
        if sub_y_pred is not None and len(sub_y_pred) == len(sub_indices):
            sub_correct = sub_y_pred == sub_true
        else:
            sub_correct = ~misclassified if len(misclassified) == len(sub_indices) else np.ones(len(sub_indices), dtype=bool)
        
        if np.sum(sub_correct) > 0:
            ax3.scatter(sub_pca[sub_correct, 0], sub_pca[sub_correct, 1],
                       s=80, alpha=0.9, color='green', 
                       edgecolor='darkgreen', linewidth=1.5,
                       label=f'Correct ({np.sum(sub_correct)})', zorder=10)
        
        if np.sum(~sub_correct) > 0:
            ax3.scatter(sub_pca[~sub_correct, 0], sub_pca[~sub_correct, 1],
                       s=120, alpha=0.95, color='red',
                       edgecolor='darkred', linewidth=2, marker='X',
                       label=f'Misclassified ({np.sum(~sub_correct)})', zorder=11)
    
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax3.set_title(f"PCA Projection with {best_subject} Highlighted\n(Green=Correct, Red=Misclassified)",
                 fontsize=11, fontweight="bold")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.2)
    
    # Subplot 4: Gesture-wise Performance
    ax4 = plt.subplot(2, 2, 4)
    
    per_gesture_acc = []
    per_gesture_support = []
    for gi, gest in enumerate(GESTURE_SHORT):
        mask = y_true == gi
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            per_gesture_acc.append(acc)
            per_gesture_support.append(np.sum(mask))
        else:
            per_gesture_acc.append(0.0)
            per_gesture_support.append(0)
    
    sorted_idx = np.argsort(per_gesture_acc)[::-1]
    sorted_gestures = [GESTURE_SHORT[i] for i in sorted_idx]
    sorted_acc = [per_gesture_acc[i] * 100 for i in sorted_idx]
    sorted_support = [per_gesture_support[i] for i in sorted_idx]
    
    bars = ax4.barh(sorted_gestures, sorted_acc, 
                    color=sns.color_palette("RdYlGn", len(sorted_acc)),
                    edgecolor='black')
    
    ax4.set_xlabel("Accuracy (%)")
    ax4.set_title(f"Per-Gesture Accuracy for {best_subject}\n(Overall: {best_accuracy*100:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax4.set_xlim(0, 105)
    
    for bar, acc, support in zip(bars, sorted_acc, sorted_support):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}% (n={support})', va='center', fontsize=9)
    
    ax4.grid(True, alpha=0.2, axis='x')
    
    # Main title
    fig.suptitle(
        f"Figure S72 – Comprehensive Summary of Best Performing Subject ({best_subject})\n"
        f"Model: {best_model} | Accuracy: {best_accuracy*100:.1f}% | Samples: {len(y_true)}",
        fontsize=16, fontweight="bold"
    )
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS72_Best_Subject_Summary.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S72 saved.")
    
    # Table S68
    summary_data = {
        "Subject": best_subject,
        "Best_Model": best_model,
        "Accuracy": round(best_accuracy, 4),
        "N_Samples": len(y_true),
        "N_Correct": np.sum(~misclassified),
        "N_Misclassified": np.sum(misclassified),
    }
    for gi, gest in enumerate(GESTURE_SHORT):
        mask = y_true == gi
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            summary_data[f"{gest}_Accuracy"] = round(acc, 4)
            summary_data[f"{gest}_Support"] = np.sum(mask)
        else:
            summary_data[f"{gest}_Accuracy"] = 0.0
            summary_data[f"{gest}_Support"] = 0
    
    df_summary = pd.DataFrame([summary_data])
    save_csv(df_summary, OUT_DIR / "Table_S68_Best_Subject_Summary.csv")
    logger.info("  Table S68 saved.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_block11() -> Dict:
    """Main entry point for BLOCK 11."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 11 | VIS1: Feature Space Separability and Representation Analysis")
    logger.info("=" * 70)
    
    if ENABLE_UMAP:
        logger.info("  UMAP: ENABLED (ENABLE_UMAP=True)")
    else:
        logger.info("  UMAP: DISABLED (ENABLE_UMAP=False) - Safe mode")
    
    logger.info(f"Python : {sys.version}")
    logger.info(f"Numpy  : {np.__version__}")
    logger.info(f"Pandas : {pd.__version__}")
    logger.info(f"SciPy  : {scipy.__version__}")
    logger.info(f"Sklearn: {sklearn.__version__}")
    logger.info(f"XGBoost: {xgboost.__version__}")
    logger.info(f"Imblearn: {imblearn.__version__}")
    logger.info(f"PID    : {os.getpid()}")
    
    # ── Load data ──────────────────────────────────────────────────────────
    try:
        X, y, subjects, feature_names = load_feature_matrix()
        logger.info(f"  Loaded {len(X)} windows from BLOCK_2 artifacts")
    except (FileNotFoundError, KeyError, TypeError) as e:
        logger.warning(f"  BLOCK_2 artifacts not found: {e}")
        logger.info("  Falling back to raw data loading...")
        X, y, subjects = load_fallback_data()
        feature_names = FEATURE_NAMES
    
    # Verify subject count
    n_subjects_detected = len(np.unique(subjects))
    if n_subjects_detected != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
            f"but detected {n_subjects_detected} subjects in the data."
        )
    logger.info(f"  Verified {n_subjects_detected} subjects (matches EXPECTED_SUBJECTS)")
    
    # ── Load feature groups ──────────────────────────────────────────────
    try:
        feature_groups = load_json(BLOCK_DIRS["BLOCK_2"] / "feature_groups.json")
        logger.info(f"  Loaded {len(feature_groups)} feature groups")
    except Exception:
        feature_groups = {
            "Time_Domain": ["MAV", "RMS", "WL", "DASDV", "WAMP", "LOGVAR", "TKEO", "IEMG", "ZC"],
            "Hjorth": ["ACTIVITY", "MOBILITY", "COMPLEXITY"],
            "AR": ["AR1", "AR2", "AR3", "AR4"],
            "Wavelet": ["WAV1", "WAV2", "WAV3", "WAV4"],
            "Frequency": ["PSD_LOW", "PSD_MID", "PSD_HIGH", "MDF", "MNF", "SPEC_ENT"],
            "Entropy": ["SH_HILB", "SAMPEN"],
            "Statistics": ["SKEW", "KURT"],
        }
        logger.info("  Using default feature groups")
    
    # ── Generate all visualizations ──────────────────────────────────────
    
    # VIS-01: PCA Projection + Explained Variance
    _plot_pca_projection(X, y, OUT_DIR / "FigS2_PCA_Projection.png")
    _plot_pca_explained_variance(X, OUT_DIR / "FigS7_PCA_Explained_Variance.png")
    
    # VIS-02: t-SNE Projection
    _plot_tsne_projection(X, y, OUT_DIR / "FigS3_tSNE_Projection.png")
    
    # VIS-03: UMAP Projection
    _plot_umap_projection(X, y, OUT_DIR / "FigS4_UMAP_Projection.png")
    
    # VIS-04: LOSO Subject Heatmap
    _plot_loso_subject_heatmap(X, y, subjects, OUT_DIR / "FigS5_LOSO_Subject_Heatmap.png")
    
    # VIS-05: Feature Correlation Matrix
    _plot_feature_correlation_matrix(X, feature_names, OUT_DIR / "FigS6_Feature_Correlation_Matrix.png")
    
    # VIS-06: Gesture Distance and Separability
    dist_df = _compute_gesture_distances(X, y)
    _plot_class_distance_matrix(dist_df, OUT_DIR / "FigS13_Class_Distance_Matrix.png")
    
    # VIS-07: Subject Representation Variability
    _plot_subject_representation(X, subjects, OUT_DIR / "FigS60_Subject_Representation_Projection.png")
    
    # VIS-08: Calibration Representation Analysis
    _plot_calibration_representation(X, y, subjects)
    
    # VIS-09: Feature Group Representation Analysis
    _plot_feature_group_representation(X, y, feature_groups, OUT_DIR / "FigS64_Feature_Group_Representation.png")
    
    # Subject Similarity Matrix (VIS-04 extension)
    sim_df = _compute_subject_similarity(X, subjects)
    _plot_subject_similarity_matrix(sim_df, OUT_DIR / "FigS59_Subject_Similarity_Matrix.png")
    
    # VIS-10: Best Performing Subject Visualization
    best_stats, best_subject = _plot_best_subject_confusion(X, y, subjects)
    
    # VIS-11: Worst Performing Subject Visualization
    worst_stats, worst_subject = _plot_worst_subject_confusion(X, y, subjects)
    
    # VIS-12: Gesture Centroid Representation
    _plot_gesture_centroid_representation(X, y, OUT_DIR / "FigS67_Gesture_Centroid_Representation.png")
    
    # VIS-13: Correct vs Misclassified Sample Visualization (FIXED - Publication Grade)
    _plot_correct_vs_misclassified(X, y, subjects, OUT_DIR / "FigS68_Correct_vs_Misclassified_Samples.png")
    
    # VIS-14: Per-Gesture Accuracy Heatmap
    _plot_per_gesture_accuracy_heatmap(X, y, subjects)
    
    # VIS-15: Gesture Centroid Distance Network
    _plot_gesture_centroid_network(X, y)
    
    # VIS-16: Class Distribution Before and After Calibration
    _plot_calibration_class_distribution(X, y, subjects)
    
    # VIS-17: Best Subject Summary Figure (STRONGEST)
    _plot_best_subject_summary(X, y, subjects)
    
    # ── Save environment versions ──────────────────────────────────────────
    _save_environment_versions()
    
    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "fig_s2": OUT_DIR / "FigS2_PCA_Projection.png",
        "fig_s3": OUT_DIR / "FigS3_tSNE_Projection.png",
        "fig_s4": OUT_DIR / "FigS4_UMAP_Projection.png",
        "fig_s5": OUT_DIR / "FigS5_LOSO_Subject_Heatmap.png",
        "fig_s6": OUT_DIR / "FigS6_Feature_Correlation_Matrix.png",
        "fig_s7": OUT_DIR / "FigS7_PCA_Explained_Variance.png",
        "fig_s13": OUT_DIR / "FigS13_Class_Distance_Matrix.png",
        "fig_s59": OUT_DIR / "FigS59_Subject_Similarity_Matrix.png",
        "fig_s60": OUT_DIR / "FigS60_Subject_Representation_Projection.png",
        "fig_s61": OUT_DIR / "FigS61_Feature_Space_Before_Calibration.png",
        "fig_s62": OUT_DIR / "FigS62_Feature_Space_After_Calibration.png",
        "fig_s63": OUT_DIR / "FigS63_Calibration_Performance_Improvement.png",
        "fig_s64": OUT_DIR / "FigS64_Feature_Group_Representation.png",
        "fig_s65": OUT_DIR / "FigS65_Best_Subject_Confusion_Matrix.png",
        "fig_s66": OUT_DIR / "FigS66_Worst_Subject_Confusion_Matrix.png",
        "fig_s67": OUT_DIR / "FigS67_Gesture_Centroid_Representation.png",
        "fig_s68": OUT_DIR / "FigS68_Correct_vs_Misclassified_Samples.png",
        "fig_s69": OUT_DIR / "FigS69_PerGesture_Accuracy_Heatmap.png",
        "fig_s70": OUT_DIR / "FigS70_Gesture_Centroid_Network.png",
        "fig_s71": OUT_DIR / "FigS71_Calibration_Class_Distribution.png",
        "fig_s72": OUT_DIR / "FigS72_Best_Subject_Summary.png",
        "env_versions": OUT_DIR / "environment_versions.json",
        "table_s59": OUT_DIR / "Table_S59_Highly_Correlated_Feature_Pairs.csv",
        "table_s60": OUT_DIR / "Table_S60_Gesture_Distance_Statistics.csv",
        "table_s61": OUT_DIR / "Table_S61_Subject_Similarity_Statistics.csv",
        "table_s62": OUT_DIR / "Table_S62_Feature_Group_Separability_Statistics.csv",
        "table_s63": OUT_DIR / "Table_S63_Best_Subject_Statistics.csv",
        "table_s64": OUT_DIR / "Table_S64_Worst_Subject_Statistics.csv",
        "table_s65": OUT_DIR / "Table_S65_PerGesture_Accuracy_Summary.csv",
        "table_s66": OUT_DIR / "Table_S66_Gesture_Centroid_Distances.csv",
        "table_s67": OUT_DIR / "Table_S67_Calibration_Class_Improvement.csv",
        "table_s68": OUT_DIR / "Table_S68_Best_Subject_Summary.csv",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)
    
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK 11 runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    
    logger.info("=" * 70)
    logger.info("BLOCK 11 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Generated 22 figures (S2, S3, S4, S5, S6, S7, S13, S59-S72)")
    logger.info(f"  Generated 10 supplementary tables (S59-S68)")
    logger.info(f"  Best subject: {best_subject} (Acc: {best_stats.get('Accuracy', 0):.4f})")
    logger.info(f"  Worst subject: {worst_subject} (Acc: {worst_stats.get('Accuracy', 0):.4f})")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)
    
    return {
        "n_windows": len(X),
        "n_features": X.shape[1],
        "n_subjects": len(np.unique(subjects)),
        "umap_available": UMAP_AVAILABLE,
        "umap_enabled": ENABLE_UMAP,
        "best_subject": best_subject,
        "best_subject_accuracy": best_stats.get("Accuracy", 0),
        "worst_subject": worst_subject,
        "worst_subject_accuracy": worst_stats.get("Accuracy", 0),
        "runtime_sec": elapsed,
        "runtime_minutes": minutes,
        "runtime_seconds": seconds,
        "n_subjects_verified": n_subjects_detected,
        "pipeline": PIPELINE_NAME,
    }


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return
    
    try:
        result = run_block11()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()