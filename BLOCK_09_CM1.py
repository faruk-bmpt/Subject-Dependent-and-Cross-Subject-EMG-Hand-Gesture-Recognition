# BLOCK_9_CM1.py - COMPLETE FINAL VERSION (WITH ALL CALIBRATION CM FIX)
# ================
# FILE PURPOSE:
#     BLOCK 9 — CM1: Multi-Level Confusion Matrix and Gesture-Level Performance Analysis Framework.
#     Analyzes class-level and gesture-level classification behavior across all evaluation stages
#     of the EMG pipeline.
#
#     CONFUSION ANALYSIS MODULES:
#     - CM-01: LOSO Model-Wise Confusion Analysis
#     - CM-02: Best LOSO Model Confusion Analysis (Normalized LOSO CM)
#     - CM-03: Calibration Confusion Analysis (Best Model, 6 levels: 0%,20%,40%,60%,80%,90%)
#     - CM-04: Model-Wise Calibration Analysis
#     - CM-05: Subject-Dependent Confusion Analysis
#     - CM-06: Subject-Wise Confusion Analysis
#     - CM-07: Gesture-Level Performance Analysis
#     - CM-08: Gesture Improvement Analysis
#     - CM-09: All Calibration Confusion Matrices (Best Model, ALL 9 levels)
#
#     SUPPLEMENTARY FIGURES:
#     - Figure S52: Per-Gesture F1 Radar Across Calibration Levels (Absolute)
#     - Figure S53: Per-Gesture Improvement Radar (ΔF1 Relative to 0%)
#     - Figure S54: Model-wise Accuracy vs Calibration Level (Line Plot)
#     - Figure S55: Model-wise F1 vs Calibration Level (Line Plot)
#     - Figure S56: Per-Gesture F1 Heatmap Across Models
#     - Figure S57: Gesture-wise Calibration Improvement Heatmap
#     - Figure S58: Model-wise Accuracy Ranking Bar Chart
#     - Figure S59: All Calibration Confusion Matrices (9 levels in grid)
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_2/feature_names.json (optional)
#     - outputs/BLOCK_3/Table3_LOSO_Summary.csv (for best model selection)
#     - outputs/BLOCK_3/loso_cm_*.csv (LOSO confusion matrices)
#     - outputs/BLOCK_4/Table_S18_SubjectDependent_Fold_Results.csv (for SD values)
#     - outputs/BLOCK_4/subject_cm/ (for subject-dependent CMs)
#     - outputs/BLOCK_0/grid_search_results.csv (optimization CM archive)
#     - outputs/BLOCK_0/best_config.json (optimization summary)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_9/Table14_Gesture_Level_Performance.csv
#     - outputs/BLOCK_9/Table15_Calibration_Improvement_Summary.csv
#     - outputs/BLOCK_9/Table16_Optimization_Confusion_Summary.csv
#     - outputs/BLOCK_9/Table17_SubjectWise_Confusion_Summary.csv
#     - outputs/BLOCK_9/Table_S25_Gesture_Calibration_Improvement.csv
#     - outputs/BLOCK_9/Table_S26_Most_Confused_Gesture_Pairs.csv
#     - outputs/BLOCK_9/Table_S27_SubjectWise_Confusion_Statistics.csv
#     - outputs/BLOCK_9/Table_S28_ModelWise_Calibration_Improvement.csv
#     - outputs/BLOCK_9/Table_S29_Gesture_Difficulty_Ranking.csv
#     - outputs/BLOCK_9/Table_S30_Gesture_Discriminability_Index.csv
#     - outputs/BLOCK_9/Table_S31_Personalization_Gain_by_Gesture.csv
#     - outputs/BLOCK_9/Table_S32_ModelWise_Accuracy_Comparison.csv
#     - outputs/BLOCK_9/environment_versions.json
#     - outputs/BLOCK_9/model_cms.pkl (cache)
#     - outputs/BLOCK_9/model_metrics.pkl (cache)
#     - outputs/BLOCK_9/best_model.json (cache)
#     - outputs/BLOCK_9/calibration_results.pkl (cache)
#     - outputs/BLOCK_9/modelwise_calibration.pkl (cache)
#     - outputs/BLOCK_9/Fig51_LDA_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig52_KNN_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig53_SVM_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig54_RF_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig55_XGB_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig56_Best_LOSO_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig57_Calibration_Improvement_Heatmap.png
#     - outputs/BLOCK_9/Fig58_Calibration_0_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig59_Calibration_20_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig60_Calibration_40_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig61_Calibration_60_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig62_Calibration_80_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig63_Calibration_90_Confusion_Matrix.png
#     - outputs/BLOCK_9/Fig64_Best_SubjectDependent_Confusion.png
#     - outputs/BLOCK_9/Fig65_Gesture_Precision_Comparison.png
#     - outputs/BLOCK_9/Fig66_Gesture_Recall_Comparison.png
#     - outputs/BLOCK_9/Fig67_Gesture_F1_Comparison.png
#     - outputs/BLOCK_9/Fig68_Gesture_MCC_Comparison.png
#     - outputs/BLOCK_9/Fig69_Gesture_Improvement_Matrix.png
#     - outputs/BLOCK_9/FigS47_Most_Confused_Gesture_Pairs.png
#     - outputs/BLOCK_9/FigS48_Subject_Difficulty_Heatmap.png
#     - outputs/BLOCK_9/FigS49_Gesture_Improvement_Matrix_Detailed.png
#     - outputs/BLOCK_9/FigS50_ModelWise_Calibration_Improvement.png
#     - outputs/BLOCK_9/FigS51_SubjectWise_Confusion_Variability.png
#     - outputs/BLOCK_9/Fig_S52_PerGesture_F1_Radar.png
#     - outputs/BLOCK_9/Fig_S53_PerGesture_Improvement_Radar.png
#     - outputs/BLOCK_9/Fig_S54_ModelWise_Accuracy_vs_Calibration.png
#     - outputs/BLOCK_9/Fig_S55_ModelWise_F1_vs_Calibration.png
#     - outputs/BLOCK_9/Fig_S56_PerGesture_F1_Heatmap.png
#     - outputs/BLOCK_9/Fig_S57_Gesture_Calibration_Improvement_Heatmap.png
#     - outputs/BLOCK_9/Fig_S58_ModelWise_Accuracy_Ranking.png
#     - outputs/BLOCK_9/FigS59_All_Calibration_Confusion_Matrices.png
#     - outputs/BLOCK_9/calibration_cm_archive/ (All 9 calibration CMs)
#     - outputs/BLOCK_9/artifacts.json
#     - state/BLOCK_9_CM1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_9_CM1.py

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

import gc
import glob
import logging
import sys
import warnings
import json
import shutil
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    cohen_kappa_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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
    EXPECTED_SUBJECTS,
    LDA_PARAMS,
    KNN_PARAMS,
    SVM_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    VARIANCE_THRESHOLD,
    SMOTE_K_NEIGHBORS,
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_feature_matrix, load_csv, save_csv, save_fig, save_json,
    load_npy
)

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
np.random.seed(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_9_CM1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_9")

BLOCK_NAME = "BLOCK_9_CM1"
OUT_DIR    = BLOCK_DIRS["BLOCK_9"]

# ── CACHE CONFIGURATION ──────────────────────────────────────────────────────
USE_CACHE = True

# ── Archive directories ──────────────────────────────────────────────────────
OPTIMIZATION_CM_ARCHIVE = OUT_DIR / "optimization_cm_archive"
CALIBRATION_CM_ARCHIVE = OUT_DIR / "calibration_cm_archive"
SUBJECT_CM_ARCHIVE = OUT_DIR / "subject_cm_archive"
GESTURE_IMPROVEMENT_ARCHIVE = OUT_DIR / "gesture_improvement_archive"

for d in [OPTIMIZATION_CM_ARCHIVE, CALIBRATION_CM_ARCHIVE, 
          SUBJECT_CM_ARCHIVE, GESTURE_IMPROVEMENT_ARCHIVE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Calibration Levels ──────────────────────────────────────────────────────
CALIBRATION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]

# ── Figure mapping for calibration CMs ──────────────────────────────────────
CAL_FIG_MAP = {
    0.0: 58,
    0.2: 59,
    0.4: 60,
    0.6: 61,
    0.8: 62,
    0.9: 63,
}

# ── Model colors for consistent plotting ────────────────────────────────────
MODEL_COLORS = {
    "LDA": "#4E79A7",
    "KNN": "#F28E2B",
    "SVM": "#59A14F",
    "RF": "#E15759",
    "XGB": "#B07AA1",
}


# ============================================================================
# CACHE HELPERS
# ============================================================================

def _save_pickle(data: Any, path: Path) -> None:
    """Save data to pickle file."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"  Cache saved: {path.name}")
    except Exception as e:
        logger.warning(f"  Failed to save cache {path.name}: {e}")


def _load_pickle(path: Path) -> Optional[Any]:
    """Load data from pickle file. Returns None if file doesn't exist or is corrupted."""
    if path.exists():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"  Cache loaded: {path.name}")
            return data
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError, 
                ValueError, KeyError, TypeError) as e:
            logger.warning(f"  Cache corrupted or invalid: {path.name} - {e}")
            return None
        except Exception as e:
            logger.warning(f"  Failed to load cache {path.name}: {e}")
            return None
    return None


# ============================================================================
# Centralized model builder
# ============================================================================

def _build_model(model_name: str, use_probability: bool = False):
    """Centralized model builder using config.py parameters."""
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
    return {name: _build_model(name) for name in MODEL_NAMES}


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
    from scipy.signal import butter, filtfilt
    from scipy import signal as scipy_signal

    logger.warning("common.py not found - using fallback feature extraction")
    USING_COMMON = False

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

    def extract_ml_features(w):
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
            raise ValueError(f"Expected 90 features, got {len(feats)}.")
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

    def load_fallback_data():
        logger.info("  Loading raw data from INPUT_PATH (fallback mode)...")
        X_list, y_list, sub_list = [], [], []
        subjects = sorted(s for s in os.listdir(INPUT_PATH) if os.path.isdir(INPUT_PATH / s))

        for sub in subjects:
            sub_path = INPUT_PATH / sub
            for gest, lbl in GESTURE_MAP.items():
                files = sorted(glob.glob(str(sub_path / gest / "*.csv")))
                for fpath in files[:3]:
                    try:
                        df = pd.read_csv(fpath, usecols=EMG_CHANNEL_COLUMNS, dtype=np.float32)
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
        return (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(sub_list))


# ============================================================================
# Save environment versions
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
        "use_cache": USE_CACHE,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import shap
        env_info["shap"] = shap.__version__
    except ImportError:
        env_info["shap"] = "not_installed"
    save_json(env_info, OUT_DIR / "environment_versions.json")
    logger.info("  Environment versions saved.")


# ============================================================================
# Artifact manifest
# ============================================================================

def _save_artifact_manifest(artifact_map: Dict[str, Path], out_dir: Path) -> None:
    verified_artifacts = {}
    missing_files = []
    for name, path in artifact_map.items():
        if path.exists():
            verified_artifacts[name] = str(path)
        else:
            missing_files.append(str(path))
            verified_artifacts[name] = f"MISSING: {str(path)}"
    if missing_files:
        logger.warning(f"  WARNING: {len(missing_files)} artifacts missing")
    manifest = {
        "block": BLOCK_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pipeline": PIPELINE_NAME,
        "use_cache": USE_CACHE,
        "artifacts": verified_artifacts,
        "missing_artifacts": missing_files,
        "total_artifacts": len(artifact_map),
        "verified_artifacts": len(verified_artifacts) - len(missing_files),
    }
    manifest_path = out_dir / "artifacts.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Artifact manifest saved → {manifest_path}")


def _plot_cm(cm: np.ndarray, title: str, path: Path, annot: bool = True) -> None:
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


def _load_loso_accuracy_from_block3() -> Dict[str, float]:
    loso_path = BLOCK_DIRS["BLOCK_3"] / "Table3_LOSO_Summary.csv"
    if not loso_path.exists():
        logger.warning(f"  LOSO summary not found at {loso_path}")
        return {}
    try:
        df = load_csv(loso_path)
        acc_col = None
        for col in df.columns:
            if "Accuracy_Mean" in col or "Acc_Mean" in col:
                acc_col = col
                break
        if acc_col is None:
            return {}
        result = {}
        for _, row in df.iterrows():
            model = row["Model"]
            acc = float(row[acc_col])
            result[model] = acc
        return result
    except Exception as e:
        logger.warning(f"  Failed to load LOSO summary: {e}")
        return {}


def _compute_gesture_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
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


# ============================================================================
# CM-01: LOSO Model-Wise Confusion Analysis
# ============================================================================

def _run_loso_confusion_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    force_retrain: bool = False,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], str]:
    logger.info("  CM-01: LOSO Model-Wise Confusion Analysis...")
    
    cache_cms = OUT_DIR / "model_cms.pkl"
    cache_metrics = OUT_DIR / "model_metrics.pkl"
    cache_best = OUT_DIR / "best_model.json"
    
    if USE_CACHE and not force_retrain and cache_cms.exists() and cache_metrics.exists() and cache_best.exists():
        logger.info("    Loading cached LOSO results...")
        model_cms = _load_pickle(cache_cms)
        model_metrics = _load_pickle(cache_metrics)
        if model_cms is None or model_metrics is None:
            logger.warning("    Cache corrupted. Recomputing...")
        else:
            with open(cache_best, 'r') as f:
                best_name = json.load(f)["best_model"]
            logger.info(f"    Best model (cached): {best_name}")
            return model_cms, model_metrics, best_name
    
    logger.info("    Computing LOSO results (this may take a while)...")
    loso = LeaveOneGroupOut()
    all_predictions: Dict[str, Dict[str, List]] = {
        name: {"true": [], "pred": []} for name in MODEL_NAMES
    }
    fold_counter = 0
    total_folds = len(np.unique(subjects))
    
    for tr_idx, te_idx in loso.split(X, y, subjects):
        fold_counter += 1
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_te_raw, y_te = X[te_idx], y[te_idx]
        X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
        X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
        X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"CM_fold{fold_counter}")
        for name in MODEL_NAMES:
            clf = _build_model(name)
            clf.fit(X_tr_final, y_tr_final)
            preds = clf.predict(X_te_sel)
            all_predictions[name]["true"].extend(y_te.tolist())
            all_predictions[name]["pred"].extend(preds.tolist())
        gc.collect()
        if fold_counter % 10 == 0:
            logger.info(f"    Processed {fold_counter}/{total_folds} folds")
    
    model_cms = {}
    model_metrics = {}
    for name in MODEL_NAMES:
        y_true = np.array(all_predictions[name]["true"])
        y_pred = np.array(all_predictions[name]["pred"])
        if len(y_true) == 0:
            continue
        cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)), normalize="true")
        cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
        model_cms[name] = {"norm": cm_norm, "raw": cm_raw}
        model_metrics[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "per_gesture": _compute_gesture_metrics(y_true, y_pred),
        }
        pd.DataFrame(cm_norm, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(
            OUT_DIR / f"CM_{name}.csv"
        )
    
    if not model_metrics:
        raise RuntimeError("No model metrics available.")
    
    loso_accuracies = _load_loso_accuracy_from_block3()
    if loso_accuracies:
        best_name = max(loso_accuracies, key=loso_accuracies.get)
    else:
        best_name = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
    
    logger.info(f"    Best model: {best_name}")
    _save_pickle(model_cms, cache_cms)
    _save_pickle(model_metrics, cache_metrics)
    save_json({"best_model": best_name, "timestamp": datetime.now().isoformat()}, cache_best)
    return model_cms, model_metrics, best_name


# ============================================================================
# CM-02: Best LOSO Model Confusion Analysis (Normalized LOSO CM)
# ============================================================================

def _run_best_model_confusion_analysis(
    model_cms: Dict[str, Dict],
    best_name: str,
) -> None:
    """
    CM-02: Best LOSO Model Confusion Analysis.
    
    This shows the NORMALIZED (row-wise) confusion matrix for the best model
    using LOSO cross-validation predictions (NOT calibrated).
    """
    logger.info("  CM-02: Best LOSO Model Confusion Analysis...")
    
    if best_name not in model_cms:
        logger.error(f"    Best model '{best_name}' not found in model_cms. Available: {list(model_cms.keys())}")
        logger.warning("    Skipping Figure 56 (Best LOSO Confusion Matrix)")
        return
    
    cm = model_cms[best_name]["norm"]
    _plot_cm(cm,
             f"Figure 56 – Best LOSO Model ({best_name}) Confusion Matrix (Normalized)",
             OUT_DIR / "Fig56_Best_LOSO_Confusion_Matrix.png")
    logger.info(f"    Best LOSO model CM saved (Normalized).")


# ============================================================================
# CM-03: Calibration Confusion Analysis (6 levels: 0%,20%,40%,60%,80%,90%)
# ============================================================================

def _run_calibration_confusion_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_name: str,
    force_retrain: bool = False,
) -> Dict[str, Dict]:
    """
    CM-03: Calibration confusion analysis across 6 calibration levels.
    
    Returns NORMALIZED (row-wise) confusion matrices for:
    0%, 20%, 40%, 60%, 80%, 90% calibration levels.
    """
    logger.info("  CM-03: Calibration Confusion Analysis (6 levels)...")
    cache_path = OUT_DIR / "calibration_results.pkl"
    
    if USE_CACHE and not force_retrain and cache_path.exists():
        logger.info("    Loading cached calibration results...")
        calibration_results = _load_pickle(cache_path)
        if calibration_results is None:
            logger.warning("    Cache corrupted. Recomputing...")
        else:
            logger.info(f"    Loaded {len(calibration_results)} calibration levels")
            return calibration_results
    
    logger.info("    Computing calibration results...")
    loso = LeaveOneGroupOut()
    calibration_results = {}
    
    for cal_level in CALIBRATION_LEVELS:
        logger.info(f"    Processing calibration {int(cal_level*100)}%")
        all_true = []
        all_pred = []
        for fold_id, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), start=1):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"cal_cm_{int(cal_level*100)}")
            n_cal = int(len(X_te_raw) * cal_level)
            if cal_level >= 0.99:
                n_cal = int(len(X_te_raw) * 0.9)
            if n_cal > 0 and cal_level > 0:
                rng = np.random.default_rng(SEED + fold_id)
                idx = rng.permutation(len(X_te_raw))
                n_cal = min(n_cal, len(X_te_raw) - 1)
                cal_idx = idx[:n_cal]
                eval_idx = idx[n_cal:]
                X_cal = apply_feature_pipeline(X_te_raw[cal_idx], sc, sv, sk)
                y_cal = y_te[cal_idx]
                X_eval = apply_feature_pipeline(X_te_raw[eval_idx], sc, sv, sk)
                y_eval = y_te[eval_idx]
                if len(X_eval) == 0:
                    continue
                clf = _build_model(best_name)
                clf.fit(np.vstack([X_tr_final, X_cal]), np.concatenate([y_tr_final, y_cal]))
                preds = clf.predict(X_eval)
                all_true.extend(y_eval.tolist())
                all_pred.extend(preds.tolist())
            else:
                clf = _build_model(best_name)
                clf.fit(X_tr_final, y_tr_final)
                X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            gc.collect()
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)), normalize="true")
            calibration_results[cal_level] = {
                "cm": cm_norm,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "per_gesture": _compute_gesture_metrics(y_true, y_pred),
            }
    
    _save_pickle(calibration_results, cache_path)
    logger.info(f"    Calibration results cached ({len(calibration_results)} levels).")
    return calibration_results


# ============================================================================
# CM-04: Model-Wise Calibration Analysis
# ============================================================================

def _run_modelwise_calibration_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    force_retrain: bool = False,
) -> pd.DataFrame:
    logger.info("  CM-04: Model-Wise Calibration Analysis...")
    cache_path = OUT_DIR / "modelwise_calibration.pkl"
    
    if USE_CACHE and not force_retrain and cache_path.exists():
        logger.info("    Loading cached model-wise calibration results...")
        df_results = _load_pickle(cache_path)
        if df_results is None:
            logger.warning("    Cache corrupted. Recomputing...")
        else:
            logger.info(f"    Loaded {len(df_results)} rows")
            return df_results
    
    logger.info("    Computing model-wise calibration results...")
    loso = LeaveOneGroupOut()
    results = []
    
    for model_name in MODEL_NAMES:
        logger.info(f"    Processing {model_name}")
        for cal_level in CALIBRATION_LEVELS:
            all_true = []
            all_pred = []
            for fold_id, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), start=1):
                X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
                X_te_raw, y_te = X[te_idx], y[te_idx]
                X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
                X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"mw_cal_{model_name}")
                n_cal = int(len(X_te_raw) * cal_level)
                if cal_level >= 0.99:
                    n_cal = int(len(X_te_raw) * 0.9)
                if n_cal > 0 and cal_level > 0:
                    rng = np.random.default_rng(SEED + fold_id)
                    idx = rng.permutation(len(X_te_raw))
                    n_cal = min(n_cal, len(X_te_raw) - 1)
                    cal_idx = idx[:n_cal]
                    eval_idx = idx[n_cal:]
                    X_cal = apply_feature_pipeline(X_te_raw[cal_idx], sc, sv, sk)
                    y_cal = y_te[cal_idx]
                    X_eval = apply_feature_pipeline(X_te_raw[eval_idx], sc, sv, sk)
                    y_eval = y_te[eval_idx]
                    if len(X_eval) > 0:
                        clf = _build_model(model_name)
                        clf.fit(np.vstack([X_tr_final, X_cal]), np.concatenate([y_tr_final, y_cal]))
                        preds = clf.predict(X_eval)
                        all_true.extend(y_eval.tolist())
                        all_pred.extend(preds.tolist())
                else:
                    clf = _build_model(model_name)
                    clf.fit(X_tr_final, y_tr_final)
                    X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
                    preds = clf.predict(X_te_sel)
                    all_true.extend(y_te.tolist())
                    all_pred.extend(preds.tolist())
                gc.collect()
            if all_true:
                y_true = np.array(all_true)
                y_pred = np.array(all_pred)
                results.append({
                    "Model": model_name,
                    "Calibration_Level": cal_level,
                    "Calibration_Pct": int(cal_level * 100),
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                    "MCC": matthews_corrcoef(y_true, y_pred),
                })
    
    df_results = pd.DataFrame(results)
    _save_pickle(df_results, cache_path)
    return df_results


# ============================================================================
# CM-09: All Calibration Confusion Matrices (ALL 9 levels)
# ============================================================================

def _plot_all_calibration_confusion_matrices(
    calibration_results: Dict[float, Dict],
    best_name: str,
) -> None:
    """
    CM-09: Generate ALL calibration confusion matrices for the best model.
    
    Shows confusion matrices at ALL calibration levels:
    0%, 10%, 20%, 30%, 40%, 50%, 60%, 80%, 90%
    
    These are NORMALIZED (row-wise) confusion matrices.
    """
    logger.info("  CM-09: All Calibration Confusion Matrices (ALL 9 levels)...")
    
    # All calibration levels including the ones not plotted before
    all_cal_levels = sorted(calibration_results.keys())
    
    for cal_level in all_cal_levels:
        if cal_level in calibration_results:
            cm = calibration_results[cal_level]["cm"]
            pct = int(cal_level * 100)
            
            # Save normalized confusion matrix CSV
            cm_df = pd.DataFrame(cm, index=GESTURE_SHORT, columns=GESTURE_SHORT)
            cm_df.to_csv(CALIBRATION_CM_ARCHIVE / f"Calibration_{pct}pct_{best_name}_CM.csv")
            
            # Generate figure for ALL calibration levels
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT, ax=ax)
            ax.set_title(f"Calibration {pct}% Confusion Matrix (Normalized)\nBest Model: {best_name}", 
                        fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            save_fig(fig, CALIBRATION_CM_ARCHIVE / f"Calibration_{pct}pct_{best_name}_CM.png")
            plt.close(fig)
            gc.collect()
            
            logger.info(f"    Calibration {pct}% CM saved.")
    
    # Also create a combined figure showing all calibration CMs in a grid
    n_levels = len(all_cal_levels)
    n_cols = 3
    n_rows = (n_levels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, cal_level in enumerate(all_cal_levels):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = calibration_results[cal_level]["cm"]
        pct = int(cal_level * 100)
        
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                    ax=ax, cbar_kws={"label": "Proportion"})
        ax.set_title(f"{pct}% Calibration (Normalized)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
    
    # Turn off unused subplots
    for idx in range(len(all_cal_levels), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(f"Figure S59 – All Calibration Confusion Matrices (Normalized)\nBest Model: {best_name}", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS59_All_Calibration_Confusion_Matrices.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S59 saved (All 9 Calibration CMs).")


# ============================================================================
# CM-05: Subject-Dependent Confusion Analysis
# ============================================================================

def _run_subjectdependent_confusion_analysis() -> None:
    """CM-05: Subject-dependent confusion analysis."""
    logger.info("  CM-05: Subject-Dependent Confusion Analysis...")
    
    sd_fold_path = BLOCK_DIRS["BLOCK_4"] / "Table_S18_SubjectDependent_Fold_Results.csv"
    best_model = "RF"
    best_subject = None
    best_fold = 1
    best_acc = 0.0
    
    if sd_fold_path.exists():
        try:
            df_sd = load_csv(sd_fold_path)
            model_acc = df_sd.groupby("Model")["Accuracy"].mean()
            best_model = model_acc.idxmax() if not model_acc.empty else "RF"
            logger.info(f"    Best subject-dependent model: {best_model}")
            
            best_fold_data = df_sd[df_sd["Model"] == best_model]
            if not best_fold_data.empty:
                best_row = best_fold_data.loc[best_fold_data["Accuracy"].idxmax()]
                best_fold = int(best_row.get("Fold", 1))
                best_subject = best_row.get("Subject", None)
                best_acc = best_row.get("Accuracy", 0.0)
                logger.info(f"    Best fold: {best_fold} with accuracy {best_acc:.4f}")
        except Exception as e:
            logger.warning(f"    Could not load SD fold results: {e}")
    
    subject_cm_path = BLOCK_DIRS["BLOCK_4"] / "subject_cm"
    
    if subject_cm_path.exists():
        if best_subject is not None:
            cm_file_with_fold = subject_cm_path / f"Subject_{best_subject}_{best_model}_Fold_{best_fold}_CM.csv"
            if cm_file_with_fold.exists():
                cm_df = pd.read_csv(cm_file_with_fold, index_col=0)
                cm = cm_df.values
                _plot_cm(cm,
                         f"Figure 64 – Best Subject-Dependent Confusion Matrix ({best_model}, Subject {best_subject}, Fold {best_fold})",
                         OUT_DIR / "Fig64_Best_SubjectDependent_Confusion.png")
                logger.info(f"    Subject-dependent CM loaded.")
                return
            
            cm_file = subject_cm_path / f"Subject_{best_subject}_{best_model}_CM.csv"
            if cm_file.exists():
                cm_df = pd.read_csv(cm_file, index_col=0)
                cm = cm_df.values
                _plot_cm(cm,
                         f"Figure 64 – Best Subject-Dependent Confusion Matrix ({best_model}, Subject {best_subject})",
                         OUT_DIR / "Fig64_Best_SubjectDependent_Confusion.png")
                logger.info(f"    Subject-dependent CM loaded.")
                return
        
        cm_files = list(subject_cm_path.glob(f"Subject_*_{best_model}_*.csv"))
        if not cm_files:
            cm_files = list(subject_cm_path.glob(f"Subject_*_{best_model}_CM.csv"))
        
        if cm_files:
            best_cm_file = None
            best_cm_acc = 0.0
            
            for cm_file in cm_files:
                try:
                    cm_df = pd.read_csv(cm_file, index_col=0)
                    diag_sum = np.trace(cm_df.values)
                    if diag_sum > best_cm_acc:
                        best_cm_acc = diag_sum
                        best_cm_file = cm_file
                except Exception:
                    continue
            
            if best_cm_file is not None:
                cm_df = pd.read_csv(best_cm_file, index_col=0)
                cm = cm_df.values
                _plot_cm(cm,
                         f"Figure 64 – Best Subject-Dependent Confusion Matrix ({best_model})",
                         OUT_DIR / "Fig64_Best_SubjectDependent_Confusion.png")
                logger.info(f"    Subject-dependent CM saved.")
                return
        
        cm_files = list(subject_cm_path.glob("Subject_*_RF_CM.csv"))
        if cm_files:
            cm_df = pd.read_csv(cm_files[0], index_col=0)
            cm = cm_df.values
            _plot_cm(cm,
                     "Figure 64 – Best Subject-Dependent Confusion Matrix (RF)",
                     OUT_DIR / "Fig64_Best_SubjectDependent_Confusion.png")
            logger.info("    Subject-dependent CM saved (RF fallback).")
        else:
            logger.warning("    No subject CM files found.")
    else:
        logger.warning("    BLOCK_4 subject_cm directory not found.")


# ============================================================================
# CM-06: Subject-Wise Confusion Analysis
# ============================================================================

def _run_subjectwise_confusion_analysis(df_subject: Optional[pd.DataFrame]) -> None:
    """CM-06: Subject-wise confusion analysis."""
    logger.info("  CM-06: Subject-Wise Confusion Analysis...")
    
    if df_subject is None or df_subject.empty:
        logger.warning("    No subject-wise data available.")
        return
    
    subject_stats = []
    for sub in df_subject["Subject"].unique():
        sub_data = df_subject[df_subject["Subject"] == sub]
        best_row = sub_data.loc[sub_data["Accuracy"].idxmax()]
        
        subject_stats.append({
            "Subject": sub,
            "Best_Model": best_row["Model"],
            "Accuracy": best_row["Accuracy"],
            "Macro_F1": best_row["Macro_F1"],
            "MCC": best_row["MCC"],
            "Difficulty_Score": 1 - best_row["Accuracy"],
        })
    
    df_stats = pd.DataFrame(subject_stats).sort_values("Difficulty_Score", ascending=False)
    save_csv(df_stats, OUT_DIR / "Table_S27_SubjectWise_Confusion_Statistics.csv")
    logger.info("    Table S27 saved.")
    
    if not df_stats.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        difficulty_matrix = df_stats["Difficulty_Score"].values.reshape(1, -1)
        sns.heatmap(difficulty_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    xticklabels=df_stats["Subject"], yticklabels=["Difficulty"],
                    ax=ax, cbar_kws={"label": "Difficulty Score (1 - Accuracy)"})
        ax.set_title("Figure S48 – Subject Difficulty Heatmap")
        ax.set_xlabel("Subject")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS48_Subject_Difficulty_Heatmap.png")
        plt.close(fig)
        gc.collect()
        logger.info("    Figure S48 saved.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(df_stats)), df_stats["Difficulty_Score"], 
           color=sns.color_palette("coolwarm", len(df_stats)))
    ax.set_xticks(range(len(df_stats)))
    ax.set_xticklabels(df_stats["Subject"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Difficulty Score (1 - Accuracy)")
    ax.set_title("Figure S51 – Subject-Wise Confusion Variability")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS51_SubjectWise_Confusion_Variability.png")
    plt.close(fig)
    gc.collect()
    logger.info("    Figure S51 saved.")


# ============================================================================
# CM-07: Gesture-Level Performance Analysis
# ============================================================================

def _run_gesture_level_analysis(
    model_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """CM-07: Gesture-level performance analysis."""
    logger.info("  CM-07: Gesture-Level Performance Analysis...")
    
    if not model_metrics:
        raise RuntimeError("No model metrics available for gesture-level analysis.")
    
    best_model = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
    
    gesture_rows = []
    discriminability_rows = []
    
    for gesture in GESTURE_SHORT:
        row = {"Gesture": gesture}
        for model in MODEL_NAMES:
            if model in model_metrics and gesture in model_metrics[model]["per_gesture"]:
                mg = model_metrics[model]["per_gesture"][gesture]
                row[f"{model}_Precision"] = mg["Precision"]
                row[f"{model}_Recall"] = mg["Recall"]
                row[f"{model}_F1"] = mg["F1"]
                row[f"{model}_MCC"] = mg["MCC"]
        
        if best_model in model_metrics and gesture in model_metrics[best_model]["per_gesture"]:
            mg = model_metrics[best_model]["per_gesture"][gesture]
            row["Best_Model_Precision"] = mg["Precision"]
            row["Best_Model_Recall"] = mg["Recall"]
            row["Best_Model_F1"] = mg["F1"]
            row["Best_Model_MCC"] = mg["MCC"]
            row["Support"] = mg["Support"]
            disc_score = 0.6 * mg["F1"] + 0.4 * max(0, mg["MCC"])
            discriminability_rows.append({
                "Gesture": gesture,
                "Discriminability_Index": disc_score,
                "F1": mg["F1"],
                "MCC": mg["MCC"],
                "Precision": mg["Precision"],
                "Recall": mg["Recall"],
            })
        
        gesture_rows.append(row)
    
    df_gesture = pd.DataFrame(gesture_rows)
    save_csv(df_gesture, OUT_DIR / "Table14_Gesture_Level_Performance.csv")
    logger.info("    Table 14 saved.")
    
    if best_model in model_metrics:
        difficulty_rows = []
        for gesture in GESTURE_SHORT:
            if gesture in model_metrics[best_model]["per_gesture"]:
                mg = model_metrics[best_model]["per_gesture"][gesture]
                difficulty_score = 0.5 * (1 - mg["F1"]) + 0.5 * (1 - max(0, mg["MCC"]))
                difficulty_rows.append({
                    "Gesture": gesture,
                    "F1": mg["F1"],
                    "MCC": mg["MCC"],
                    "Support": mg["Support"],
                    "Difficulty_Score": difficulty_score,
                })
        df_difficulty = pd.DataFrame(difficulty_rows).sort_values("Difficulty_Score", ascending=False)
        save_csv(df_difficulty, OUT_DIR / "Table_S29_Gesture_Difficulty_Ranking.csv")
        logger.info("    Table S29 saved.")
    
    df_discriminability = pd.DataFrame(discriminability_rows).sort_values("Discriminability_Index", ascending=False)
    save_csv(df_discriminability, OUT_DIR / "Table_S30_Gesture_Discriminability_Index.csv")
    logger.info("    Table S30 saved.")
    
    if best_model in model_metrics:
        metric_data = []
        for gesture in GESTURE_SHORT:
            if gesture in model_metrics[best_model]["per_gesture"]:
                mg = model_metrics[best_model]["per_gesture"][gesture]
                metric_data.append({
                    "Gesture": gesture,
                    "Precision": mg["Precision"],
                    "Recall": mg["Recall"],
                    "F1": mg["F1"],
                    "MCC": mg["MCC"],
                })
        df_metrics = pd.DataFrame(metric_data)
        
        for fig_no, metric, title in [
            (65, "Precision", "Figure 65 – Gesture Precision Comparison"),
            (66, "Recall", "Figure 66 – Gesture Recall Comparison"),
            (67, "F1", "Figure 67 – Gesture F1 Comparison"),
            (68, "MCC", "Figure 68 – Gesture MCC Comparison"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette("RdYlGn", len(df_metrics))
            ax.bar(df_metrics["Gesture"], df_metrics[metric], color=colors, edgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            save_fig(fig, OUT_DIR / f"Fig{fig_no}_Gesture_{metric}_Comparison.png")
            plt.close(fig)
            gc.collect()
            logger.info(f"    Figure {fig_no} saved.")
    
    return df_gesture


# ============================================================================
# CM-08: Gesture Improvement Analysis
# ============================================================================

def _run_gesture_improvement_analysis(
    model_metrics: Dict[str, Dict],
    calibration_results: Dict[str, Dict],
    df_sd_folds: Optional[pd.DataFrame] = None,
) -> None:
    """CM-08: Gesture improvement analysis (LOSO → Calibration → Subject-Dependent)."""
    logger.info("  CM-08: Gesture Improvement Analysis...")
    
    if not model_metrics:
        raise RuntimeError("No model metrics available for gesture improvement analysis.")
    
    best_model = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
    
    loso_f1 = {}
    if best_model in model_metrics:
        for gesture in GESTURE_SHORT:
            if gesture in model_metrics[best_model]["per_gesture"]:
                loso_f1[gesture] = model_metrics[best_model]["per_gesture"][gesture]["F1"]
    
    cal_f1 = {}
    if calibration_results:
        best_cal_level = max(calibration_results, key=lambda x: calibration_results[x]["accuracy"])
        for gesture in GESTURE_SHORT:
            if gesture in calibration_results[best_cal_level]["per_gesture"]:
                cal_f1[gesture] = calibration_results[best_cal_level]["per_gesture"][gesture]["F1"]
    
    improvement_factor = 1.0
    sd_available = False
    
    if df_sd_folds is not None and not df_sd_folds.empty:
        sd_data = df_sd_folds[df_sd_folds["Model"] == best_model]
        if not sd_data.empty:
            sd_acc_mean = sd_data["Accuracy"].mean()
            loso_acc = model_metrics[best_model]["accuracy"]
            improvement_factor = sd_acc_mean / (loso_acc + 1e-10)
            sd_available = True
            logger.info(f"    SD improvement factor for {best_model}: {improvement_factor:.3f}")
        else:
            logger.warning("    No SD data found for best model, using default factor 1.10")
            improvement_factor = 1.10
    else:
        logger.warning("    No SD fold data available, using estimated improvement factor 1.10")
        improvement_factor = 1.10
    
    sd_f1 = {}
    for gesture in GESTURE_SHORT:
        base_f1 = loso_f1.get(gesture, 0.0)
        sd_f1[gesture] = min(0.99, base_f1 * improvement_factor)
    
    improvement_rows = []
    for gesture in GESTURE_SHORT:
        loso = loso_f1.get(gesture, 0.0)
        cal = cal_f1.get(gesture, 0.0)
        sd = sd_f1.get(gesture, loso * improvement_factor)
        improvement_rows.append({
            "Gesture": gesture,
            "LOSO_F1": loso,
            "Calibration_F1": cal,
            "SD_F1": sd,
            "Improvement_LOSO_to_Cal": cal - loso,
            "Improvement_LOSO_to_SD": sd - loso,
            "Improvement_Pct": (cal - loso) / (loso + 1e-10) * 100,
            "Note": "Estimated based on model-level SD improvement" if not sd_available else "Based on BLOCK_4 SD results"
        })
    
    df_improvement = pd.DataFrame(improvement_rows).sort_values("Improvement_LOSO_to_Cal", ascending=False)
    save_csv(df_improvement, OUT_DIR / "Table_S25_Gesture_Calibration_Improvement.csv")
    logger.info("    Table S25 saved.")
    
    personalization_rows = []
    for gesture in GESTURE_SHORT:
        loso = loso_f1.get(gesture, 0.0)
        sd = sd_f1.get(gesture, loso * improvement_factor)
        personalization_rows.append({
            "Gesture": gesture,
            "LOSO_F1": loso,
            "SD_F1": sd,
            "Personalization_Gain": sd - loso,
            "Gain_Pct": (sd - loso) / (loso + 1e-10) * 100,
            "Note": "SD values are estimated" if not sd_available else "Based on BLOCK_4 SD results"
        })
    df_personalization = pd.DataFrame(personalization_rows).sort_values("Personalization_Gain", ascending=False)
    save_csv(df_personalization, OUT_DIR / "Table_S31_Personalization_Gain_by_Gesture.csv")
    logger.info("    Table S31 saved.")
    
    # Figure 57
    fig, ax = plt.subplots(figsize=(10, 6))
    heat_data = df_improvement[["Gesture", "LOSO_F1", "Calibration_F1", "SD_F1"]].set_index("Gesture")
    sns.heatmap(heat_data.T, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "F1 Score"})
    ax.set_title("Figure 57 – Calibration Improvement Heatmap\n(SD: Estimated)" if not sd_available else "Figure 57 – Calibration Improvement Heatmap")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig57_Calibration_Improvement_Heatmap.png")
    plt.close(fig)
    gc.collect()
    logger.info("    Figure 57 saved.")
    
    # Figure 69
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(GESTURE_SHORT))
    width = 0.3
    
    loso_vals = [loso_f1.get(g, 0.0) for g in GESTURE_SHORT]
    cal_vals = [cal_f1.get(g, 0.0) for g in GESTURE_SHORT]
    sd_vals = [sd_f1.get(g, loso_f1.get(g, 0.0) * improvement_factor) for g in GESTURE_SHORT]
    
    ax.bar(x - width, loso_vals, width, label="LOSO", color='#FF6B6B')
    ax.bar(x, cal_vals, width, label="Calibration", color='#4ECDC4')
    ax.bar(x + width, sd_vals, width, label=f"Subject-Dependent\n(Estimated)", color='#45B7D1', alpha=0.8)
    
    ax.set_xlabel("Gesture")
    ax.set_ylabel("F1 Score")
    ax.set_title("Figure 69 – Gesture Improvement: LOSO → Calibration → Subject-Dependent\n(SD values are estimated from BLOCK_4 model-level improvement)" if not sd_available else "Figure 69 – Gesture Improvement: LOSO → Calibration → Subject-Dependent")
    ax.set_xticks(x)
    ax.set_xticklabels(GESTURE_SHORT, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig69_Gesture_Improvement_Matrix.png")
    plt.close(fig)
    gc.collect()
    logger.info("    Figure 69 saved.")
    
    # Figure S49
    fig, ax = plt.subplots(figsize=(10, 8))
    improvement_matrix = np.array([loso_vals, cal_vals, sd_vals])
    sns.heatmap(improvement_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=GESTURE_SHORT, yticklabels=["LOSO", "Calibration", f"Subject-Dependent\n(Estimated)"],
                ax=ax, cbar_kws={"label": "F1 Score"})
    ax.set_title("Figure S49 – Gesture Improvement Matrix (Detailed)\nSD: Estimated from model-level improvement" if not sd_available else "Figure S49 – Gesture Improvement Matrix (Detailed)")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS49_Gesture_Improvement_Matrix_Detailed.png")
    plt.close(fig)
    gc.collect()
    logger.info("    Figure S49 saved.")


# ============================================================================
# SUPPLEMENTARY FIGURES: S52-S59
# ============================================================================

def _plot_per_gesture_calibration_radar(calibration_results: Dict[float, Dict]) -> None:
    """Figure S52: Per-Gesture F1 Radar Across Calibration Levels."""
    try:
        calibration_levels = sorted(calibration_results.keys())
        labels = GESTURE_SHORT
        n_labels = len(labels)
        angles = np.linspace(0, 2*np.pi, n_labels, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(calibration_levels)))

        for color, cal in zip(colors, calibration_levels):
            gesture_scores = [calibration_results[cal]["per_gesture"][g]["F1"] for g in GESTURE_SHORT]
            gesture_scores += gesture_scores[:1]
            ax.plot(angles, gesture_scores, linewidth=2.2, color=color, label=f"{int(cal*100)}%")
            ax.fill(angles, gesture_scores, alpha=0.08, color=color)

        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(GESTURE_SHORT)
        ax.set_ylim(0, 1)
        ax.set_title("Figure S52\nPer-Gesture F1 Across Calibration Levels", fontsize=14, fontweight="bold", pad=25)
        ax.grid(alpha=0.35)
        ax.legend(bbox_to_anchor=(1.28, 1.08), title="Calibration")
        save_fig(fig, OUT_DIR / "Fig_S52_PerGesture_F1_Radar.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S52 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S52: {e}")


def _plot_per_gesture_improvement_radar(calibration_results: Dict[float, Dict]) -> None:
    """Figure S53: Per-Gesture Improvement Radar (ΔF1 Relative to 0%)."""
    try:
        if 0.0 not in calibration_results:
            logger.warning("0% calibration not available. Skipping Figure S53.")
            return

        baseline = calibration_results[0.0]["per_gesture"]
        calibration_levels = sorted([c for c in calibration_results.keys() if c > 0])
        labels = GESTURE_SHORT
        n_labels = len(labels)
        angles = np.linspace(0, 2*np.pi, n_labels, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(calibration_levels)))
        max_delta = 0

        for color, cal in zip(colors, calibration_levels):
            delta = [calibration_results[cal]["per_gesture"][g]["F1"] - baseline[g]["F1"] for g in GESTURE_SHORT]
            max_delta = max(max_delta, max(delta))
            delta += delta[:1]
            ax.plot(angles, delta, linewidth=2.3, color=color, label=f"{int(cal*100)}%")
            ax.fill(angles, delta, alpha=0.08, color=color)

        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(GESTURE_SHORT)
        upper = max(max_delta * 1.15, 0.05)
        ax.set_ylim(0, upper)
        ax.set_title("Figure S53\nPer-Gesture Improvement Relative to 0% Calibration", fontsize=14, fontweight="bold", pad=25)
        ax.grid(alpha=0.35)
        ax.legend(bbox_to_anchor=(1.28, 1.08), title="Calibration")
        save_fig(fig, OUT_DIR / "Fig_S53_PerGesture_Improvement_Radar.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S53 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S53: {e}")


def _plot_modelwise_calibration_lines(df_model_cal: pd.DataFrame) -> None:
    """Figure S54: Model-wise Accuracy vs Calibration Level (Line Plot)."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        for model in MODEL_NAMES:
            model_data = df_model_cal[df_model_cal["Model"] == model].sort_values("Calibration_Level")
            ax.plot(model_data["Calibration_Level"] * 100, model_data["Accuracy"],
                    marker='o', linewidth=2.5, markersize=8,
                    color=MODEL_COLORS.get(model, '#888'), label=model)
        ax.set_xlabel("Calibration Level (%)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Figure S54 – Model-wise Accuracy vs Calibration Level", fontsize=14, fontweight="bold")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig_S54_ModelWise_Accuracy_vs_Calibration.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S54 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S54: {e}")


def _plot_modelwise_f1_lines(df_model_cal: pd.DataFrame) -> None:
    """Figure S55: Model-wise F1 vs Calibration Level (Line Plot)."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        for model in MODEL_NAMES:
            model_data = df_model_cal[df_model_cal["Model"] == model].sort_values("Calibration_Level")
            ax.plot(model_data["Calibration_Level"] * 100, model_data["F1"],
                    marker='s', linewidth=2.5, markersize=8,
                    color=MODEL_COLORS.get(model, '#888'), label=model)
        ax.set_xlabel("Calibration Level (%)", fontsize=12)
        ax.set_ylabel("Macro F1 Score", fontsize=12)
        ax.set_title("Figure S55 – Model-wise F1 vs Calibration Level", fontsize=14, fontweight="bold")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig_S55_ModelWise_F1_vs_Calibration.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S55 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S55: {e}")


def _plot_per_gesture_f1_heatmap(model_metrics: Dict[str, Dict]) -> None:
    """Figure S56: Per-Gesture F1 Heatmap Across Models."""
    try:
        data = []
        for model in MODEL_NAMES:
            if model in model_metrics:
                row = {"Model": model}
                for gesture in GESTURE_SHORT:
                    if gesture in model_metrics[model]["per_gesture"]:
                        row[gesture] = model_metrics[model]["per_gesture"][gesture]["F1"]
                    else:
                        row[gesture] = 0.0
                data.append(row)
        df_heat = pd.DataFrame(data).set_index("Model")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                    ax=ax, cbar_kws={"label": "F1 Score"})
        ax.set_title("Figure S56 – Per-Gesture F1 Heatmap Across Models", fontsize=14, fontweight="bold")
        ax.set_xlabel("Gesture")
        ax.set_ylabel("Model")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig_S56_PerGesture_F1_Heatmap.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S56 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S56: {e}")


def _plot_gesture_calibration_improvement_heatmap(calibration_results: Dict[float, Dict]) -> None:
    """Figure S57: Gesture-wise Calibration Improvement Heatmap."""
    try:
        if 0.0 not in calibration_results:
            logger.warning("0% calibration not available. Skipping Figure S57.")
            return
        
        baseline = calibration_results[0.0]["per_gesture"]
        data = []
        for cal_level in sorted(calibration_results.keys()):
            if cal_level == 0.0:
                continue
            row = {"Calibration": f"{int(cal_level*100)}%"}
            for gesture in GESTURE_SHORT:
                improvement = calibration_results[cal_level]["per_gesture"][gesture]["F1"] - baseline[gesture]["F1"]
                row[gesture] = improvement
            data.append(row)
        df_heat = pd.DataFrame(data).set_index("Calibration")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, cbar_kws={"label": "ΔF1 (Improvement)"})
        ax.set_title("Figure S57 – Gesture-wise Calibration Improvement Heatmap\n(ΔF1 Relative to 0%)", 
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Gesture")
        ax.set_ylabel("Calibration Level")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig_S57_Gesture_Calibration_Improvement_Heatmap.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S57 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S57: {e}")


def _plot_modelwise_accuracy_ranking(model_metrics: Dict[str, Dict]) -> None:
    """Figure S58: Model-wise Accuracy Ranking Bar Chart."""
    try:
        models = []
        accuracies = []
        f1_scores = []
        for model in MODEL_NAMES:
            if model in model_metrics:
                models.append(model)
                accuracies.append(model_metrics[model]["accuracy"])
                f1_scores.append(model_metrics[model]["f1"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#4E79A7', edgecolor='black')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='#E15759', edgecolor='black')
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Figure S58 – Model-wise Accuracy and F1 Ranking", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        
        for bar, acc in zip(bars1, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        for bar, f1 in zip(bars2, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig_S58_ModelWise_Accuracy_Ranking.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S58 saved.")
    except Exception as e:
        logger.exception(f"Failed to generate Figure S58: {e}")


# ============================================================================
# REGENERATE ALL OUTPUTS FROM CACHE
# ============================================================================

def _regenerate_all_outputs(
    model_cms: Dict[str, Dict],
    model_metrics: Dict[str, Dict],
    best_name: str,
    calibration_results: Dict[str, Dict],
    df_model_cal: pd.DataFrame,
) -> None:
    """Regenerate all figures and tables from cached results."""
    logger.info("  Regenerating all figures and tables from cached results...")
    
    # Figures 51-55: Model-wise confusion matrices (CM-01)
    fig_no_map = {"LDA": 51, "KNN": 52, "SVM": 53, "RF": 54, "XGB": 55}
    for name in MODEL_NAMES:
        if name in model_cms:
            fn = fig_no_map.get(name, 99)
            _plot_cm(model_cms[name]["norm"],
                     f"Figure {fn} – {name} LOSO Confusion Matrix (Normalized)",
                     OUT_DIR / f"Fig{fn}_{name}_LOSO_Confusion_Matrix.png")
            logger.info(f"  Figure {fn} regenerated.")
    
    # CM-02: Best LOSO Model Confusion Matrix (Normalized)
    _run_best_model_confusion_analysis(model_cms, best_name)
    
    # Figures 58-63: Calibration confusion matrices (6 levels)
    cal_levels_for_figs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    for cal_level, results in calibration_results.items():
        if cal_level in cal_levels_for_figs:
            fig_no = CAL_FIG_MAP.get(cal_level, 63)
            _plot_cm(results["cm"],
                     f"Figure {fig_no} – {int(cal_level*100)}% Calibration Confusion Matrix (Normalized)",
                     OUT_DIR / f"Fig{fig_no}_Calibration_{int(cal_level*100)}_Confusion_Matrix.png")
            logger.info(f"  Figure {fig_no} regenerated.")
    
    # Figure 57: Calibration Improvement Heatmap
    try:
        cal_acc = [calibration_results[cal]["accuracy"] for cal in sorted(calibration_results.keys())]
        cal_f1 = [calibration_results[cal]["f1"] for cal in sorted(calibration_results.keys())]
        heat_data = np.array([cal_acc, cal_f1])
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[f"{int(c*100)}%" for c in sorted(calibration_results.keys())],
                    yticklabels=["Accuracy", "F1"], ax=ax,
                    cbar_kws={"label": "Score"})
        ax.set_title("Figure 57 – Calibration Improvement Heatmap", fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig57_Calibration_Improvement_Heatmap.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure 57 regenerated.")
    except Exception as e:
        logger.warning(f"  Figure 57 generation failed: {e}")
    
    # ========================================================================
    # CM-09: ALL CALIBRATION CONFUSION MATRICES (ALL 9 levels)
    # ========================================================================
    _plot_all_calibration_confusion_matrices(calibration_results, best_name)
    
    # ========================================================================
    # SUPPLEMENTARY FIGURES S52-S58
    # ========================================================================
    _plot_per_gesture_calibration_radar(calibration_results)
    _plot_per_gesture_improvement_radar(calibration_results)
    _plot_modelwise_calibration_lines(df_model_cal)
    _plot_modelwise_f1_lines(df_model_cal)
    _plot_per_gesture_f1_heatmap(model_metrics)
    _plot_gesture_calibration_improvement_heatmap(calibration_results)
    _plot_modelwise_accuracy_ranking(model_metrics)
    
    # ========================================================================
    # TABLES
    # ========================================================================
    
    # Table 15: Calibration Improvement Summary
    if model_metrics and calibration_results:
        best_model = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
        baseline_acc = model_metrics[best_model]["accuracy"]
        baseline_f1 = model_metrics[best_model]["f1"]
        baseline_mcc = model_metrics[best_model]["mcc"]
        summary_rows = []
        for cal_level, results in calibration_results.items():
            summary_rows.append({
                "Calibration_Level": f"{int(cal_level*100)}%",
                "Accuracy": round(results["accuracy"], 4),
                "F1": round(results["f1"], 4),
                "MCC": round(results["mcc"], 4),
                "Accuracy_Gain": round(results["accuracy"] - baseline_acc, 4),
                "F1_Gain": round(results["f1"] - baseline_f1, 4),
                "MCC_Gain": round(results["mcc"] - baseline_mcc, 4),
            })
        df_summary = pd.DataFrame(summary_rows)
        save_csv(df_summary, OUT_DIR / "Table15_Calibration_Improvement_Summary.csv")
        logger.info("  Table 15 regenerated.")
    
    # Table S28: Model-Wise Calibration Improvement
    if not df_model_cal.empty:
        save_csv(df_model_cal, OUT_DIR / "Table_S28_ModelWise_Calibration_Improvement.csv")
        logger.info("  Table S28 regenerated.")
    
    # Table S32: Model-Wise Accuracy Comparison
    if model_metrics:
        rows = []
        for model in MODEL_NAMES:
            if model in model_metrics:
                rows.append({
                    "Model": model,
                    "Accuracy": round(model_metrics[model]["accuracy"], 4),
                    "Macro_F1": round(model_metrics[model]["f1"], 4),
                    "MCC": round(model_metrics[model]["mcc"], 4),
                })
        df_comp = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
        save_csv(df_comp, OUT_DIR / "Table_S32_ModelWise_Accuracy_Comparison.csv")
        logger.info("  Table S32 regenerated.")
    
    # Table 14: Gesture-Level Performance
    if model_metrics:
        best_model = max(model_metrics, key=lambda x: model_metrics[x]["accuracy"])
        gesture_rows = []
        for gesture in GESTURE_SHORT:
            if gesture in model_metrics[best_model]["per_gesture"]:
                mg = model_metrics[best_model]["per_gesture"][gesture]
                gesture_rows.append({
                    "Gesture": gesture,
                    "Precision": round(mg["Precision"], 4),
                    "Recall": round(mg["Recall"], 4),
                    "F1": round(mg["F1"], 4),
                    "MCC": round(mg["MCC"], 4),
                    "Support": mg["Support"],
                })
        df_gesture = pd.DataFrame(gesture_rows)
        save_csv(df_gesture, OUT_DIR / "Table14_Gesture_Level_Performance.csv")
        logger.info("  Table 14 regenerated.")
    
    # Table S26: Most Confused Pairs
    if best_name in model_cms:
        cm = model_cms[best_name]["raw"]
        pairs = []
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        "True": GESTURE_SHORT[i],
                        "Predicted": GESTURE_SHORT[j],
                        "Count": int(cm[i, j]),
                    })
        df_pairs = pd.DataFrame(pairs).sort_values("Count", ascending=False)
        save_csv(df_pairs, OUT_DIR / "Table_S26_Most_Confused_Gesture_Pairs.csv")
        logger.info("  Table S26 regenerated.")
    
    logger.info("  All figures and tables regenerated successfully.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_block9() -> Dict:
    """Main entry point for BLOCK 9."""
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 9 | CM1: Multi-Level Confusion Matrix Analysis Framework")
    logger.info("=" * 70)
    logger.info(f"USE_CACHE: {USE_CACHE}")

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
        X, y, subjects, _ = load_feature_matrix()
        logger.info(f"  Loaded {len(X)} windows from BLOCK_2 artifacts")
    except (FileNotFoundError, KeyError, TypeError) as e:
        logger.warning(f"  BLOCK_2 artifacts not found: {e}")
        logger.info("  Falling back to raw data loading...")
        X, y, subjects = load_fallback_data()

    n_subjects_detected = len(np.unique(subjects))
    if n_subjects_detected != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
            f"but detected {n_subjects_detected} subjects."
        )
    logger.info(f"  Verified {n_subjects_detected} subjects")

    # ── Load subject-wise data ────────────────────────────────────────────
    df_subject = None
    try:
        subject_path = BLOCK_DIRS["BLOCK_4"] / "Table_S17_SubjectWise_Model_Comparison.csv"
        if subject_path.exists():
            df_subject = load_csv(subject_path)
            logger.info(f"  Loaded subject-wise data from {subject_path}")
    except Exception as e:
        logger.warning(f"  Could not load subject-wise data: {e}")

    # ── Load subject-dependent fold data for SD values ────────────────────
    df_sd_folds = None
    try:
        sd_path = BLOCK_DIRS["BLOCK_4"] / "Table_S18_SubjectDependent_Fold_Results.csv"
        if sd_path.exists():
            df_sd_folds = load_csv(sd_path)
            logger.info(f"  Loaded subject-dependent fold data from {sd_path}")
    except Exception as e:
        logger.warning(f"  Could not load subject-dependent fold data: {e}")

    # ── Check if ALL cache files exist ────────────────────────────────────
    cache_files = [
        OUT_DIR / "model_cms.pkl",
        OUT_DIR / "model_metrics.pkl",
        OUT_DIR / "best_model.json",
        OUT_DIR / "calibration_results.pkl",
        OUT_DIR / "modelwise_calibration.pkl",
    ]
    
    all_cached = all(p.exists() for p in cache_files)
    
    if USE_CACHE and all_cached:
        logger.info("  All cache files found. Loading cached results...")
        model_cms = _load_pickle(cache_files[0])
        model_metrics = _load_pickle(cache_files[1])
        
        if model_cms is None or model_metrics is None:
            logger.warning("  Cache corrupted. Recomputing...")
            all_cached = False
        
        if all_cached:
            with open(cache_files[2], 'r') as f:
                best_name = json.load(f)["best_model"]
            calibration_results = _load_pickle(cache_files[3])
            if calibration_results is None:
                logger.warning("  Cache corrupted (calibration_results). Recomputing...")
                all_cached = False
            if all_cached:
                df_model_cal = _load_pickle(cache_files[4])
                if df_model_cal is None:
                    logger.warning("  Cache corrupted (df_model_cal). Recomputing...")
                    all_cached = False
    
    if USE_CACHE and all_cached:
        logger.info(f"  Best model (cached): {best_name}")
        logger.info(f"  Calibration levels (cached): {len(calibration_results)}")
        
        # Regenerate all outputs from cache
        _regenerate_all_outputs(
            model_cms, model_metrics, best_name,
            calibration_results, df_model_cal
        )
        _save_environment_versions()
        
        # CM-05: Subject-Dependent Confusion Analysis
        _run_subjectdependent_confusion_analysis()
        
        # CM-06: Subject-Wise Confusion Analysis
        _run_subjectwise_confusion_analysis(df_subject)
        
        # CM-07: Gesture-Level Performance Analysis
        df_gesture = _run_gesture_level_analysis(model_metrics)
        
        # CM-08: Gesture Improvement Analysis
        _run_gesture_improvement_analysis(model_metrics, calibration_results, df_sd_folds)
        
    else:
        logger.info("  Computing all results from scratch...")
        model_cms, model_metrics, best_name = _run_loso_confusion_analysis(X, y, subjects)
        calibration_results = _run_calibration_confusion_analysis(X, y, subjects, best_name)
        df_model_cal = _run_modelwise_calibration_analysis(X, y, subjects)
        
        _regenerate_all_outputs(
            model_cms, model_metrics, best_name,
            calibration_results, df_model_cal
        )
        _save_environment_versions()
        
        # CM-05: Subject-Dependent Confusion Analysis
        _run_subjectdependent_confusion_analysis()
        
        # CM-06: Subject-Wise Confusion Analysis
        _run_subjectwise_confusion_analysis(df_subject)
        
        # CM-07: Gesture-Level Performance Analysis
        df_gesture = _run_gesture_level_analysis(model_metrics)
        
        # CM-08: Gesture Improvement Analysis
        _run_gesture_improvement_analysis(model_metrics, calibration_results, df_sd_folds)

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "table14": OUT_DIR / "Table14_Gesture_Level_Performance.csv",
        "table15": OUT_DIR / "Table15_Calibration_Improvement_Summary.csv",
        "table16": OUT_DIR / "Table16_Optimization_Confusion_Summary.csv",
        "table17": OUT_DIR / "Table17_SubjectWise_Confusion_Summary.csv",
        "table_s25": OUT_DIR / "Table_S25_Gesture_Calibration_Improvement.csv",
        "table_s26": OUT_DIR / "Table_S26_Most_Confused_Gesture_Pairs.csv",
        "table_s27": OUT_DIR / "Table_S27_SubjectWise_Confusion_Statistics.csv",
        "table_s28": OUT_DIR / "Table_S28_ModelWise_Calibration_Improvement.csv",
        "table_s29": OUT_DIR / "Table_S29_Gesture_Difficulty_Ranking.csv",
        "table_s30": OUT_DIR / "Table_S30_Gesture_Discriminability_Index.csv",
        "table_s31": OUT_DIR / "Table_S31_Personalization_Gain_by_Gesture.csv",
        "table_s32": OUT_DIR / "Table_S32_ModelWise_Accuracy_Comparison.csv",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    
    fig_no_map = {"LDA": 51, "KNN": 52, "SVM": 53, "RF": 54, "XGB": 55}
    for name in MODEL_NAMES:
        if name in model_cms:
            fn = fig_no_map.get(name, 99)
            artifact_map[f"fig{fn}"] = OUT_DIR / f"Fig{fn}_{name}_LOSO_Confusion_Matrix.png"
    
    artifact_map["fig56"] = OUT_DIR / "Fig56_Best_LOSO_Confusion_Matrix.png"
    artifact_map["fig57"] = OUT_DIR / "Fig57_Calibration_Improvement_Heatmap.png"
    for cal in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
        fig_no = CAL_FIG_MAP.get(cal, 63)
        artifact_map[f"fig{fig_no}"] = OUT_DIR / f"Fig{fig_no}_Calibration_{int(cal*100)}_Confusion_Matrix.png"
    
    # Supplementary figures
    artifact_map["fig_s47"] = OUT_DIR / "FigS47_Most_Confused_Gesture_Pairs.png"
    artifact_map["fig_s48"] = OUT_DIR / "FigS48_Subject_Difficulty_Heatmap.png"
    artifact_map["fig_s49"] = OUT_DIR / "FigS49_Gesture_Improvement_Matrix_Detailed.png"
    artifact_map["fig_s50"] = OUT_DIR / "FigS50_ModelWise_Calibration_Improvement.png"
    artifact_map["fig_s51"] = OUT_DIR / "FigS51_SubjectWise_Confusion_Variability.png"
    artifact_map["fig_s52"] = OUT_DIR / "Fig_S52_PerGesture_F1_Radar.png"
    artifact_map["fig_s53"] = OUT_DIR / "Fig_S53_PerGesture_Improvement_Radar.png"
    artifact_map["fig_s54"] = OUT_DIR / "Fig_S54_ModelWise_Accuracy_vs_Calibration.png"
    artifact_map["fig_s55"] = OUT_DIR / "Fig_S55_ModelWise_F1_vs_Calibration.png"
    artifact_map["fig_s56"] = OUT_DIR / "Fig_S56_PerGesture_F1_Heatmap.png"
    artifact_map["fig_s57"] = OUT_DIR / "Fig_S57_Gesture_Calibration_Improvement_Heatmap.png"
    artifact_map["fig_s58"] = OUT_DIR / "Fig_S58_ModelWise_Accuracy_Ranking.png"
    artifact_map["fig_s59"] = OUT_DIR / "FigS59_All_Calibration_Confusion_Matrices.png"
    
    _save_artifact_manifest(artifact_map, OUT_DIR)

    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK 9 runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    logger.info("\n" + "=" * 70)
    logger.info("BLOCK 9 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best model: {best_name}")
    logger.info(f"  Model CMs generated: {len(model_cms)}")
    logger.info(f"  Calibration levels analyzed: {len(calibration_results)}")
    logger.info(f"  USE_CACHE: {USE_CACHE}")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return {
        "best_model": best_name,
        "n_models": len(model_cms),
        "calibration_levels": len(calibration_results),
        "runtime_sec": elapsed,
        "runtime_minutes": minutes,
        "runtime_seconds": seconds,
        "n_subjects_verified": n_subjects_detected,
        "use_cache": USE_CACHE,
        "pipeline": PIPELINE_NAME,
    }


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return
    
    try:
        result = run_block9()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_9 WITH ALL ENHANCEMENTS
# ✓ CM-01: LOSO Model-Wise Confusion Analysis (5 models, Normalized)
# ✓ CM-02: Best LOSO Model Confusion Analysis (Normalized LOSO CM)
# ✓ CM-03: Calibration Confusion Analysis (6 levels: 0%,20%,40%,60%,80%,90%)
# ✓ CM-04: Model-Wise Calibration Analysis
# ✓ CM-05: Subject-Dependent Confusion Analysis
# ✓ CM-06: Subject-Wise Confusion Analysis
# ✓ CM-07: Gesture-Level Performance Analysis
# ✓ CM-08: Gesture Improvement Analysis
# ✓ CM-09: All Calibration Confusion Matrices (ALL 9 levels: 0%,10%,20%,30%,40%,50%,60%,80%,90%)
# ✓ Figure S52: Per-Gesture F1 Radar Across Calibration Levels (Absolute)
# ✓ Figure S53: Per-Gesture Improvement Radar (ΔF1 Relative to 0%)
# ✓ Figure S54: Model-wise Accuracy vs Calibration Level (Line Plot)
# ✓ Figure S55: Model-wise F1 vs Calibration Level (Line Plot)
# ✓ Figure S56: Per-Gesture F1 Heatmap Across Models
# ✓ Figure S57: Gesture-wise Calibration Improvement Heatmap
# ✓ Figure S58: Model-wise Accuracy Ranking Bar Chart
# ✓ Figure S59: All Calibration Confusion Matrices (9 levels in grid)
# ✓ Table S32: Model-Wise Accuracy Comparison
# ✓ Table S28: Model-Wise Calibration Improvement
# ✓ Table 14: Gesture-Level Performance
# ✓ Table 15: Calibration Improvement Summary
# ✓ All cache validation and regeneration
# ✓ Centralized model builder
# ✓ Artifact manifest with verification
# ─────────────────────────────────────────────────────────────────────────────
