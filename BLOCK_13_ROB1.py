# BLOCK_13_ROB1.py - COMPLETE FIXED VERSION (ALL IMPORTS ADDED)
# =================
# FILE PURPOSE:
#     BLOCK 13 — ROB1: Robustness to Noise and Signal Degradation Analysis Framework.
#     Evaluates the resilience of the EMG gesture recognition system to various types
#     of noise, signal degradation, and data quality issues.
#
#     ROBUSTNESS ASSESSMENT MODULES:
#     - ROB-01: Gaussian Feature Noise
#     - ROB-02: Feature SNR Stress
#     - ROB-03: Channel Dropout Robustness
#     - ROB-04: Temporal Jitter Robustness
#     - ROB-05: Feature Corruption Robustness
#     - ROB-06: Noise Type Comparison
#     - ROB-07: Robustness Score Framework
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_3/Table3_LOSO_Summary.csv (for best model selection)
#     - outputs/BLOCK_3/Table_S5_PerFold_LOSO.csv (for fold-level accuracy)
#     - outputs/BLOCK_7/Table_S44_Calibration_Statistics_by_Model.csv (optional)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_13/Table_S69_Gaussian_Noise_Results.csv
#     - outputs/BLOCK_13/Table_S70_SNR_Stress_Results.csv
#     - outputs/BLOCK_13/Table_S71_Channel_Dropout_Results.csv
#     - outputs/BLOCK_13/Table_S72_Temporal_Jitter_Results.csv
#     - outputs/BLOCK_13/Table_S73_Feature_Corruption_Results.csv
#     - outputs/BLOCK_13/Table_S74_Noise_Type_Comparison.csv
#     - outputs/BLOCK_13/Table_S75_Robustness_Score_Summary.csv
#     - outputs/BLOCK_13/environment_versions.json
#     - outputs/BLOCK_13/FigS73_Gaussian_Noise_Robustness.png
#     - outputs/BLOCK_13/FigS74_SNR_Stress_Curve.png
#     - outputs/BLOCK_13/FigS75_Channel_Dropout_Robustness.png
#     - outputs/BLOCK_13/FigS76_Temporal_Jitter_Robustness.png
#     - outputs/BLOCK_13/FigS77_Feature_Corruption_Robustness.png
#     - outputs/BLOCK_13/FigS78_Noise_Type_Comparison.png
#     - outputs/BLOCK_13/FigS79_Robustness_Score_Radar.png
#     - outputs/BLOCK_13/artifacts.json
#     - state/BLOCK_13_ROBUSTNESS.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_13_ROB1.py

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

# ── CRITICAL FIX: Disable MKL threading to prevent double-free ──────────────
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── CRITICAL FIX: Set memory allocation limits ──────────────────────────────
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["MALLOC_ARENA_MAX"] = "2"

import gc
import glob
import logging
import sys
import warnings
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

# ── FIX: Add all required imports ──────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy
import xgboost
import imblearn
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── FIX: Add sklearn imports ──────────────────────────────────────────────
from sklearn.model_selection import LeaveOneGroupOut  # <-- ADD THIS
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── FIX: Set scikit-learn and XGBoost to use single thread ──────────────────
sklearn.set_config(assume_finite=True)

# ── Suppress warnings ──────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")

# ── FIX: Import from config ──────────────────────────────────────────────────
from config import (
    BLOCK_DIRS,
    INPUT_PATH,
    GESTURE_SHORT,
    GESTURE_MAP,
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
    CALIBRATION_LEVELS,
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_feature_matrix, load_csv, save_csv, save_fig, save_json,
)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_13_ROBUSTNESS.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_13")

BLOCK_NAME = "BLOCK_13_ROBUSTNESS"
OUT_DIR    = BLOCK_DIRS["BLOCK_13"]

# ── Archive directories ──────────────────────────────────────────────────────
NOISE_ARCHIVE = OUT_DIR / "noise_archive"
SNR_ARCHIVE = OUT_DIR / "snr_archive"
DROPOUT_ARCHIVE = OUT_DIR / "dropout_archive"
JITTER_ARCHIVE = OUT_DIR / "jitter_archive"
CORRUPTION_ARCHIVE = OUT_DIR / "corruption_archive"

for d in [NOISE_ARCHIVE, SNR_ARCHIVE, DROPOUT_ARCHIVE, JITTER_ARCHIVE, CORRUPTION_ARCHIVE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model colors ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "LDA": "#4E79A7",
    "KNN": "#F28E2B",
    "SVM": "#59A14F",
    "RF": "#E15759",
    "XGB": "#B07AA1",
}


# ============================================================================
# Centralized model builder
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
    # ── Fallback: Define locally if common.py not available ────────────────
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
                        df = pd.read_csv(
                            fpath,
                            usecols=EMG_CHANNEL_COLUMNS,
                            dtype=np.float32,
                        )
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
# Artifact manifest with verification
# ============================================================================

def _save_artifact_manifest(
    artifact_map: Dict[str, Path],
    out_dir: Path,
) -> None:
    """
    Save artifacts.json manifest with file verification.
    """
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


def _get_best_model() -> str:
    """Get best model from BLOCK_3."""
    loso_path = BLOCK_DIRS["BLOCK_3"] / "Table3_LOSO_Summary.csv"
    if loso_path.exists():
        try:
            df = load_csv(loso_path)
            acc_col = None
            for col in df.columns:
                if "Accuracy_Mean" in col or "Acc_Mean" in col:
                    acc_col = col
                    break
            if acc_col is not None:
                best = df.loc[df[acc_col].idxmax(), "Model"]
                logger.info(f"  Best model from BLOCK_3: {best}")
                return best
        except Exception as e:
            logger.warning(f"  Could not load best model: {e}")
    logger.info("  Using fallback model: RF")
    return "RF"


# ============================================================================
# ROB-01: Gaussian Feature Noise
# ============================================================================

def _run_gaussian_noise_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_model: str,
) -> pd.DataFrame:
    """
    ROB-01: Gaussian noise robustness analysis.
    """
    logger.info("  ROB-01: Gaussian Feature Noise...")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = []
    loso = LeaveOneGroupOut()
    fold_indices = list(loso.split(X, y, subjects))[:10]
    
    for noise_level in noise_levels:
        logger.info(f"    Noise level: {noise_level}")
        all_true = []
        all_pred = []
        
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            if noise_level > 0:
                feat_std = np.std(X_tr_raw, axis=0)
                noise = np.random.default_rng(SEED + fold_idx).normal(0, noise_level * feat_std, X_tr_raw.shape)
                X_tr_noisy = X_tr_raw + noise
            else:
                X_tr_noisy = X_tr_raw
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_noisy, y_tr)
            X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"gauss_fold{fold_idx}")
            
            try:
                clf = _build_model(best_model)
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} failed: {e}")
            
            gc.collect()
        
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            acc = accuracy_score(y_true, y_pred)
            results.append({
                "Noise_Level": noise_level,
                "Accuracy": round(acc, 4),
                "Relative_Accuracy": round(acc / (results[0]["Accuracy"] if results else 1.0), 4),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(NOISE_ARCHIVE / "gaussian_noise_results.csv", index=False)
    save_csv(df, OUT_DIR / "Table_S69_Gaussian_Noise_Results.csv")
    logger.info("    Table S69 saved.")
    
    return df


# ============================================================================
# ROB-02: Feature SNR Stress
# ============================================================================

def _run_snr_stress_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_model: str,
) -> pd.DataFrame:
    """
    ROB-02: SNR stress analysis.
    """
    logger.info("  ROB-02: Feature SNR Stress...")
    
    snr_levels = [100, 50, 20, 10, 5, 2, 1]
    
    results = []
    loso = LeaveOneGroupOut()
    fold_indices = list(loso.split(X, y, subjects))[:10]
    
    for snr in snr_levels:
        logger.info(f"    SNR: {snr}")
        all_true = []
        all_pred = []
        
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            if snr < float('inf'):
                signal_power = np.mean(X_tr_raw ** 2)
                noise_power = signal_power / (snr + 1e-10)
                noise_std = np.sqrt(noise_power)
                noise = np.random.default_rng(SEED + fold_idx).normal(0, noise_std, X_tr_raw.shape)
                X_tr_noisy = X_tr_raw + noise
            else:
                X_tr_noisy = X_tr_raw
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_noisy, y_tr)
            X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"snr_fold{fold_idx}")
            
            try:
                clf = _build_model(best_model)
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} failed: {e}")
            
            gc.collect()
        
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            acc = accuracy_score(y_true, y_pred)
            results.append({
                "SNR": snr,
                "Accuracy": round(acc, 4),
                "Relative_Accuracy": round(acc / (results[0]["Accuracy"] if results else 1.0), 4),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(SNR_ARCHIVE / "snr_stress_results.csv", index=False)
    save_csv(df, OUT_DIR / "Table_S70_SNR_Stress_Results.csv")
    logger.info("    Table S70 saved.")
    
    return df


# ============================================================================
# ROB-03: Channel Dropout Robustness
# ============================================================================

def _run_channel_dropout_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_model: str,
) -> pd.DataFrame:
    """
    ROB-03: Channel dropout robustness analysis.
    """
    logger.info("  ROB-03: Channel Dropout Robustness...")
    
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    results = []
    loso = LeaveOneGroupOut()
    fold_indices = list(loso.split(X, y, subjects))[:10]
    
    for dropout_rate in dropout_rates:
        logger.info(f"    Dropout rate: {dropout_rate}")
        all_true = []
        all_pred = []
        
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            if dropout_rate > 0:
                rng = np.random.default_rng(SEED + fold_idx)
                mask = rng.random(X_tr_raw.shape) > dropout_rate
                X_tr_dropout = X_tr_raw * mask
            else:
                X_tr_dropout = X_tr_raw
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_dropout, y_tr)
            X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"dropout_fold{fold_idx}")
            
            try:
                clf = _build_model(best_model)
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} failed: {e}")
            
            gc.collect()
        
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            acc = accuracy_score(y_true, y_pred)
            results.append({
                "Dropout_Rate": dropout_rate,
                "Accuracy": round(acc, 4),
                "Relative_Accuracy": round(acc / (results[0]["Accuracy"] if results else 1.0), 4),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(DROPOUT_ARCHIVE / "channel_dropout_results.csv", index=False)
    save_csv(df, OUT_DIR / "Table_S71_Channel_Dropout_Results.csv")
    logger.info("    Table S71 saved.")
    
    return df


# ============================================================================
# ROB-04: Temporal Jitter Robustness
# ============================================================================

def _run_temporal_jitter_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_model: str,
) -> pd.DataFrame:
    """
    ROB-04: Temporal jitter robustness analysis.
    """
    logger.info("  ROB-04: Temporal Jitter Robustness...")
    
    jitter_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = []
    loso = LeaveOneGroupOut()
    fold_indices = list(loso.split(X, y, subjects))[:10]
    
    for jitter in jitter_levels:
        logger.info(f"    Jitter: {jitter}")
        all_true = []
        all_pred = []
        
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            if jitter > 0:
                n_samples = X_tr_raw.shape[0]
                rng = np.random.default_rng(SEED + fold_idx)
                shifts = rng.integers(-int(jitter * n_samples), int(jitter * n_samples), size=n_samples)
                X_tr_jitter = np.zeros_like(X_tr_raw)
                for i in range(n_samples):
                    shift = shifts[i]
                    if 0 <= i + shift < n_samples:
                        X_tr_jitter[i] = X_tr_raw[i + shift]
                    else:
                        X_tr_jitter[i] = X_tr_raw[i]
            else:
                X_tr_jitter = X_tr_raw
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_jitter, y_tr)
            X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"jitter_fold{fold_idx}")
            
            try:
                clf = _build_model(best_model)
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} failed: {e}")
            
            gc.collect()
        
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            acc = accuracy_score(y_true, y_pred)
            results.append({
                "Jitter_Level": jitter,
                "Accuracy": round(acc, 4),
                "Relative_Accuracy": round(acc / (results[0]["Accuracy"] if results else 1.0), 4),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(JITTER_ARCHIVE / "temporal_jitter_results.csv", index=False)
    save_csv(df, OUT_DIR / "Table_S72_Temporal_Jitter_Results.csv")
    logger.info("    Table S72 saved.")
    
    return df


# ============================================================================
# ROB-05: Feature Corruption Robustness
# ============================================================================

def _run_feature_corruption_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    best_model: str,
) -> pd.DataFrame:
    """
    ROB-05: Feature corruption robustness analysis.
    """
    logger.info("  ROB-05: Feature Corruption Robustness...")
    
    corruption_rates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
    
    results = []
    loso = LeaveOneGroupOut()
    fold_indices = list(loso.split(X, y, subjects))[:10]
    
    for corr_rate in corruption_rates:
        logger.info(f"    Corruption rate: {corr_rate}")
        all_true = []
        all_pred = []
        
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            if corr_rate > 0:
                rng = np.random.default_rng(SEED + fold_idx)
                X_tr_corr = X_tr_raw.copy()
                n_samples, n_features = X_tr_raw.shape
                n_corrupt = int(corr_rate * n_samples * n_features)
                for _ in range(min(n_corrupt, n_samples * n_features // 2)):
                    sample_idx = rng.integers(0, n_samples)
                    feat_idx = rng.integers(0, n_features)
                    feat_values = X_tr_raw[:, feat_idx]
                    X_tr_corr[sample_idx, feat_idx] = rng.choice(feat_values)
            else:
                X_tr_corr = X_tr_raw
            
            X_tr_sel, sc, sv, sk = build_feature_pipeline(X_tr_corr, y_tr)
            X_te_sel = apply_feature_pipeline(X_te_raw, sc, sv, sk)
            X_tr_final, y_tr_final = apply_smote(X_tr_sel, y_tr, tag=f"corrupt_fold{fold_idx}")
            
            try:
                clf = _build_model(best_model)
                clf.fit(X_tr_final, y_tr_final)
                preds = clf.predict(X_te_sel)
                all_true.extend(y_te.tolist())
                all_pred.extend(preds.tolist())
            except Exception as e:
                logger.warning(f"      Fold {fold_idx} failed: {e}")
            
            gc.collect()
        
        if all_true:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            acc = accuracy_score(y_true, y_pred)
            results.append({
                "Corruption_Rate": corr_rate,
                "Accuracy": round(acc, 4),
                "Relative_Accuracy": round(acc / (results[0]["Accuracy"] if results else 1.0), 4),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(CORRUPTION_ARCHIVE / "feature_corruption_results.csv", index=False)
    save_csv(df, OUT_DIR / "Table_S73_Feature_Corruption_Results.csv")
    logger.info("    Table S73 saved.")
    
    return df


# ============================================================================
# ROB-06: Noise Type Comparison
# ============================================================================

def _run_noise_type_comparison(
    gaussian_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    jitter_df: pd.DataFrame,
    corruption_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ROB-06: Noise type comparison.
    """
    logger.info("  ROB-06: Noise Type Comparison...")
    
    comparison_data = []
    
    if not gaussian_df.empty:
        for _, row in gaussian_df.iterrows():
            comparison_data.append({
                "Noise_Type": "Gaussian",
                "Noise_Level": f"{row['Noise_Level']:.3f}",
                "Noise_Value": row['Noise_Level'],
                "Accuracy": row['Accuracy'],
                "Relative_Accuracy": row.get('Relative_Accuracy', 1.0),
            })
    
    if not snr_df.empty:
        for _, row in snr_df.iterrows():
            comparison_data.append({
                "Noise_Type": "SNR_Stress",
                "Noise_Level": f"{row['SNR']:.1f}",
                "Noise_Value": row['SNR'],
                "Accuracy": row['Accuracy'],
                "Relative_Accuracy": row.get('Relative_Accuracy', 1.0),
            })
    
    if not dropout_df.empty:
        for _, row in dropout_df.iterrows():
            comparison_data.append({
                "Noise_Type": "Channel_Dropout",
                "Noise_Level": f"{row['Dropout_Rate']:.3f}",
                "Noise_Value": row['Dropout_Rate'],
                "Accuracy": row['Accuracy'],
                "Relative_Accuracy": row.get('Relative_Accuracy', 1.0),
            })
    
    if not jitter_df.empty:
        for _, row in jitter_df.iterrows():
            comparison_data.append({
                "Noise_Type": "Temporal_Jitter",
                "Noise_Level": f"{row['Jitter_Level']:.3f}",
                "Noise_Value": row['Jitter_Level'],
                "Accuracy": row['Accuracy'],
                "Relative_Accuracy": row.get('Relative_Accuracy', 1.0),
            })
    
    if not corruption_df.empty:
        for _, row in corruption_df.iterrows():
            comparison_data.append({
                "Noise_Type": "Feature_Corruption",
                "Noise_Level": f"{row['Corruption_Rate']:.3f}",
                "Noise_Value": row['Corruption_Rate'],
                "Accuracy": row['Accuracy'],
                "Relative_Accuracy": row.get('Relative_Accuracy', 1.0),
            })
    
    df = pd.DataFrame(comparison_data)
    save_csv(df, OUT_DIR / "Table_S74_Noise_Type_Comparison.csv")
    logger.info("    Table S74 saved.")
    
    return df


# ============================================================================
# ROB-07: Robustness Score Framework
# ============================================================================

def _compute_robustness_scores(
    gaussian_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    jitter_df: pd.DataFrame,
    corruption_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ROB-07: Robustness score framework.
    """
    logger.info("  ROB-07: Robustness Score Framework...")
    
    scores = []
    
    def compute_robustness_score(df: pd.DataFrame, noise_col: str, acc_col: str = 'Accuracy') -> float:
        if df.empty:
            return 0.0
        x = df[noise_col].values
        y = df[acc_col].values
        x_norm = x / (x.max() + 1e-10)
        auc = np.trapz(y, x_norm) / (len(x) + 1e-10)
        return min(1.0, auc)
    
    gaussian_score = compute_robustness_score(gaussian_df, 'Noise_Level')
    scores.append({
        "Noise_Type": "Gaussian_Noise",
        "Robustness_Score": round(gaussian_score, 4),
        "Interpretation": "High" if gaussian_score > 0.7 else "Medium" if gaussian_score > 0.4 else "Low"
    })
    
    snr_score = compute_robustness_score(snr_df, 'SNR')
    scores.append({
        "Noise_Type": "SNR_Stress",
        "Robustness_Score": round(snr_score, 4),
        "Interpretation": "High" if snr_score > 0.7 else "Medium" if snr_score > 0.4 else "Low"
    })
    
    dropout_score = compute_robustness_score(dropout_df, 'Dropout_Rate')
    scores.append({
        "Noise_Type": "Channel_Dropout",
        "Robustness_Score": round(dropout_score, 4),
        "Interpretation": "High" if dropout_score > 0.7 else "Medium" if dropout_score > 0.4 else "Low"
    })
    
    jitter_score = compute_robustness_score(jitter_df, 'Jitter_Level')
    scores.append({
        "Noise_Type": "Temporal_Jitter",
        "Robustness_Score": round(jitter_score, 4),
        "Interpretation": "High" if jitter_score > 0.7 else "Medium" if jitter_score > 0.4 else "Low"
    })
    
    corruption_score = compute_robustness_score(corruption_df, 'Corruption_Rate')
    scores.append({
        "Noise_Type": "Feature_Corruption",
        "Robustness_Score": round(corruption_score, 4),
        "Interpretation": "High" if corruption_score > 0.7 else "Medium" if corruption_score > 0.4 else "Low"
    })
    
    df = pd.DataFrame(scores)
    save_csv(df, OUT_DIR / "Table_S75_Robustness_Score_Summary.csv")
    logger.info("    Table S75 saved.")
    
    return df


# ============================================================================
# FIGURES S73-S79
# ============================================================================

def _figs_s73_s79(
    gaussian_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    dropout_df: pd.DataFrame,
    jitter_df: pd.DataFrame,
    corruption_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> None:
    """Generate supplementary figures S73-S79."""
    
    # Figure S73 – Gaussian Noise Robustness
    if not gaussian_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gaussian_df["Noise_Level"], gaussian_df["Accuracy"], 
                marker='o', lw=2, color='#1976D2')
        ax.set_xlabel("Noise Level (σ)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S73 – Gaussian Noise Robustness")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS73_Gaussian_Noise_Robustness.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S73 saved.")
    
    # Figure S74 – SNR Stress Curve
    if not snr_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(snr_df["SNR"], snr_df["Accuracy"], 
                marker='s', lw=2, color='#E15759')
        ax.set_xlabel("SNR")
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S74 – SNR Stress Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS74_SNR_Stress_Curve.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S74 saved.")
    
    # Figure S75 – Channel Dropout Robustness
    if not dropout_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dropout_df["Dropout_Rate"], dropout_df["Accuracy"], 
                marker='^', lw=2, color='#59A14F')
        ax.set_xlabel("Dropout Rate")
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S75 – Channel Dropout Robustness")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS75_Channel_Dropout_Robustness.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S75 saved.")
    
    # Figure S76 – Temporal Jitter Robustness
    if not jitter_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(jitter_df["Jitter_Level"], jitter_df["Accuracy"], 
                marker='D', lw=2, color='#F28E2B')
        ax.set_xlabel("Jitter Level")
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S76 – Temporal Jitter Robustness")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS76_Temporal_Jitter_Robustness.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S76 saved.")
    
    # Figure S77 – Feature Corruption Robustness
    if not corruption_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(corruption_df["Corruption_Rate"], corruption_df["Accuracy"], 
                marker='*', lw=2, color='#B07AA1')
        ax.set_xlabel("Corruption Rate")
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S77 – Feature Corruption Robustness")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS77_Feature_Corruption_Robustness.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S77 saved.")
    
    # Figure S78 – Noise Type Comparison
    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = comparison_df.pivot_table(index="Noise_Type", columns="Noise_Level", values="Relative_Accuracy")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                    cbar_kws={"label": "Relative Accuracy"})
        ax.set_title("Figure S78 – Noise Type Comparison Heatmap")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS78_Noise_Type_Comparison.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S78 saved.")
    
    # Figure S79 – Robustness Score Radar
    if not scores_df.empty:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        categories = scores_df["Noise_Type"].tolist()
        values = scores_df["Robustness_Score"].tolist()
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, linewidth=2, color='#1976D2')
        ax.fill(angles, values, alpha=0.25, color='#1976D2')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Figure S79 – Robustness Score Radar", fontsize=14, fontweight="bold")
        ax.grid(True)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS79_Robustness_Score_Radar.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S79 saved.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_block13() -> Dict:
    """Main entry point for BLOCK 13."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 13 | ROB1: Robustness to Noise and Signal Degradation")
    logger.info("=" * 70)

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
    
    # Verify subject count
    n_subjects_detected = len(np.unique(subjects))
    if n_subjects_detected != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
            f"but detected {n_subjects_detected} subjects in the data."
        )
    logger.info(f"  Verified {n_subjects_detected} subjects (matches EXPECTED_SUBJECTS)")
    
    # ── Get best model ──────────────────────────────────────────────────────
    best_model = _get_best_model()
    logger.info(f"  Best model: {best_model}")
    
    # ── ROB-01: Gaussian Feature Noise ────────────────────────────────────
    gaussian_df = _run_gaussian_noise_analysis(X, y, subjects, best_model)
    
    # ── ROB-02: Feature SNR Stress ────────────────────────────────────────
    snr_df = _run_snr_stress_analysis(X, y, subjects, best_model)
    
    # ── ROB-03: Channel Dropout Robustness ────────────────────────────────
    dropout_df = _run_channel_dropout_analysis(X, y, subjects, best_model)
    
    # ── ROB-04: Temporal Jitter Robustness ────────────────────────────────
    jitter_df = _run_temporal_jitter_analysis(X, y, subjects, best_model)
    
    # ── ROB-05: Feature Corruption Robustness ─────────────────────────────
    corruption_df = _run_feature_corruption_analysis(X, y, subjects, best_model)
    
    # ── ROB-06: Noise Type Comparison ─────────────────────────────────────
    comparison_df = _run_noise_type_comparison(
        gaussian_df, snr_df, dropout_df, jitter_df, corruption_df
    )
    
    # ── ROB-07: Robustness Score Framework ───────────────────────────────
    scores_df = _compute_robustness_scores(
        gaussian_df, snr_df, dropout_df, jitter_df, corruption_df
    )
    
    # ── Figures ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Generating Figures")
    logger.info("=" * 60)
    
    _figs_s73_s79(
        gaussian_df, snr_df, dropout_df, jitter_df, 
        corruption_df, comparison_df, scores_df
    )
    
    # ── Save environment versions ──────────────────────────────────────────
    _save_environment_versions()
    
    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "table_s69": OUT_DIR / "Table_S69_Gaussian_Noise_Results.csv",
        "table_s70": OUT_DIR / "Table_S70_SNR_Stress_Results.csv",
        "table_s71": OUT_DIR / "Table_S71_Channel_Dropout_Results.csv",
        "table_s72": OUT_DIR / "Table_S72_Temporal_Jitter_Results.csv",
        "table_s73": OUT_DIR / "Table_S73_Feature_Corruption_Results.csv",
        "table_s74": OUT_DIR / "Table_S74_Noise_Type_Comparison.csv",
        "table_s75": OUT_DIR / "Table_S75_Robustness_Score_Summary.csv",
        "env_versions": OUT_DIR / "environment_versions.json",
        "fig_s73": OUT_DIR / "FigS73_Gaussian_Noise_Robustness.png",
        "fig_s74": OUT_DIR / "FigS74_SNR_Stress_Curve.png",
        "fig_s75": OUT_DIR / "FigS75_Channel_Dropout_Robustness.png",
        "fig_s76": OUT_DIR / "FigS76_Temporal_Jitter_Robustness.png",
        "fig_s77": OUT_DIR / "FigS77_Feature_Corruption_Robustness.png",
        "fig_s78": OUT_DIR / "FigS78_Noise_Type_Comparison.png",
        "fig_s79": OUT_DIR / "FigS79_Robustness_Score_Radar.png",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)
    
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK 13 runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    
    logger.info("\n" + "=" * 70)
    logger.info("BLOCK 13 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best Model: {best_model}")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return {
        "best_model": best_model,
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
        result = run_block13()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()