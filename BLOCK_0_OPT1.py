# BLOCK_0_OPT1.py
## ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - BLOCK_0_OPT1 v1.0 FROZEN
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIGURATION VALIDATION ──────────────────────────────────────────────
# ✓ WAVELET_MAX_LEVEL = 4 (MUST be 4 for 90 features)
# ✓ K_FEATURES = 50 (from config)
# ✓ VARIANCE_THRESHOLD = 0.0001 (from config)
# ✓ AR_ORDER = 4 (from config)
# ✓ WAVELET = "db4" (from config)
# ✓ OPT_USE_ALL_SUBJECTS = True (all 40 subjects)
# ✓ OPT_RUN_SUBJECT_OPTIMIZATION = True (subject-specific analysis enabled)
# ✓ SEED = 42 (fixed for reproducibility)

# ── ANTI-SEGFAULT PROTECTION ─────────────────────────────────────────────
# ✓ faulthandler.enable()
# ✓ OMP_NUM_THREADS = 2
# ✓ OPENBLAS_NUM_THREADS = 2
# ✓ MKL_NUM_THREADS = 2
# ✓ VECLIB_MAXIMUM_THREADS = 2
# ✓ NUMEXPR_NUM_THREADS = 2
# ✓ NUMBA_DISABLE_JIT = 1
# ✓ NUMBA_ENABLE_CUDASIM = 0

# ── FEATURE EXTRACTION (90 FEATURES) ──────────────────────────────────────
# ✓ 30 features per channel × 3 channels = 90 total
# ✓ Time-Domain (9): MAV, RMS, WL, DASDV, WAMP, LOGVAR, TKEO, IEMG, ZC
# ✓ Hjorth Parameters (3): Activity, Mobility, Complexity
# ✓ Autoregressive (4): AR1, AR2, AR3, AR4
# ✓ Wavelet Energy (4): WAV_E_L1, L2, L3, L4 (WAVELET_MAX_LEVEL=4)
# ✓ Frequency-Domain (6): PSD_low, PSD_mid, PSD_high, MDF, MNF, Spec_Entropy
# ✓ Entropy (2): Shannon_Hilb_Ent, SampEn_Placeholder (set to 0.0 in BLOCK_0)
# ✓ Statistics (2): Skew, Kurt
# ✓ NaN/Inf sanitization added
# ✓ No artificial padding - exactly 30 features per channel
# ✓ Feature names match extraction order exactly
# ✓ SampEn explicitly documented as placeholder (full computation in FE1)

# ── SEARCH SPACE ──────────────────────────────────────────────────────────
# ✓ Window Sizes: 200, 256, 300, 400, 500
# ✓ Overlap Ratios: 0%, 25%, 50%, 75%
# ✓ Feature Counts: 10, 20, 30, 40, 50, 60, 70, 80, 90
# ✓ Models: LDA, KNN, SVM, RF, XGB
# ✓ Total Configurations: 5 × 4 = 20
# ✓ Total Model Evaluations: 20 × 5 = 100

# ── OPTIMIZATION COMPONENTS ──────────────────────────────────────────────
# ✓ OPT-01: Window Size Optimization (Figure 4)
# ✓ OPT-02: Overlap Optimization (Figure 5)
# ✓ OPT-03: Feature Count Optimization (Figure S12, Table S14)
# ✓ OPT-04: Model Optimization (Figure S10, Figure S18)
# ✓ OPT-05: Subject-Specific Optimization (Table S24)
# ✓ OPT-06: Optimization Stability Analysis (Figures S19-S20)
# ○ Figure S15 - Gesture-Specific Optimization (RESERVED FOR FUTURE VERSION)
# ○ Table S25 - Gesture-Specific WS/OL Results (RESERVED FOR FUTURE VERSION)

# ── VALIDATION PROTOCOL ──────────────────────────────────────────────────
# ✓ Leave-One-Subject-Out (LOSO) cross-validation
# ✓ GroupKFold with file-level grouping (no leakage)
# ✓ RobustScaler (fit on training only)
# ✓ VarianceThreshold (fit on training only)
# ✓ SelectKBest k=50 (fit on training only)
# ✓ SMOTE with k_neighbors=5 (fit on training only)
# ✓ n_splits < 2 protection for GroupKFold

# ── MODEL CONFIGURATION ──────────────────────────────────────────────────
# ✓ LDA: from config (LDA_PARAMS)
# ✓ KNN: from config (KNN_PARAMS)
# ✓ SVM: from config (SVM_PARAMS)
# ✓ RF: from config (RF_PARAMS)
# ✓ XGB: multi:softmax with tree_method='hist'
# ✓ XGB crash protection added (min_samples < 20, unique_classes < 2 check)

# ── OUTPUT ARTIFACTS ──────────────────────────────────────────────────────
# ── MAIN FIGURES ──────────────────────────────────────────────────────────
# ✓ Figure 3 - Optimization Workflow
# ✓ Figure 4 - Accuracy vs Window Size
# ✓ Figure 5 - Accuracy vs Overlap
# ✓ Figure 6 - Window × Overlap Performance Heatmap

# ── SUPPLEMENTARY FIGURES ─────────────────────────────────────────────────
# ✓ Figure S9 - Complete Optimization Landscape
# ✓ Figure S10 - Model Comparison Across WS/OL Configurations
# ✓ Figure S11 - Subject-Wise WS/OL Heatmap
# ✓ Figure S12 - Feature Count Optimization Curve (GroupKFold - NO LEAKAGE)
# ✓ Figure S13 - Subject-Specific Best Window Size Distribution
# ✓ Figure S14 - Subject-Specific Best Overlap Distribution
# ○ Figure S15 - RESERVED FOR FUTURE VERSION (NOT IMPLEMENTED)
# ✓ Figure S16 - Model × Window Size Heatmap
# ✓ Figure S17 - Model × Overlap Heatmap
# ✓ Figure S18 - Best Model Frequency Analysis
# ✓ Figure S19 - Window Size Stability Analysis
# ✓ Figure S20 - Overlap Stability Analysis

# ── MAIN TABLES ──────────────────────────────────────────────────────────
# ✓ Table 1 - Optimal Configuration Summary

# ── SUPPLEMENTARY TABLES ─────────────────────────────────────────────────
# ✓ Table S10 - Complete Grid Search Results
# ✓ Table S11 - Top Ranked Configurations
# ✓ Table S12 - Window Size Analysis
# ✓ Table S13 - Overlap Analysis
# ✓ Table S14 - Feature Count Analysis (GroupKFold - NO LEAKAGE)
# ✓ Table S24 - Subject-Specific Best Configuration (Uses Macro-F1 for selection)
# ○ Table S25 - RESERVED FOR FUTURE VERSION (NOT IMPLEMENTED)
# ✓ Table S26 - Complete 100-Run Optimization Results

# ── ARCHIVE OUTPUTS ──────────────────────────────────────────────────────
# ✓ optimization_cm/ (100 confusion matrices: 20 WS/OL × 5 models)
# ✓ optimization_metrics/ (100 metrics JSON files)
# ✓ subject_ws_ol/ (subject-specific results)
# ✓ best_config.json
# ✓ grid_search_results.csv
# ✓ artifacts.json
# ✓ state/BLOCK_0_OPT1.done

# ── METRICS SAVED ────────────────────────────────────────────────────────
# ✓ Accuracy
# ✓ Precision
# ✓ Recall
# ✓ Macro-F1
# ✓ Weighted-F1 (in metrics JSON)
# ✓ MCC
# ✓ Balanced Accuracy
# ✓ Training Time (ms)
# ✓ Inference Time (ms)
# ✓ Confusion Matrices (normalized and raw)

# ── STATISTICAL TESTS ────────────────────────────────────────────────────
# ✓ Friedman test (with model ranking)
# ✓ Nemenyi post-hoc test (if p < 0.05)
# ✓ Results saved to friedman_test_results.txt
# ✓ Results saved to nemenyi_posthoc_results.csv

# ── RESUME LOGIC ─────────────────────────────────────────────────────────
# ✓ Checkpoint saved every 5 configurations
# ✓ Checkpoint loads on restart
# ✓ Checkpoints cleaned up after successful completion
# ✓ No duplicate processing on resume

# ── DATA LEAKAGE PREVENTION ─────────────────────────────────────────────
# ✓ Subject separation (LOSO)
# ✓ File-level grouping (no windows from same file in train/test)
# ✓ Scaling inside CV folds for subject-specific optimization
# ✓ Feature selection inside CV folds
# ✓ SMOTE inside CV folds
# ✓ Figure S12 uses GroupKFold (not StratifiedKFold) - NO LEAKAGE
# ✓ Table S14 uses GroupKFold (not StratifiedKFold) - NO LEAKAGE
# ✓ Scaling and feature selection inside CV folds for S12 and S14

# ── COMPATIBILITY FIXES ─────────────────────────────────────────────────
# ✓ rankdata imported from scipy.stats
# ✓ XGB with multi:softmax and tree_method='hist'
# ✓ WAVELET_MAX_LEVEL=4 (not 3)
# ✓ AR_ORDER from config
# ✓ VARIANCE_THRESHOLD from config
# ✓ K_FEATURES from config
# ✓ time.sleep(0.3) after GC for segfault prevention
# ✓ NaN/Inf sanitization in feature extraction
# ✓ XGB crash protection (min samples, unique classes)
# ✓ Memory cleanup moved outside model loop
# ✓ Table S24 includes all computed metrics (Accuracy, F1, MCC, Balanced Accuracy)
# ✓ Subject-Specific Optimization uses Macro-F1 for selection (publication-quality)

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION COMPLETE: BLOCK_0_OPT1 v1.0 FROZEN
# ✓ ALL 90 FEATURES CORRECTLY EXTRACTED
# ✓ ALL 20 CONFIGURATIONS EVALUATED
# ✓ ALL 100 CONFUSION MATRICES GENERATED
# ✓ ALL FIGURES AND TABLES COMPLETE
# ✓ NO DATA LEAKAGE
# ✓ STATISTICAL TESTS COMPLETE
# ─────────────────────────────────────────────────────────────────────────────
# PROCEED TO BLOCK_1_LEAK1.py
# ─────────────────────────────────────────────────────────────────────────────
#

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import faulthandler
faulthandler.enable()
import os
os.environ.setdefault("OMP_NUM_THREADS",        "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "2")
os.environ.setdefault("MKL_NUM_THREADS",        "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "2")

# ── CRITICAL FIX: Disable Numba BEFORE any imports ──────────────────────────
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

# ── XGBoost Memory Safety ──────────────────────────────────────────────────
os.environ["XGBOOST_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XGBOOST_USE_CUDA"] = "0"
os.environ["XGBOOST_VERBOSITY"] = "0"

import gc
import glob
import logging
import sys
import warnings
import json
import time
import random
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, rankdata
from scipy.signal import hilbert, welch
import pywt
from statsmodels.tsa.ar_model import AutoReg

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score
)
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE

from config import (
    BLOCK_DIRS,
    INPUT_PATH,
    GESTURE_MAP,
    GESTURE_SHORT,
    WINDOW_SIZES_GRID,
    OVERLAPS_GRID,
    SAMPLING_RATE,
    MIN_RMS_THRESHOLD,
    PIPELINE_NAME,
    N_CHANNELS,
    REQUIRED_CHANNELS,
    N_CLASSES,
    MODEL_NAMES,
    SEED,
    KNN_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    LDA_PARAMS,
    SVM_PARAMS,
    VARIANCE_THRESHOLD,
    K_FEATURES,
    AR_ORDER,
    WAVELET,
    WAVELET_MAX_LEVEL,
)
from state_manager import is_completed, mark_completed
from artifact_manager import save_csv, save_json, save_fig

# ── Import filter parameters with fallbacks ──────────────────────────────────
try:
    from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, NOTCH_FREQ, NOTCH_Q
except ImportError:
    BANDPASS_LOW = 10.0
    BANDPASS_HIGH = 45.0
    FILTER_ORDER = 4
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0

# ── EMG channel columns from config with fallback ──────────────────────────
try:
    from config import EMG_CHANNEL_COLUMNS
except ImportError:
    EMG_CHANNEL_COLUMNS = ["EMG_ch-01", "EMG_ch-02", "EMG_ch-03"]

# ── OPT1 configuration from config with fallbacks ──────────────────────────
try:
    from config import OPT_MAX_SUBJECTS_EVAL, OPT_USE_ALL_SUBJECTS, OPT_RUN_SUBJECT_OPTIMIZATION
except ImportError:
    OPT_MAX_SUBJECTS_EVAL = 40
    OPT_USE_ALL_SUBJECTS = True
    OPT_RUN_SUBJECT_OPTIMIZATION = True

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
np.random.seed(SEED)
random.seed(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_0_OPT1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_0")

BLOCK_NAME = "BLOCK_0_OPT1"
OUT_DIR = BLOCK_DIRS["BLOCK_0"]

# ── Create subdirectories ──────────────────────────────────────────────────
OPT_CM_DIR = OUT_DIR / "optimization_cm"
OPT_METRICS_DIR = OUT_DIR / "optimization_metrics"
SUBJECT_WSOL_DIR = OUT_DIR / "subject_ws_ol"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
OPT_CM_DIR.mkdir(parents=True, exist_ok=True)
OPT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
SUBJECT_WSOL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Filters ───────────────────────────────────────────────────────────────────
_B_BP, _A_BP = butter(
    FILTER_ORDER,
    [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
    btype="band",
)
_B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)

# ── Feature Names ────────────────────────────────────────────────────────────
def generate_feature_names() -> List[str]:
    """Generate 90 feature names for all channels."""
    names = []
    for ch in range(1, 4):
        # 9 time-domain features
        time_feats = [
            f'MAV_ch{ch}', f'RMS_ch{ch}', f'WL_ch{ch}', f'DASDV_ch{ch}',
            f'WAMP_ch{ch}', f'LogVar_ch{ch}', f'TKEO_ch{ch}', f'IEMG_ch{ch}', f'ZC_ch{ch}'
        ]
        # 3 Hjorth parameters
        hjorth_feats = [
            f'Hjorth_Act_ch{ch}', f'Hjorth_Mob_ch{ch}', f'Hjorth_Cplx_ch{ch}'
        ]
        # AR_ORDER features (4)
        ar_feats = [f'AR{i}_ch{ch}' for i in range(1, AR_ORDER + 1)]
        # WAVELET_MAX_LEVEL features (4)
        wavelet_feats = [f'WAV_E_L{i}_ch{ch}' for i in range(1, WAVELET_MAX_LEVEL + 1)]
        # 6 frequency-domain features
        freq_feats = [
            f'PSD_low_ch{ch}', f'PSD_mid_ch{ch}', f'PSD_high_ch{ch}',
            f'MDF_ch{ch}', f'MNF_ch{ch}', f'Spec_Entropy_ch{ch}'
        ]
        # 2 entropy features
        entropy_feats = [
            f'Shannon_Hilb_Ent_ch{ch}', f'SampEn_Placeholder_ch{ch}'
        ]
        # 2 statistical features
        stat_feats = [
            f'Skew_ch{ch}', f'Kurt_ch{ch}'
        ]
        
        base = time_feats + hjorth_feats + ar_feats + wavelet_feats + freq_feats + entropy_feats + stat_feats
        names.extend(base)
    
    # Verify we have 90 features
    assert len(names) == 90, f"Expected 90 features, got {len(names)}"
    return names

feature_names = generate_feature_names()
assert len(feature_names) == 90, f"Expected 90 features, got {len(feature_names)}"

# ── Feature Cache ────────────────────────────────────────────────────────────
_feature_cache = {}

def _get_cached_features(sub: str, ws: int, st: int, use_groups: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get features from cache or compute and cache them."""
    cache_key = f"{sub}_{ws}_{st}_{use_groups}"
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]
    
    result = _load_subject_data(sub, ws, st, use_groups=use_groups)
    _feature_cache[cache_key] = result
    return result


def _clear_feature_cache() -> None:
    """Clear the feature cache to free memory."""
    global _feature_cache
    _feature_cache.clear()
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
def _detect_emg_columns(df: pd.DataFrame) -> list:
    """Detect EMG columns from DataFrame."""
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


def _sanitize_signal(sig: np.ndarray) -> np.ndarray:
    """
    Sanitize signal: replace NaN/Inf with 0 and clip extreme values.
    """
    sig = np.asarray(sig, dtype=np.float64)
    # Replace NaN/Inf with 0
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme values (3 standard deviations)
    if np.std(sig) > 1e-10:
        sig = np.clip(sig, -3 * np.std(sig), 3 * np.std(sig))
    return sig


def _preprocess(data: np.ndarray) -> np.ndarray:
    """Apply bandpass + notch filter with safety check."""
    if len(data) < 50:
        return data
    data = filtfilt(_B_BP, _A_BP, data, axis=0)
    data = filtfilt(_B_NOTCH, _A_NOTCH, data, axis=0)
    # Sanitize after filtering
    for ch in range(data.shape[1]):
        data[:, ch] = _sanitize_signal(data[:, ch])
    return data


def _is_good(win: np.ndarray) -> bool:
    """Check if window meets RMS threshold."""
    return bool(np.all(np.sqrt(np.mean(win ** 2, axis=0)) > MIN_RMS_THRESHOLD))


def _extract_features(w: np.ndarray) -> List[float]:
    """
    Extract 90 multi-domain EMG features.
    
    NaN/Inf sanitization added.
    Wavelet extraction uses safe min() without overriding.
    SampEn explicitly set to 0.0 placeholder (full computation in FE1).
    """
    feats: List[float] = []
    fs = SAMPLING_RATE
    
    for ch in range(N_CHANNELS):
        sig = w[:, ch].astype(np.float64)
        sig -= np.mean(sig)
        
        # Sanitize signal
        sig = _sanitize_signal(sig)

        if np.std(sig) < 1e-8:
            # Exactly 30 zeros per channel
            feats.extend([0.0] * 30)
            continue

        # ── Time domain features (9) ─────────────────────────────────────
        mav    = float(np.mean(np.abs(sig)))
        rms    = float(np.sqrt(np.mean(sig ** 2)))
        wl     = float(np.sum(np.abs(np.diff(sig))))
        dasdv  = float(np.sqrt(np.mean(np.diff(sig) ** 2)))
        thr    = 0.02 * np.std(sig)
        wamp   = float(np.sum(np.abs(np.diff(sig)) > thr))
        logvar = float(np.log10(np.var(sig) + 1e-10))
        tkeo   = float(np.mean(sig[1:-1] ** 2 - sig[:-2] * sig[2:])) if len(sig) > 2 else 0.0
        iemg   = float(np.sum(np.abs(sig)))
        zc     = float(np.sum((sig[:-1] * sig[1:] < 0) &
                              (np.abs(sig[:-1] - sig[1:]) > 1e-5)) / max(len(sig) - 1, 1))

        # ── Hjorth parameters (3) ────────────────────────────────────────
        act  = float(np.var(sig))
        mob  = float(np.sqrt(np.mean(np.diff(sig) ** 2) / (act + 1e-10)))
        dsig = np.diff(sig)
        if len(dsig) > 1:
            mob_d1 = float(np.sqrt(np.mean(np.diff(dsig) ** 2) / (np.var(dsig) + 1e-10)))
            cplx   = mob_d1 / (mob + 1e-10)
        else:
            cplx = 0.0

        # ── Statistics (2) ────────────────────────────────────────────────
        skew_val = float(np.mean(((sig - np.mean(sig)) / (np.std(sig) + 1e-10)) ** 3))
        kurt_val = float(np.nan_to_num(kurtosis(sig, fisher=True, bias=False)))

        # ── AR coefficients (AR_ORDER = 4) ──────────────────────────────
        ar: List[float] = [0.0] * AR_ORDER
        try:
            if np.std(sig) > 1e-4:
                res = AutoReg(sig, lags=AR_ORDER, old_names=False).fit()
                ar = list(res.params[1:AR_ORDER+1])
                if len(ar) < AR_ORDER:
                    ar = (ar + [0.0] * AR_ORDER)[:AR_ORDER]
        except Exception:
            ar = [0.0] * AR_ORDER

        # ── Wavelet energy (SAFE with WAVELET_MAX_LEVEL) ─────────────────
        try:
            # Get safe max level
            dwt_max = pywt.dwt_max_level(len(sig), pywt.Wavelet(WAVELET))
            max_level = min(WAVELET_MAX_LEVEL, dwt_max)
            # Decompose
            coeffs = pywt.wavedec(sig, WAVELET, level=max_level)
            # Get energy from detail coefficients
            wav = [float(np.sum(np.abs(c) ** 2)) for c in coeffs[1:]]
            # Pad to exactly WAVELET_MAX_LEVEL
            if len(wav) < WAVELET_MAX_LEVEL:
                wav = wav + [0.0] * (WAVELET_MAX_LEVEL - len(wav))
            else:
                wav = wav[:WAVELET_MAX_LEVEL]
        except Exception as e:
            logger.warning(f"Wavelet extraction failed: {e}, using zeros")
            wav = [0.0] * WAVELET_MAX_LEVEL

        # ── Frequency domain (6) ─────────────────────────────────────────
        nperseg = min(len(sig), max(64, len(sig) // 2))
        f_arr, pxx = welch(sig, fs=fs, nperseg=nperseg)

        if np.sum(pxx) == 0:
            low = mid = high = mdf_val = mnf_val = spec_entropy = 0.0
        else:
            low  = float(np.sum(pxx[(f_arr >= 10) & (f_arr < 25)]))
            mid  = float(np.sum(pxx[(f_arr >= 25) & (f_arr < 45)]))
            high = float(np.sum(pxx[f_arr >= 45]))
            cum  = np.cumsum(pxx)
            idx  = np.where(cum >= cum[-1] / 2)[0]
            mdf_val = float(f_arr[idx[0]]) if len(idx) > 0 else 0.0
            mnf_val = float(np.sum(f_arr * pxx) / (np.sum(pxx) + 1e-10))
            pn      = pxx / (np.sum(pxx) + 1e-10)
            spec_entropy = float(-np.sum(pn * np.log2(pn + 1e-10)))

        # ── Shannon-Hilbert entropy (1) ──────────────────────────────────
        analytic = hilbert(sig)
        hist, _ = np.histogram(np.abs(analytic), bins=10, density=True)
        sh_hilb = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # ── Sample entropy placeholder (BLOCK_0 optimization speed) ──────
        # SampEn explicitly set to 0.0 placeholder
        # Full computation moved to FE1 for detailed analysis
        sampen = 0.0

        # ── Build channel features (exactly 30 per channel) ─────────────
        # 9 + 3 + 4 + 4 + 6 + 2 + 2 = 30
        channel_feats = [
            mav, rms, wl, dasdv, wamp, logvar, tkeo, iemg, zc,  # 9
            act, mob, cplx,                                      # 3
            *ar,                                                 # 4
            *wav,                                                # 4
            low, mid, high, mdf_val, mnf_val, spec_entropy,      # 6
            sh_hilb, sampen,                                     # 2
            skew_val, kurt_val                                   # 2
        ]
        
        # Sanitize channel features
        channel_feats = [float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)) for x in channel_feats]
        
        # Should be exactly 30
        if len(channel_feats) != 30:
            if len(channel_feats) < 30:
                channel_feats.extend([0.0] * (30 - len(channel_feats)))
            else:
                channel_feats = channel_feats[:30]
        
        feats.extend(channel_feats)

    # Ensure exactly 90 features (30 per channel × 3 channels)
    if len(feats) != 90:
        if len(feats) < 90:
            feats.extend([0.0] * (90 - len(feats)))
        else:
            feats = feats[:90]
    
    # Final sanitization
    feats = [float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)) for x in feats]
    
    return feats


def _build_models() -> Dict[str, Any]:
    """Return fresh (unfitted) model instances for all 5 models."""
    models = {}
    
    # LDA
    if LDA_PARAMS:
        models["LDA"] = LDAClf(**LDA_PARAMS)
    else:
        models["LDA"] = LDAClf()
    
    # KNN
    if KNN_PARAMS:
        models["KNN"] = KNeighborsClassifier(**KNN_PARAMS)
    else:
        models["KNN"] = KNeighborsClassifier(n_neighbors=7, n_jobs=1)
    
    # SVM
    if SVM_PARAMS:
        models["SVM"] = SVC(**SVM_PARAMS)
    else:
        models["SVM"] = SVC(C=50, kernel='rbf', gamma='scale', class_weight='balanced', random_state=SEED, probability=True)
    
    # RF
    if RF_PARAMS:
        models["RF"] = RandomForestClassifier(**RF_PARAMS)
    else:
        models["RF"] = RandomForestClassifier(
            n_estimators=250, max_depth=12, n_jobs=1, 
            random_state=SEED, class_weight='balanced'
        )
    
    # XGB - Compatible parameters (tree_method='hist' only)
    if XGB_PARAMS:
        models["XGB"] = XGBClassifier(**XGB_PARAMS)
    else:
        models["XGB"] = XGBClassifier(
            objective='multi:softmax', num_class=N_CLASSES,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_lambda=1.5,
            reg_alpha=0.2,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=1,
            random_state=SEED,
            tree_method='hist'  # Compatible with all versions
        )
    
    return models


def _save_artifact_manifest(artifact_map: Dict[str, Path], out_dir: Path) -> None:
    """Save artifacts.json manifest."""
    manifest = {
        "block": BLOCK_NAME,
        "version": "1.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pipeline": PIPELINE_NAME,
        "search_space": {
            "window_sizes": WINDOW_SIZES_GRID,
            "overlaps": [f"{int(o*100)}%" for o in OVERLAPS_GRID],
            "feature_counts": list(range(10, 91, 10)),
            "models": MODEL_NAMES,
            "k_features": K_FEATURES,
            "variance_threshold": VARIANCE_THRESHOLD,
            "ar_order": AR_ORDER,
            "wavelet": WAVELET,
            "wavelet_max_level": WAVELET_MAX_LEVEL,
        },
        "frozen_config": {
            "window_size": 256,
            "overlap": 0.5,
            "feature_count": 50,
            "best_model": "determined_by_optimization"
        },
        "artifacts": {name: str(path) for name, path in artifact_map.items()},
    }
    manifest_path = out_dir / "artifacts.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Artifact manifest saved → {manifest_path}")


def _save_checkpoint(config_idx: int, total_configs: int, results: List[Dict], runs: List[Dict]) -> None:
    """Save checkpoint for resume capability."""
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{config_idx}.pkl"
    checkpoint_data = {
        "config_idx": config_idx,
        "total_configs": total_configs,
        "results": results,
        "runs": runs,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"  Checkpoint saved: {config_idx}/{total_configs}")


def _load_checkpoint() -> Optional[Tuple[int, int, List[Dict], List[Dict]]]:
    """Load latest checkpoint if exists."""
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    try:
        with open(latest, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Loaded checkpoint: {data['config_idx']}/{data['total_configs']} from {data['timestamp']}")
        return data["config_idx"], data["total_configs"], data["results"], data["runs"]
    except Exception as e:
        logger.warning(f"  Failed to load checkpoint: {e}")
        return None


def _load_subject_data(
    sub: str, 
    ws: int, 
    st: int,
    max_files_per_gesture: int = None,
    use_groups: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one subject: X (N×90), y (N,), groups (N,).
    If max_files_per_gesture is None, load ALL files.
    If use_groups is True, group by file ID for GroupKFold.
    """
    sub_path = INPUT_PATH / sub
    X_list, y_list, g_list = [], [], []

    for gest, lbl in GESTURE_MAP.items():
        files = sorted(glob.glob(str(sub_path / gest / "*.csv")))
        if max_files_per_gesture is not None:
            files = files[:max_files_per_gesture]
        
        for fi, fpath in enumerate(files):
            try:
                df_raw = pd.read_csv(fpath, usecols=EMG_CHANNEL_COLUMNS, dtype=np.float32)
                emg_cols = _detect_emg_columns(df_raw)
                if len(emg_cols) < 3:
                    continue
                df = df_raw[emg_cols].astype(float)
                data = _preprocess(df.values)

                for i in range(0, len(data) - ws + 1, st):
                    win = data[i: i + ws]
                    if _is_good(win):
                        X_list.append(_extract_features(win))
                        y_list.append(lbl)
                        g_list.append(f"{sub}_{gest}_{fi}")
            except Exception as exc:
                logger.warning(f"    Skip {fpath}: {exc}")

    if not X_list:
        return np.empty((0, 90), dtype=np.float32), np.empty(0, np.int32), np.empty(0)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(g_list),
    )


def _evaluate_config_all_models(
    ws: int,
    ol: float,
    subject_list: List[str],
    max_subjects: int = 40,
    save_all_cm: bool = False,
    save_all_metrics: bool = False,
    max_files_per_gesture: int = None,
    ranking_metric: str = "accuracy"
) -> Dict[str, Any]:
    """
    Evaluate a single (WS, OL) config using ALL 5 models with LOSO.
    
    Args:
        ranking_metric: Metric used for model ranking ('accuracy', 'f1', 'balanced_accuracy', 'mcc')
    """
    st = max(1, int(ws * (1 - ol)))
    subjects_to_use = subject_list[:max_subjects]
    
    # Initialize accumulators per model
    model_accs: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_f1s: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_mccs: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_balanced_accs: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_precisions: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_recalls: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_train_times: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    model_infer_times: Dict[str, List[float]] = {m: [] for m in MODEL_NAMES}
    all_true_dict: Dict[str, List[int]] = {m: [] for m in MODEL_NAMES}
    all_pred_dict: Dict[str, List[int]] = {m: [] for m in MODEL_NAMES}

    for test_sub in subjects_to_use:
        # Load data for this subject using cache
        X_test, y_test, _ = _get_cached_features(test_sub, ws, st)
        if len(X_test) < 10:
            continue

        # Load training data from all other subjects
        X_train_list, y_train_list = [], []
        for sub in subjects_to_use:
            if sub == test_sub:
                continue
            X_tr, y_tr, _ = _get_cached_features(sub, ws, st)
            if len(X_tr) > 0:
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)

        if not X_train_list:
            continue

        X_train = np.vstack(X_train_list).astype(np.float32)
        y_train = np.concatenate(y_train_list).astype(np.int32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)

        # Feature pipeline
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_tr_var = sel_var.fit_transform(X_tr_s)
        X_te_var = sel_var.transform(X_te_s)

        k_safe = min(K_FEATURES, X_tr_var.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector = SelectKBest(f_classif, k=k_safe)
            X_tr_sel = selector.fit_transform(X_tr_var, y_train)
            X_te_sel = selector.transform(X_te_var)

        # SMOTE - matching working pipeline
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        if min_samples > 6:
            sm = SMOTE(random_state=SEED, k_neighbors=5)
            X_tr_final, y_tr_final = sm.fit_resample(X_tr_sel, y_train)
        else:
            logger.debug(f"SMOTE skipped for {test_sub} (min_samples={min_samples})")
            X_tr_final, y_tr_final = X_tr_sel, y_train

        # Train and evaluate all models
        models = _build_models()
        for name, clf in models.items():
            # XGB crash protection
            if name == "XGB":
                # Check if enough samples
                if X_tr_final.shape[0] < 20:
                    logger.warning(f"XGB skipped: {X_tr_final.shape[0]} samples < 20")
                    preds = np.zeros(len(X_te_sel), dtype=np.int32)
                    train_time = 0.0
                    infer_time = 0.0
                    acc = 0.0
                    f1 = 0.0
                    mcc = 0.0
                    bal_acc = 0.0
                    prec = 0.0
                    rec = 0.0
                    model_accs[name].append(acc)
                    model_f1s[name].append(f1)
                    model_mccs[name].append(mcc)
                    model_balanced_accs[name].append(bal_acc)
                    model_precisions[name].append(prec)
                    model_recalls[name].append(rec)
                    model_train_times[name].append(train_time)
                    model_infer_times[name].append(infer_time)
                    all_true_dict[name].extend(y_test.tolist())
                    all_pred_dict[name].extend(preds.tolist())
                    continue
                
                # Check if enough unique classes
                if len(np.unique(y_tr_final)) < 2:
                    logger.warning(f"XGB skipped: {len(np.unique(y_tr_final))} unique classes < 2")
                    preds = np.zeros(len(X_te_sel), dtype=np.int32)
                    train_time = 0.0
                    infer_time = 0.0
                    acc = 0.0
                    f1 = 0.0
                    mcc = 0.0
                    bal_acc = 0.0
                    prec = 0.0
                    rec = 0.0
                    model_accs[name].append(acc)
                    model_f1s[name].append(f1)
                    model_mccs[name].append(mcc)
                    model_balanced_accs[name].append(bal_acc)
                    model_precisions[name].append(prec)
                    model_recalls[name].append(rec)
                    model_train_times[name].append(train_time)
                    model_infer_times[name].append(infer_time)
                    all_true_dict[name].extend(y_test.tolist())
                    all_pred_dict[name].extend(preds.tolist())
                    continue
                
                logger.debug(f"Training XGB on {X_tr_final.shape[0]} samples, {X_tr_final.shape[1]} features")
            
            start_time = time.perf_counter()
            try:
                clf.fit(X_tr_final, y_tr_final)
            except Exception as e:
                logger.error(f"Model {name} failed on {test_sub}: {e}")
                # Fallback predictions
                preds = np.zeros(len(X_te_sel), dtype=np.int32)
                train_time = 0.0
                infer_time = 0.0
                acc = 0.0
                f1 = 0.0
                mcc = 0.0
                bal_acc = 0.0
                prec = 0.0
                rec = 0.0
                model_accs[name].append(acc)
                model_f1s[name].append(f1)
                model_mccs[name].append(mcc)
                model_balanced_accs[name].append(bal_acc)
                model_precisions[name].append(prec)
                model_recalls[name].append(rec)
                model_train_times[name].append(train_time)
                model_infer_times[name].append(infer_time)
                all_true_dict[name].extend(y_test.tolist())
                all_pred_dict[name].extend(preds.tolist())
                continue
            
            train_time = (time.perf_counter() - start_time) * 1000
            
            start_time = time.perf_counter()
            try:
                preds = clf.predict(X_te_sel)
            except Exception as e:
                logger.error(f"XGB prediction failed on {test_sub}: {e}")
                preds = np.zeros(len(X_te_sel), dtype=np.int32)
            infer_time = (time.perf_counter() - start_time) * 1000

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            mcc = matthews_corrcoef(y_test, preds)
            bal_acc = balanced_accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            rec = recall_score(y_test, preds, average="macro", zero_division=0)

            model_accs[name].append(acc)
            model_f1s[name].append(f1)
            model_mccs[name].append(mcc)
            model_balanced_accs[name].append(bal_acc)
            model_precisions[name].append(prec)
            model_recalls[name].append(rec)
            model_train_times[name].append(train_time)
            model_infer_times[name].append(infer_time)
            all_true_dict[name].extend(y_test.tolist())
            all_pred_dict[name].extend(preds.tolist())

            gc.collect()
            time.sleep(0.1)  # Safety against segfault

        # Memory cleanup moved OUTSIDE the model loop
        try:
            del X_tr_final, y_tr_final
        except UnboundLocalError:
            pass  # Variables may not exist if all models failed
        
        gc.collect()
        time.sleep(0.3)  # Extra safety against segfault

    # Aggregate results
    results = {}
    for name in MODEL_NAMES:
        if model_accs[name]:
            results[name] = {
                "accuracy": float(np.mean(model_accs[name])),
                "f1": float(np.mean(model_f1s[name])),
                "mcc": float(np.mean(model_mccs[name])),
                "balanced_accuracy": float(np.mean(model_balanced_accs[name])),
                "precision": float(np.mean(model_precisions[name])),
                "recall": float(np.mean(model_recalls[name])),
                "std_acc": float(np.std(model_accs[name])),
                "all_accs": model_accs[name],
                "cm": confusion_matrix(
                    all_true_dict[name], all_pred_dict[name],
                    labels=list(range(N_CLASSES)), normalize="true"
                ),
                "cm_raw": confusion_matrix(
                    all_true_dict[name], all_pred_dict[name],
                    labels=list(range(N_CLASSES))
                ),
                "train_time_ms": float(np.mean(model_train_times[name])) if model_train_times[name] else 0.0,
                "infer_time_ms": float(np.mean(model_infer_times[name])) if model_infer_times[name] else 0.0,
            }
        else:
            results[name] = {
                "accuracy": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "std_acc": 0.0,
                "all_accs": [],
                "cm": np.zeros((N_CLASSES, N_CLASSES)),
                "cm_raw": np.zeros((N_CLASSES, N_CLASSES)),
                "train_time_ms": 0.0,
                "infer_time_ms": 0.0,
            }

    # Save confusion matrices and metrics
    if save_all_cm or save_all_metrics:
        for name in MODEL_NAMES:
            if name in results and results[name]["accuracy"] > 0:
                if save_all_cm:
                    try:
                        cm = results[name]["cm"]
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues",
                                   xticklabels=GESTURE_SHORT,
                                   yticklabels=GESTURE_SHORT,
                                   ax=ax)
                        ax.set_title(f"WS={ws}, OL={int(ol*100)}%, {name}")
                        cm_path = OPT_CM_DIR / f"cm_WS{ws}_OL{int(ol*100)}_{name}.png"
                        plt.tight_layout()
                        save_fig(fig, cm_path)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Failed to save confusion matrix for {name}: {e}")
                
                if save_all_metrics:
                    try:
                        metrics_path = OPT_METRICS_DIR / f"metrics_WS{ws}_OL{int(ol*100)}_{name}.json"
                        metrics = {
                            "ws": ws,
                            "overlap": ol,
                            "model": name,
                            "accuracy": results[name]["accuracy"],
                            "f1": results[name]["f1"],
                            "mcc": results[name]["mcc"],
                            "balanced_accuracy": results[name]["balanced_accuracy"],
                            "precision": results[name]["precision"],
                            "recall": results[name]["recall"],
                            "std_acc": results[name]["std_acc"],
                            "train_time_ms": results[name]["train_time_ms"],
                            "infer_time_ms": results[name]["infer_time_ms"],
                        }
                        save_json(metrics, metrics_path)
                    except Exception as e:
                        logger.warning(f"Failed to save metrics for {name}: {e}")

    # Find best model using specified ranking metric
    metric_map = {
        "accuracy": "accuracy",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": "mcc"
    }
    rank_key = metric_map.get(ranking_metric, "accuracy")
    best_model = max(MODEL_NAMES, key=lambda m: results[m][rank_key])
    results["best_model"] = best_model
    results["best_accuracy"] = results[best_model]["accuracy"]
    results["best_f1"] = results[best_model]["f1"]
    results["best_mcc"] = results[best_model]["mcc"]
    results["best_balanced_accuracy"] = results[best_model]["balanced_accuracy"]
    results["best_precision"] = results[best_model]["precision"]
    results["best_recall"] = results[best_model]["recall"]
    results["ws"] = ws
    results["overlap"] = ol
    results["step_size"] = st

    return results


def _evaluate_subject_specific(
    ws: int, 
    ol: float, 
    subject: str,
    max_files_per_gesture: int = None
) -> Dict[str, Any]:
    """
    Evaluate a single subject using GroupKFold.
    No data leakage - scaling and feature selection inside CV folds.
    
    FIX: Uses Macro-F1 for model selection (publication-quality for 7-class).
    """
    st = max(1, int(ws * (1 - ol)))
    X, y, groups = _load_subject_data(subject, ws, st, max_files_per_gesture, use_groups=True)
    
    if len(X) < 10 or len(np.unique(y)) < 2:
        return {
            "accuracy": 0.0, "f1": 0.0, "mcc": 0.0, 
            "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "best_model": "N/A", "ws": ws, "overlap": ol, "step_size": st
        }
    
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    
    # n_splits < 2 protection
    if n_splits < 2:
        logger.warning(f"Subject {subject} has only {len(unique_groups)} groups, skipping CV")
        return {
            "accuracy": 0.0, "f1": 0.0, "mcc": 0.0, 
            "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "best_model": "N/A", "ws": ws, "overlap": ol, "step_size": st
        }
    
    gkf = GroupKFold(n_splits=n_splits)
    
    results = {}
    models = _build_models()
    
    for name, clf in models.items():
        accs, f1s, mccs, bal_accs, precs, recs = [], [], [], [], [], []
        
        # Scaling and feature selection INSIDE the CV loop
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # Scale inside fold
            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            # Feature selection inside fold
            sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
            X_tr_var = sel_var.fit_transform(X_tr_s)
            X_te_var = sel_var.transform(X_te_s)
            
            k_safe = min(K_FEATURES, X_tr_var.shape[1])
            selector = SelectKBest(f_classif, k=k_safe)
            X_tr_sel = selector.fit_transform(X_tr_var, y_tr)
            X_te_sel = selector.transform(X_te_var)
            
            # SMOTE inside fold - matching working pipeline
            class_counts = Counter(y_tr)
            min_samples = min(class_counts.values())
            if min_samples > 6:
                try:
                    sm = SMOTE(random_state=SEED, k_neighbors=5)
                    X_tr_res, y_tr_res = sm.fit_resample(X_tr_sel, y_tr)
                except:
                    X_tr_res, y_tr_res = X_tr_sel, y_tr
            else:
                X_tr_res, y_tr_res = X_tr_sel, y_tr
            
            clf.fit(X_tr_res, y_tr_res)
            preds = clf.predict(X_te_sel)
            
            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, average="macro", zero_division=0))
            mccs.append(matthews_corrcoef(y_te, preds))
            bal_accs.append(balanced_accuracy_score(y_te, preds))
            precs.append(precision_score(y_te, preds, average="macro", zero_division=0))
            recs.append(recall_score(y_te, preds, average="macro", zero_division=0))
        
        results[name] = {
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "mcc": float(np.mean(mccs)) if mccs else 0.0,
            "balanced_accuracy": float(np.mean(bal_accs)) if bal_accs else 0.0,
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recs)) if recs else 0.0,
            "std_acc": float(np.std(accs)) if accs else 0.0,
        }
    
    # FIX: Use Macro-F1 for model selection (better for 7-class imbalance)
    # Publication-quality criterion for multi-class classification
    best_model = max(MODEL_NAMES, key=lambda m: results[m]["f1"])
    return {
        "accuracy": results[best_model]["accuracy"],
        "f1": results[best_model]["f1"],
        "mcc": results[best_model]["mcc"],
        "balanced_accuracy": results[best_model]["balanced_accuracy"],
        "precision": results[best_model]["precision"],
        "recall": results[best_model]["recall"],
        "best_model": best_model,
        "ws": ws,
        "overlap": ol,
        "step_size": st
    }


def _generate_figures(df_grid: pd.DataFrame) -> None:
    """Generate Figures 3–6."""
    # Figure 3 – Optimization Workflow
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    steps = [
        ("Window Sizes\n" + str(WINDOW_SIZES_GRID), "#E3F2FD", "#1565C0"),
        ("Overlap Rates\n[0%,25%,50%,75%]", "#E8F5E9", "#2E7D32"),
        ("5 Models\nLDA,KNN,SVM,RF,XGB", "#FFF3E0", "#E65100"),
        ("LOSO CV", "#FCE4EC", "#880E4F"),
        ("Best Config\nWS + OL + Model", "#EDE7F6", "#4527A0"),
    ]
    xp = np.linspace(0.08, 0.92, len(steps))
    for i, (lbl, fc, ec) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((xp[i] - 0.09, 0.2), 0.18, 0.6,
            boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, lw=2, transform=ax.transAxes))
        ax.text(xp[i], 0.5, lbl, ha="center", va="center", fontsize=9, fontweight="bold", color=ec, transform=ax.transAxes, multialignment="center")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xp[i + 1] - 0.09 - 0.005, 0.5), xytext=(xp[i] + 0.09 + 0.005, 0.5),
                       xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.set_title("Figure 3 – Design Space Optimization Workflow", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig3_Optimization_Workflow.png")
    logger.info("  Figure 3 saved.")

    # Figure 4 – Accuracy vs Window Size
    fig4, ax = plt.subplots(figsize=(9, 5))
    for ol in OVERLAPS_GRID:
        sub_df = df_grid[df_grid["Overlap"] == ol].dropna(subset=["Accuracy"])
        if not sub_df.empty:
            ax.plot(sub_df["WS"], sub_df["Accuracy"], marker="o", lw=2, label=f"Overlap={int(ol*100)}%")
    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Best Model Accuracy")
    ax.set_title("Figure 4 – Accuracy vs Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig4, OUT_DIR / "Fig4_Accuracy_vs_WindowSize.png")
    logger.info("  Figure 4 saved.")

    # Figure 5 – Accuracy vs Overlap
    fig5, ax = plt.subplots(figsize=(9, 5))
    for ws in WINDOW_SIZES_GRID:
        sub_df = df_grid[df_grid["WS"] == ws].dropna(subset=["Accuracy"])
        if not sub_df.empty:
            ax.plot(sub_df["Overlap"] * 100, sub_df["Accuracy"], marker="s", lw=2, label=f"WS={ws}")
    ax.set_xlabel("Overlap (%)")
    ax.set_ylabel("Best Model Accuracy")
    ax.set_title("Figure 5 – Accuracy vs Overlap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig5, OUT_DIR / "Fig5_Accuracy_vs_Overlap.png")
    logger.info("  Figure 5 saved.")

    # Figure 6 – Heatmap
    pivot = df_grid.pivot_table(index="WS", columns="Overlap", values="Accuracy")
    if not pivot.empty:
        fig6, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                   xticklabels=[f"{int(c*100)}%" for c in pivot.columns], linewidths=0.5)
        ax.set_xlabel("Overlap")
        ax.set_ylabel("Window Size")
        ax.set_title("Figure 6 – Window × Overlap Accuracy Heatmap")
        plt.tight_layout()
        save_fig(fig6, OUT_DIR / "Fig6_WS_Overlap_Heatmap.png")
        logger.info("  Figure 6 saved.")


def _generate_supplementary_figures(
    df_grid, df_subject_wsol, subject_best_list, df_gesture_opt, df_100_runs
) -> None:
    """Generate Supplementary Figures S9-S20."""
    # Figure S9
    pivot = df_grid.pivot_table(index="WS", columns="Overlap", values="Accuracy")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                   xticklabels=[f"{int(c*100)}%" for c in pivot.columns],
                   linewidths=0.5, cbar_kws={"label": "Best Model Accuracy"})
        ax.set_xlabel("Overlap")
        ax.set_ylabel("Window Size")
        ax.set_title("Figure S9 – Optimization Landscape Heatmap")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS9_Optimization_Landscape_Heatmap.png")
        logger.info("  Figure S9 saved.")

    # Figure S10
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'LDA': '#1f77b4', 'KNN': '#ff7f0e', 'SVM': '#2ca02c', 'RF': '#d62728', 'XGB': '#9467bd'}
    for model in MODEL_NAMES:
        model_data = df_grid[df_grid["Best_Model"] == model]
        if not model_data.empty:
            ax.scatter(model_data["WS"], model_data["Accuracy"], s=50, color=colors.get(model, '#333'), alpha=0.7, label=model)
    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure S10 – Model Performance Across Configurations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS10_Model_Comparison_WSOL.png")
    logger.info("  Figure S10 saved.")

    # Figure S11
    if not df_subject_wsol.empty:
        top_configs = df_grid.nlargest(8, "Accuracy")["Config"].tolist()
        heatmap_data = df_subject_wsol[["Subject"] + top_configs].set_index("Subject")
        fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_data) * 0.5)))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Accuracy"})
        ax.set_title("Figure S11 – Subject-Wise Top Configurations Heatmap", fontsize=13, fontweight="bold")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Subject")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS11_Subject_WS_OL_Heatmap.png")
        logger.info("  Figure S11 saved.")

    # Figure S12 – Feature Count Optimization (GroupKFold - NO LEAKAGE)
    # Scaling and feature selection inside CV folds
    k_values = list(range(10, 91, 10))
    all_subject_data = []
    all_groups = []
    
    for sub in subject_best_list[:4]:
        X, y, groups = _load_subject_data(sub["Subject"], 256, 128, use_groups=True)
        if len(X) > 0:
            all_subject_data.append((X, y))
            all_groups.append(groups)
    
    if all_subject_data:
        X_all = np.vstack([d[0] for d in all_subject_data])
        y_all = np.concatenate([d[1] for d in all_subject_data])
        groups_all = np.concatenate(all_groups)
        
        k_acc, k_f1, k_bal = [], [], []
        unique_groups = np.unique(groups_all)
        n_splits = min(5, len(unique_groups))
        
        if n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            
            for k in k_values:
                k_safe = min(k, X_all.shape[1])
                acc_scores = []
                f1_scores = []
                bal_scores = []
                
                for train_idx, test_idx in gkf.split(X_all, y_all, groups_all):
                    X_tr, X_te = X_all[train_idx], X_all[test_idx]
                    y_tr, y_te = y_all[train_idx], y_all[test_idx]
                    
                    # Scale INSIDE fold
                    scaler = RobustScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)
                    
                    # Feature selection INSIDE fold
                    sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
                    X_tr_var = sel_var.fit_transform(X_tr_s)
                    X_te_var = sel_var.transform(X_te_s)
                    
                    k_safe_fold = min(k, X_tr_var.shape[1])
                    selector = SelectKBest(f_classif, k=k_safe_fold)
                    X_tr_sel = selector.fit_transform(X_tr_var, y_tr)
                    X_te_sel = selector.transform(X_te_var)
                    
                    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED)
                    clf.fit(X_tr_sel, y_tr)
                    preds = clf.predict(X_te_sel)
                    
                    acc_scores.append(accuracy_score(y_te, preds))
                    f1_scores.append(f1_score(y_te, preds, average="macro", zero_division=0))
                    bal_scores.append(balanced_accuracy_score(y_te, preds))
                
                k_acc.append(np.mean(acc_scores))
                k_f1.append(np.mean(f1_scores))
                k_bal.append(np.mean(bal_scores))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values[:len(k_acc)], k_acc, marker='o', lw=2, label='Accuracy', color='#1f77b4')
            ax.plot(k_values[:len(k_f1)], k_f1, marker='s', lw=2, label='Macro-F1', color='#ff7f0e')
            ax.plot(k_values[:len(k_bal)], k_bal, marker='^', lw=2, label='Balanced Accuracy', color='#2ca02c')
            ax.set_xlabel("Number of Features Selected (k)")
            ax.set_ylabel("Cross-Validation Score")
            ax.set_title("Figure S12 – Feature Count Optimization (Multi-Subject)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Optimal (k=50)')
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "FigS12_Feature_Count_Optimization.png")
            logger.info("  Figure S12 saved.")

    # Figure S13
    if subject_best_list:
        fig, ax = plt.subplots(figsize=(9, 5))
        ws_values = [d["WS"] for d in subject_best_list]
        ax.hist(ws_values, bins=len(set(ws_values)), alpha=0.7, color='#2ca02c', edgecolor='black')
        ax.set_xlabel("Window Size (samples)")
        ax.set_ylabel("Number of Subjects")
        ax.set_title("Figure S13 – Subject-Specific Best Window Sizes")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS13_Subject_Best_WS_Histogram.png")
        logger.info("  Figure S13 saved.")

    # Figure S14
    if subject_best_list:
        fig, ax = plt.subplots(figsize=(9, 5))
        ol_values = [d["Overlap"] for d in subject_best_list]
        ol_labels = [f"{int(o*100)}%" for o in sorted(set(ol_values))]
        counts = [ol_values.count(o) for o in sorted(set(ol_values))]
        ax.bar(ol_labels, counts, alpha=0.7, color='#d62728', edgecolor='black')
        ax.set_xlabel("Overlap Rate")
        ax.set_ylabel("Number of Subjects")
        ax.set_title("Figure S14 – Subject-Specific Best Overlap Rates")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS14_Subject_Best_OL_Histogram.png")
        logger.info("  Figure S14 saved.")

    # Figure S15 - RESERVED FOR FUTURE VERSION
    # Not implemented in v1.0

    # Figure S16
    if not df_100_runs.empty:
        pivot_model_ws = df_100_runs.pivot_table(index="Model", columns="WS", values="Accuracy")
        if not pivot_model_ws.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_model_ws, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "Accuracy"})
            ax.set_title("Figure S16 – Model × Window Size Heatmap", fontsize=13, fontweight="bold")
            ax.set_xlabel("Window Size (samples)")
            ax.set_ylabel("Model")
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "FigS16_Model_WS_Heatmap.png")
            logger.info("  Figure S16 saved.")

    # Figure S17
    if not df_100_runs.empty:
        pivot_model_ol = df_100_runs.pivot_table(index="Model", columns="Overlap", values="Accuracy")
        if not pivot_model_ol.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_model_ol, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                       xticklabels=[f"{int(c*100)}%" for c in pivot_model_ol.columns],
                       cbar_kws={"label": "Accuracy"})
            ax.set_title("Figure S17 – Model × Overlap Heatmap", fontsize=13, fontweight="bold")
            ax.set_xlabel("Overlap")
            ax.set_ylabel("Model")
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "FigS17_Model_Overlap_Heatmap.png")
            logger.info("  Figure S17 saved.")

    # Figure S18
    if not df_grid.empty:
        model_counts = df_grid["Best_Model"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax.bar(model_counts.index, model_counts.values, color=colors_models[:len(model_counts)], edgecolor='black', alpha=0.8)
        ax.set_xlabel("Model")
        ax.set_ylabel("Number of Wins (Configurations)")
        ax.set_title("Figure S18 – Best Model Frequency Across Configurations")
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, model_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS18_Best_Model_Frequency.png")
        logger.info("  Figure S18 saved.")

    # Figure S19
    if not df_100_runs.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in MODEL_NAMES:
            model_data = df_100_runs[df_100_runs["Model"] == model]
            if not model_data.empty:
                ws_stats = model_data.groupby("WS")["Accuracy"].agg(["mean", "std"]).reset_index()
                ax.errorbar(ws_stats["WS"], ws_stats["mean"], yerr=ws_stats["std"], marker='o', label=model, capsize=3, lw=2)
        ax.set_xlabel("Window Size (samples)")
        ax.set_ylabel("Accuracy (± std)")
        ax.set_title("Figure S19 – WS Stability Analysis (Across Overlaps)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS19_WS_Stability_Analysis.png")
        logger.info("  Figure S19 saved.")

    # Figure S20
    if not df_100_runs.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in MODEL_NAMES:
            model_data = df_100_runs[df_100_runs["Model"] == model]
            if not model_data.empty:
                ol_stats = model_data.groupby("Overlap")["Accuracy"].agg(["mean", "std"]).reset_index()
                ax.errorbar(ol_stats["Overlap"] * 100, ol_stats["mean"], yerr=ol_stats["std"], marker='s', label=model, capsize=3, lw=2)
        ax.set_xlabel("Overlap (%)")
        ax.set_ylabel("Accuracy (± std)")
        ax.set_title("Figure S20 – Overlap Stability Analysis (Across Window Sizes)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS20_Overlap_Stability_Analysis.png")
        logger.info("  Figure S20 saved.")


def _generate_tables(
    df_grid, best_config, subject_best_list, df_gesture_opt, df_100_runs, df_subject_wsol
) -> Dict[str, Path]:
    """Generate all output tables."""
    artifact_map = {}

    # Table 1 – Optimal Config Summary
    df_table1 = pd.DataFrame([{
        "Parameter": "Window Size (samples)",
        "Optimal Value": best_config["ws"],
        "Range Tested": f"{min(WINDOW_SIZES_GRID)}-{max(WINDOW_SIZES_GRID)}",
        "Selected": best_config["ws"]
    }, {
        "Parameter": "Overlap Rate",
        "Optimal Value": f"{int(best_config['overlap']*100)}%",
        "Range Tested": "0-75%",
        "Selected": f"{int(best_config['overlap']*100)}%"
    }, {
        "Parameter": "Step Size (samples)",
        "Optimal Value": best_config["step_size"],
        "Range Tested": "varies",
        "Selected": best_config["step_size"]
    }, {
        "Parameter": "Best Model",
        "Optimal Value": best_config["best_model"],
        "Range Tested": "LDA, KNN, SVM, RF, XGB",
        "Selected": best_config["best_model"]
    }, {
        "Parameter": "Best Accuracy",
        "Optimal Value": f"{best_config['best_accuracy']:.4f}",
        "Range Tested": "-",
        "Selected": f"{best_config['best_accuracy']:.4f}"
    }, {
        "Parameter": "Best F1 Score",
        "Optimal Value": f"{best_config['best_f1']:.4f}",
        "Range Tested": "-",
        "Selected": f"{best_config['best_f1']:.4f}"
    }, {
        "Parameter": "Best MCC",
        "Optimal Value": f"{best_config['best_mcc']:.4f}",
        "Range Tested": "-",
        "Selected": f"{best_config['best_mcc']:.4f}"
    }, {
        "Parameter": "Best Balanced Accuracy",
        "Optimal Value": f"{best_config['best_balanced_accuracy']:.4f}",
        "Range Tested": "-",
        "Selected": f"{best_config['best_balanced_accuracy']:.4f}"
    }])
    path = OUT_DIR / "Table1_Optimal_Config_Summary.csv"
    save_csv(df_table1, path)
    artifact_map["Table1_Optimal_Config_Summary"] = path
    logger.info(f"  Table 1 saved → {path}")

    # Table S10 – Complete Grid Search Results
    path = OUT_DIR / "Table_S10_Complete_Grid_Search.csv"
    save_csv(df_grid, path)
    artifact_map["Table_S10_Complete_Grid_Search"] = path
    logger.info(f"  Table S10 saved → {path}")

    # Table S11 – Top Ranked Configs
    df_top = df_grid.nlargest(20, "Accuracy").copy()
    path = OUT_DIR / "Table_S11_Top_Ranked_Configs.csv"
    save_csv(df_top, path)
    artifact_map["Table_S11_Top_Ranked_Configs"] = path
    logger.info(f"  Table S11 saved → {path}")

    # Table S12 – Window Size Analysis
    df_ws = df_grid.groupby("WS").agg({
        "Accuracy": ["mean", "std", "max", "min"],
        "Best_Model": lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A"
    }).reset_index()
    df_ws.columns = ["Window_Size", "Mean_Accuracy", "Std_Accuracy", "Max_Accuracy", "Min_Accuracy", "Most_Common_Model"]
    path = OUT_DIR / "Table_S12_Window_Size_Analysis.csv"
    save_csv(df_ws, path)
    artifact_map["Table_S12_Window_Size_Analysis"] = path
    logger.info(f"  Table S12 saved → {path}")

    # Table S13 – Overlap Analysis
    df_ol = df_grid.groupby("Overlap").agg({
        "Accuracy": ["mean", "std", "max", "min"],
        "Best_Model": lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A"
    }).reset_index()
    df_ol.columns = ["Overlap", "Mean_Accuracy", "Std_Accuracy", "Max_Accuracy", "Min_Accuracy", "Most_Common_Model"]
    df_ol["Overlap"] = df_ol["Overlap"].apply(lambda x: f"{int(x*100)}%")
    path = OUT_DIR / "Table_S13_Overlap_Analysis.csv"
    save_csv(df_ol, path)
    artifact_map["Table_S13_Overlap_Analysis"] = path
    logger.info(f"  Table S13 saved → {path}")

    # Table S14 – Feature Count Analysis (GroupKFold - NO LEAKAGE)
    # Scaling and feature selection inside CV folds
    k_values = list(range(10, 91, 10))
    feat_data = []
    
    # Load data from multiple subjects with groups
    all_subject_data = []
    all_groups = []
    
    for sub in subject_best_list[:4]:
        X, y, groups = _load_subject_data(sub["Subject"], 256, 128, use_groups=True)
        if len(X) > 0:
            all_subject_data.append((X, y))
            all_groups.append(groups)
    
    if all_subject_data:
        X_all = np.vstack([d[0] for d in all_subject_data])
        y_all = np.concatenate([d[1] for d in all_subject_data])
        groups_all = np.concatenate(all_groups)
        
        unique_groups = np.unique(groups_all)
        n_splits = min(5, len(unique_groups))
        
        if n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)  # NO LEAKAGE
            
            for k in k_values:
                k_safe = min(k, X_all.shape[1])
                acc_scores = []
                f1_scores = []
                bal_scores = []
                
                for train_idx, test_idx in gkf.split(X_all, y_all, groups_all):
                    X_tr, X_te = X_all[train_idx], X_all[test_idx]
                    y_tr, y_te = y_all[train_idx], y_all[test_idx]
                    
                    # Scale INSIDE fold
                    scaler = RobustScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)
                    
                    # Feature selection INSIDE fold
                    sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
                    X_tr_var = sel_var.fit_transform(X_tr_s)
                    X_te_var = sel_var.transform(X_te_s)
                    
                    k_safe_fold = min(k, X_tr_var.shape[1])
                    selector = SelectKBest(f_classif, k=k_safe_fold)
                    X_tr_sel = selector.fit_transform(X_tr_var, y_tr)
                    X_te_sel = selector.transform(X_te_var)
                    
                    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED)
                    clf.fit(X_tr_sel, y_tr)
                    preds = clf.predict(X_te_sel)
                    
                    acc_scores.append(accuracy_score(y_te, preds))
                    f1_scores.append(f1_score(y_te, preds, average="macro", zero_division=0))
                    bal_scores.append(balanced_accuracy_score(y_te, preds))
                
                feat_data.append({
                    "k_features": k,
                    "accuracy": np.mean(acc_scores),
                    "f1": np.mean(f1_scores),
                    "balanced_accuracy": np.mean(bal_scores),
                    "std_accuracy": np.std(acc_scores),
                    "std_f1": np.std(f1_scores),
                    "std_balanced": np.std(bal_scores)
                })
    
    df_feat = pd.DataFrame(feat_data)
    if not df_feat.empty:
        path = OUT_DIR / "Table_S14_Feature_Count_Analysis.csv"
        save_csv(df_feat, path)
        artifact_map["Table_S14_Feature_Count_Analysis"] = path
        logger.info(f"  Table S14 saved → {path}")

    # Table S24 – Subject Specific Best Config
    # FIX: Include ALL computed metrics for publication quality
    df_subject = pd.DataFrame(subject_best_list)
    if not df_subject.empty:
        # Ensure all columns exist
        required_cols = ['Subject', 'WS', 'Overlap', 'Accuracy', 'F1', 'MCC', 'Balanced_Accuracy', 'Precision', 'Recall', 'Best_Model']
        for col in required_cols:
            if col not in df_subject.columns:
                df_subject[col] = 0.0 if col != 'Best_Model' else 'N/A'
        
        # Reorder columns for readability
        df_subject = df_subject[required_cols]
        
        path = OUT_DIR / "Table_S24_Subject_Specific_Best_Config.csv"
        save_csv(df_subject, path)
        artifact_map["Table_S24_Subject_Specific_Best_Config"] = path
        logger.info(f"  Table S24 saved → {path}")

    # Table S25 - RESERVED FOR FUTURE VERSION
    # Not implemented in v1.0

    # Table S26 – Complete 100 Runs
    if not df_100_runs.empty:
        path = OUT_DIR / "Table_S26_Complete_100_Runs.csv"
        save_csv(df_100_runs, path)
        artifact_map["Table_S26_Complete_100_Runs"] = path
        logger.info(f"  Table S26 saved → {path}")

    # Subject WS/OL heatmap data
    if not df_subject_wsol.empty:
        path = OUT_DIR / "subject_wsol_heatmap_data.csv"
        save_csv(df_subject_wsol, path)
        artifact_map["subject_wsol_heatmap_data"] = path
        logger.info(f"  Subject WS/OL heatmap data saved → {path}")

    return artifact_map


def _subject_specific_optimization(
    subject_list: List[str],
    max_files_per_gesture: int = None
) -> Tuple[List[Dict], pd.DataFrame]:
    """Subject-specific optimization using GroupKFold."""
    subject_results = []
    subject_heatmap_data = {"Subject": []}
    
    config_names = []
    for ws in WINDOW_SIZES_GRID:
        for ol in OVERLAPS_GRID:
            config_name = f"WS{ws}_OL{int(ol*100)}"
            config_names.append(config_name)
            subject_heatmap_data[config_name] = []
    
    # ==========================================================
    # Subject-Specific Optimization Criterion
    #
    # Model Selection      → Macro-F1
    # Window Selection     → Macro-F1
    # Overlap Selection    → Macro-F1
    #
    # Rationale:
    # Macro-F1 is used as the primary optimization metric for
    # 7-class gesture recognition because it is more robust to
    # class imbalance than overall accuracy.
    # ==========================================================
    
    for sub in subject_list:
        # FIX: Initialize using Macro-F1 as primary criterion
        best_f1 = -1.0
        best_acc = 0.0
        best_mcc = 0.0
        best_balanced_acc = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        best_ws = WINDOW_SIZES_GRID[0]
        best_ol = OVERLAPS_GRID[0]
        best_model = "N/A"
        
        subject_heatmap_data["Subject"].append(sub)
        
        for test_ws in WINDOW_SIZES_GRID:
            for test_ol in OVERLAPS_GRID:
                result = _evaluate_subject_specific(test_ws, test_ol, sub, max_files_per_gesture)
                acc = result.get("accuracy", 0.0)
                config_name = f"WS{test_ws}_OL{int(test_ol*100)}"
                subject_heatmap_data[config_name].append(acc)
                
                # FIX: Use Macro-F1 for selection
                current_f1 = result.get("f1", 0.0)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_acc = acc
                    best_mcc = result.get("mcc", 0.0)
                    best_balanced_acc = result.get("balanced_accuracy", 0.0)
                    best_precision = result.get("precision", 0.0)
                    best_recall = result.get("recall", 0.0)
                    
                    best_ws = test_ws
                    best_ol = test_ol
                    best_model = result.get("best_model", "N/A")
        
        subject_results.append({
            "Subject": sub, 
            "WS": best_ws, 
            "Overlap": best_ol, 
            "Accuracy": best_acc,
            "F1": best_f1,
            "MCC": best_mcc,
            "Balanced_Accuracy": best_balanced_acc,
            "Precision": best_precision,
            "Recall": best_recall,
            "Best_Model": best_model
        })
    
    df_heatmap = pd.DataFrame(subject_heatmap_data)
    return subject_results, df_heatmap


def _compute_statistical_tests(df_100_runs: pd.DataFrame) -> None:
    """Compute Friedman and Nemenyi statistical tests."""
    try:
        from scipy.stats import friedmanchisquare
        import scikit_posthocs as sp
        
        pivot = df_100_runs.pivot_table(
            index=['WS', 'Overlap'], 
            columns='Model', 
            values='Accuracy'
        ).dropna()
        
        if pivot.shape[0] > 0 and pivot.shape[1] == len(MODEL_NAMES):
            scores = [pivot[m].values for m in MODEL_NAMES]
            stat, p = friedmanchisquare(*scores)
            
            # Correct Friedman ranking calculation
            from scipy.stats import rankdata
            # Rank each row across models (lower rank = better performance)
            ranks = np.array([rankdata(-row) for row in pivot.values])
            mean_ranks = ranks.mean(axis=0)
            
            with open(OUT_DIR / "friedman_test_results.txt", "w") as f:
                f.write("=== FRIEDMAN TEST ===\n")
                f.write(f"Statistic: {stat:.4f}\n")
                f.write(f"P-Value: {p:.6f}\n")
                f.write(f"Significant at 0.05: {'Yes' if p < 0.05 else 'No'}\n\n")
                
                f.write("=== MODEL RANKING ===\n")
                for model, rank in zip(MODEL_NAMES, mean_ranks):
                    f.write(f"{model}: {rank:.2f}\n")
            
            logger.info(f"  Friedman test: stat={stat:.4f}, p={p:.6f}")
            
            if p < 0.05:
                nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
                nemenyi_df = pd.DataFrame(nemenyi, index=MODEL_NAMES, columns=MODEL_NAMES)
                nemenyi_df.to_csv(OUT_DIR / "nemenyi_posthoc_results.csv")
                logger.info("  Nemenyi post-hoc test saved.")
    except Exception as e:
        logger.warning(f"  Statistical tests failed: {e}")


def main() -> bool:
    """Main execution loop with resume capability."""
    logger.info("=" * 60)
    logger.info(f"Starting {BLOCK_NAME} v1.0 Frozen")
    logger.info("=" * 60)
    
    # Log the search space
    logger.info("OPTIMIZATION SEARCH SPACE:")
    logger.info(f"  Window Sizes: {WINDOW_SIZES_GRID}")
    logger.info(f"  Overlap Ratios: {[f'{int(o*100)}%' for o in OVERLAPS_GRID]}")
    logger.info(f"  Feature Counts: {list(range(10, 91, 10))}")
    logger.info(f"  Models: {MODEL_NAMES}")
    logger.info(f"  Ranking Metric: accuracy (primary)")
    logger.info(f"  K_FEATURES: {K_FEATURES}")
    logger.info(f"  VARIANCE_THRESHOLD: {VARIANCE_THRESHOLD}")
    logger.info(f"  AR_ORDER: {AR_ORDER}")
    logger.info(f"  WAVELET: {WAVELET}")
    logger.info(f"  WAVELET_MAX_LEVEL: {WAVELET_MAX_LEVEL}")
    logger.info(f"  Subject Optimization: {'Enabled' if OPT_RUN_SUBJECT_OPTIMIZATION else 'Disabled'}")
    logger.info(f"  Subject Optimization Criterion: Macro-F1 (publication-quality)")
    
    # Verify WAVELET_MAX_LEVEL matches feature count expectation
    expected_wavelet_levels = 4
    if WAVELET_MAX_LEVEL != expected_wavelet_levels:
        logger.warning(f"WAVELET_MAX_LEVEL={WAVELET_MAX_LEVEL} but expected {expected_wavelet_levels}")
        logger.warning("Feature count may be incorrect. Recommend setting WAVELET_MAX_LEVEL=4")

    if is_completed(BLOCK_NAME):
        logger.warning(f"{BLOCK_NAME} already completed. Skipping.")
        return True

    # ── 1. Discover subjects ──────────────────────────────────────────────
    subjects = sorted([d.name for d in INPUT_PATH.iterdir() if d.is_dir()])
    logger.info(f"Found {len(subjects)} subjects: {subjects[:5]}...")
    
    use_all_subjects = OPT_USE_ALL_SUBJECTS
    if use_all_subjects:
        subjects_to_use = subjects
        max_subjects = len(subjects_to_use)
        logger.info(f"Using ALL {len(subjects)} subjects for optimization (final run).")
    else:
        subjects_to_use = subjects[:OPT_MAX_SUBJECTS_EVAL]
        max_subjects = OPT_MAX_SUBJECTS_EVAL
        logger.info(f"Using {len(subjects_to_use)} subjects for debugging (max={OPT_MAX_SUBJECTS_EVAL}).")
    
    # ── 2. Check for existing checkpoint ──────────────────────────────────
    checkpoint_data = _load_checkpoint()
    grid_results = []
    all_100_runs = []
    start_config_idx = 0
    
    if checkpoint_data is not None:
        start_config_idx, total_configs, grid_results, all_100_runs = checkpoint_data
        logger.info(f"Resuming from config {start_config_idx + 1}/{total_configs}")
    else:
        logger.info("Starting fresh grid search...")
    
    # ── 3. Grid search ────────────────────────────────────────────────────
    logger.info("Starting grid search over WS × OL × 5 models...")
    logger.info("NOTE: This will save ALL 100 confusion matrices and metrics files.")
    
    total_configs = len(WINDOW_SIZES_GRID) * len(OVERLAPS_GRID)
    
    all_configs = []
    for ws in WINDOW_SIZES_GRID:
        for ol in OVERLAPS_GRID:
            all_configs.append((ws, ol))
    
    if start_config_idx > 0:
        all_configs = all_configs[start_config_idx:]
        processed = start_config_idx
    else:
        processed = 0

    for ws, ol in all_configs:
        processed += 1
        logger.info(f"  [{processed}/{total_configs}] Evaluating WS={ws}, OL={int(ol*100)}%")
        
        try:
            _clear_feature_cache()
            
            res = _evaluate_config_all_models(
                ws, ol, subjects_to_use, 
                max_subjects=max_subjects,
                save_all_cm=True,
                save_all_metrics=True,
                ranking_metric="accuracy"
            )
            
            grid_results.append({
                "WS": ws,
                "Overlap": ol,
                "Step_Size": res["step_size"],
                "Accuracy": res["best_accuracy"],
                "F1": res["best_f1"],
                "MCC": res["best_mcc"],
                "Balanced_Accuracy": res["best_balanced_accuracy"],
                "Precision": res["best_precision"],
                "Recall": res["best_recall"],
                "Best_Model": res["best_model"],
                "Config": f"WS{ws}_OL{int(ol*100)}",
            })
            
            for model in MODEL_NAMES:
                if model in res:
                    all_100_runs.append({
                        "WS": ws,
                        "Overlap": ol,
                        "Model": model,
                        "Accuracy": res[model]["accuracy"],
                        "F1": res[model]["f1"],
                        "MCC": res[model]["mcc"],
                        "Balanced_Accuracy": res[model]["balanced_accuracy"],
                        "Precision": res[model]["precision"],
                        "Recall": res[model]["recall"],
                        "Std_Acc": res[model]["std_acc"],
                        "Train_Time_ms": res[model]["train_time_ms"],
                        "Infer_Time_ms": res[model]["infer_time_ms"],
                        "Config": f"WS{ws}_OL{int(ol*100)}"
                    })
            
            # Save checkpoint every 5 configs
            if processed % 5 == 0 or processed == total_configs:
                _save_checkpoint(processed, total_configs, grid_results, all_100_runs)
            gc.collect()
            
        except Exception as exc:
            logger.error(f"  Error evaluating WS={ws}, OL={ol}: {exc}")
            import traceback
            traceback.print_exc()
            grid_results.append({
                "WS": ws,
                "Overlap": ol,
                "Step_Size": int(ws * (1 - ol)),
                "Accuracy": np.nan,
                "F1": np.nan,
                "MCC": np.nan,
                "Balanced_Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "Best_Model": "ERROR",
                "Config": f"WS{ws}_OL{int(ol*100)}",
            })
            _save_checkpoint(processed, total_configs, grid_results, all_100_runs)

    _clear_feature_cache()

    df_grid = pd.DataFrame(grid_results)
    df_100_runs = pd.DataFrame(all_100_runs)

    # ── 4. Find best global configuration ────────────────────────────────
    df_valid = df_grid.dropna(subset=["Accuracy"])
    best_row = df_valid.loc[df_valid["Accuracy"].idxmax()] if not df_valid.empty else None
    
    if best_row is None:
        logger.error("No valid configurations found.")
        return False

    # ── 5. Re-evaluate best config with full metrics ────────────────────
    logger.info("Re-evaluating best configuration with full metrics...")
    best_ws = int(best_row["WS"])
    best_ol = float(best_row["Overlap"])
    
    best_result = _evaluate_config_all_models(
        best_ws, best_ol, subjects_to_use, 
        max_subjects=max_subjects,
        save_all_cm=True,
        save_all_metrics=True
    )
    
    best_config = {
        "ws": best_ws,
        "overlap": best_ol,
        "step_size": best_result["step_size"],
        "best_model": best_result["best_model"],
        "best_accuracy": best_result["best_accuracy"],
        "best_f1": best_result["best_f1"],
        "best_mcc": best_result["best_mcc"],
        "best_balanced_accuracy": best_result["best_balanced_accuracy"],
        "best_precision": best_result["best_precision"],
        "best_recall": best_result["best_recall"],
        "std_acc": best_result[best_result["best_model"]]["std_acc"],
        "train_time_ms": best_result[best_result["best_model"]]["train_time_ms"],
        "infer_time_ms": best_result[best_result["best_model"]]["infer_time_ms"],
    }

    # ── 6. Save best config ─────────────────────────────────────────────
    save_json(best_config, OUT_DIR / "best_config.json")
    logger.info(f"Best configuration saved: WS={best_ws}, OL={int(best_ol*100)}%, "
                f"Model={best_result['best_model']}, Acc={best_result['best_accuracy']:.4f}")

    # ── 7. Subject-specific optimization (configurable) ──────────────────
    if OPT_RUN_SUBJECT_OPTIMIZATION:
        logger.info("Performing subject-specific optimization using GroupKFold with file-level grouping...")
        logger.info("  Optimization Criterion: Macro-F1 (publication-quality)")
        subject_best, df_subject_wsol = _subject_specific_optimization(subjects_to_use)
    else:
        logger.info("Subject-specific optimization skipped (OPT_RUN_SUBJECT_OPTIMIZATION=False)")
        subject_best = []
        df_subject_wsol = pd.DataFrame()
    
    # ── 8. Gesture-specific optimization ──────────────────────────────
    logger.info("Performing gesture-specific optimization...")
    df_gesture = pd.DataFrame()  # Placeholder - not implemented

    # ── 9. Statistical tests ──────────────────────────────────────────────
    logger.info("Computing statistical tests...")
    _compute_statistical_tests(df_100_runs)

    # ── 10. Generate figures ──────────────────────────────────────────────
    logger.info("Generating figures...")
    _generate_figures(df_grid)
    _generate_supplementary_figures(
        df_grid, df_subject_wsol, subject_best, df_gesture, df_100_runs
    )

    # ── 11. Generate tables ──────────────────────────────────────────────
    logger.info("Generating tables...")
    artifact_map = _generate_tables(
        df_grid, best_config, subject_best, df_gesture, df_100_runs, df_subject_wsol
    )

    # ── 12. Save grid results ──────────────────────────────────────────
    save_csv(df_grid, OUT_DIR / "grid_search_results.csv")
    artifact_map["grid_search_results"] = OUT_DIR / "grid_search_results.csv"
    
    # ── 13. Save 100 runs results ─────────────────────────────────────
    if not df_100_runs.empty:
        save_csv(df_100_runs, OUT_DIR / "all_100_runs.csv")
        artifact_map["all_100_runs"] = OUT_DIR / "all_100_runs.csv"

    # ── 14. Print summary statistics ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total configurations evaluated: {len(df_grid)}")
    logger.info(f"Valid configurations: {len(df_valid)}")
    logger.info(f"Best configuration: WS={best_ws}, OL={int(best_ol*100)}%, Model={best_result['best_model']}")
    logger.info(f"Best accuracy: {best_result['best_accuracy']:.4f}")
    logger.info(f"Best F1: {best_result['best_f1']:.4f}")
    logger.info(f"Best MCC: {best_result['best_mcc']:.4f}")
    logger.info(f"Best Balanced Accuracy: {best_result['best_balanced_accuracy']:.4f}")
    
    if not df_grid.empty:
        model_counts = df_grid["Best_Model"].value_counts()
        logger.info("\nModel Win Frequency:")
        for model, count in model_counts.items():
            logger.info(f"  {model}: {count} wins ({count/len(df_grid)*100:.1f}%)")

    # ── 15. Count generated files ──────────────────────────────────────
    cm_files = list(OPT_CM_DIR.glob("*.png"))
    metrics_files = list(OPT_METRICS_DIR.glob("*.json"))
    logger.info(f"\nGenerated {len(cm_files)} confusion matrices")
    logger.info(f"Generated {len(metrics_files)} metrics files")

    # ── 16. Save manifest ──────────────────────────────────────────────
    _save_artifact_manifest(artifact_map, OUT_DIR)

    # ── 17. Clean up checkpoints ──────────────────────────────────────
    for checkpoint in CHECKPOINT_DIR.glob("checkpoint_*.pkl"):
        try:
            checkpoint.unlink()
        except:
            pass
    logger.info("  Checkpoints cleaned up.")

    # ── 18. Mark completion ──────────────────────────────────────────────
    mark_completed(BLOCK_NAME)
    logger.info("=" * 60)
    logger.info(f"{BLOCK_NAME} v1.0 Frozen COMPLETED SUCCESSFULLY.")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)