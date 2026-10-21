# BLOCK_2_FE1.py
# ===============
# FILE PURPOSE:
#     BLOCK 2 — FE1: Signal Processing and Feature Engineering Framework.
#     Extracts 90 multi-domain EMG features from raw signals and saves
#     the feature matrix for downstream blocks.
#
#     ORIGINAL 90 FEATURES PRESERVED:
#     - 30 features per channel × 3 channels = 90 features
#     - Time-domain features (MAV, RMS, WL, DASDV, WAMP, LOGVAR, TKEO, IEMG, ZC)
#     - Hjorth parameters (Activity, Mobility, Complexity)
#     - AR coefficients (4 features)
#     - Wavelet energy (4 features)
#     - PSD features (Low, Mid, High, MDF, MNF, Spectral Entropy)
#     - Shannon-Hilbert entropy
#     - Sample entropy
#     - Skewness, Kurtosis
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files)
#     - outputs/BLOCK_0/best_config.json (optional, for optimal WS/OL)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_2/X_features.npy (Feature Matrix - 90 features)
#     - outputs/BLOCK_2/y_labels.npy (Labels)
#     - outputs/BLOCK_2/subjects.npy (Subject IDs)
#     - outputs/BLOCK_2/groups.npy (Group IDs)
#     - outputs/BLOCK_2/feature_names.json (Feature Names)
#     - outputs/BLOCK_2/feature_groups.json (Feature Group Mapping)
#     - outputs/BLOCK_2/feature_metadata.json (Feature Metadata)
#     - outputs/BLOCK_2/feature_statistics.csv (Feature Statistics)
#     - outputs/BLOCK_2/feature_stability.csv (Feature Stability Scores - Diagnostic)
#     - outputs/BLOCK_2/feature_redundancy.csv (Redundant Feature Pairs - Diagnostic)
#     - outputs/BLOCK_2/feature_ranking.csv (Feature Ranking - Diagnostic Only)
#     - outputs/BLOCK_2/subject_separability.csv (Subject Separability Scores - Diagnostic)
#     - outputs/BLOCK_2/Table2_Feature_Library_Summary.csv
#     - outputs/BLOCK_2/Table_S4_Complete_Feature_Dictionary.csv
#     - outputs/BLOCK_2/Table_S16_Feature_Stability_Statistics.csv
#     - outputs/BLOCK_2/Table_S17_Feature_Correlation_Summary.csv
#     - outputs/BLOCK_2/Table_S18_Feature_Robustness_Ranking.csv
#     - outputs/BLOCK_2/Table_S19_Subject_Separability_Scores.csv
#     - outputs/BLOCK_2/Table_S21_Feature_Group_Summary.csv   # NEW
#     - outputs/BLOCK_2/Fig8_Feature_Engineering_Workflow.png
#     - outputs/BLOCK_2/Fig9_Window_Segmentation_Example.png
#     - outputs/BLOCK_2/Fig10_Feature_Category_Distribution.png
#     - outputs/BLOCK_2/Fig11_Feature_Stability_Analysis.png
#     - outputs/BLOCK_2/Fig12_Feature_Correlation_Heatmap.png
#     - outputs/BLOCK_2/Fig13_Mutual_Information_Ranking.png
#     - outputs/BLOCK_2/Fig14_Feature_Redundancy_Analysis.png
#     - outputs/BLOCK_2/Fig15_Subject_Separability_Analysis.png
#     - outputs/BLOCK_2/artifacts.json
#     - state/BLOCK_2_FE1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_2_FE1.py
#

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS",        "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "2")
os.environ.setdefault("MKL_NUM_THREADS",        "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "2")

# ── CRITICAL FIX: Disable Numba BEFORE any imports ──────────────────────────
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

import gc
import glob
import logging
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy import signal as scipy_signal
from scipy.stats import kurtosis
import pywt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

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
    FEATURE_NAMES,
    FEATURE_GROUPS,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FILTER_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
    ARTIFACTS,
    REQUIRED_CHANNELS,
    N_CLASSES,
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    save_csv, save_fig, save_json, save_npy,
    load_json, load_npy
)

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
np.random.seed(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_2_FE1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_2")

BLOCK_NAME = "BLOCK_2_FE1"
OUT_DIR    = BLOCK_DIRS["BLOCK_2"]

# ── Pre-compute filters ──────────────────────────────────────────────────────
_B_BP, _A_BP = butter(
    FILTER_ORDER,
    [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
    btype="band",
)
_B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)


# ============================================================================
# ORIGINAL 90 FEATURE ORDERING
# ============================================================================

# ORIGINAL 30 features per channel (from SD_EMG_7G_WS_OL_Q1_Full_Final_Pipeline_5_ML.py)
PER_CHANNEL_FEATURE_ORDER = [
    # Time domain features (9)
    'MAV', 'RMS', 'WL', 'DASDV', 'WAMP', 'LOGVAR', 'TKEO', 'IEMG', 'ZC',
    # Hjorth parameters (3)
    'ACTIVITY', 'MOBILITY', 'COMPLEXITY',
    # AR coefficients (4)
    'AR1', 'AR2', 'AR3', 'AR4',
    # Wavelet energy (4)
    'WAV1', 'WAV2', 'WAV3', 'WAV4',
    # PSD features (6)
    'PSD_LOW', 'PSD_MID', 'PSD_HIGH', 'MDF', 'MNF', 'SPEC_ENT',
    # Shannon-Hilbert entropy (1)
    'SH_HILB',
    # Sample entropy (1)
    'SAMPEN',
    # Statistics (2)
    'SKEW', 'KURT',
]

# Feature group definitions (matching the 30 features per channel)
FEATURE_GROUP_DEFS = {
    'time_domain': ['MAV', 'RMS', 'WL', 'DASDV', 'WAMP', 'LOGVAR', 'TKEO', 'IEMG', 'ZC'],
    'hjorth': ['ACTIVITY', 'MOBILITY', 'COMPLEXITY'],
    'ar': ['AR1', 'AR2', 'AR3', 'AR4'],
    'wavelet': ['WAV1', 'WAV2', 'WAV3', 'WAV4'],
    'psd': ['PSD_LOW', 'PSD_MID', 'PSD_HIGH', 'MDF', 'MNF', 'SPEC_ENT'],
    'entropy': ['SH_HILB', 'SAMPEN'],
    'statistics': ['SKEW', 'KURT'],
}


# ============================================================================
# SIGNAL PREPROCESSING
# ============================================================================

def preprocess_emg(data: np.ndarray) -> np.ndarray:
    """Band-pass (10–45 Hz) + 50 Hz notch filter."""
    data = filtfilt(_B_BP, _A_BP, data, axis=0)
    data = filtfilt(_B_NOTCH, _A_NOTCH, data, axis=0)
    return data


def is_good_window(win: np.ndarray) -> bool:
    """Reject low-energy windows based on RMS threshold."""
    rms = np.sqrt(np.mean(win ** 2, axis=0))
    return bool(np.all(rms > MIN_RMS_THRESHOLD))


def _detect_emg_columns(df: pd.DataFrame) -> list:
    """Detect EMG columns from DataFrame with multiple strategies."""
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


def sample_entropy(sig: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """Compute sample entropy (SampEn)."""
    sig = sig.astype(np.float64)
    if r is None:
        r = 0.2 * np.std(sig)
    if r < 1e-8:
        return 0.0
    N = len(sig)

    def _phi(ml: int) -> float:
        x = np.array([sig[i: i + ml] for i in range(N - ml + 1)])
        if len(x) < 2:
            return 0.0
        diff = np.abs(x[:, None, :] - x[None, :, :])
        dist = np.max(diff, axis=2)
        count = np.sum(dist <= r) - len(x)
        return count / (len(x) * (len(x) - 1) + 1e-10)

    try:
        p1, p2 = _phi(m), _phi(m + 1)
        if p1 <= 0 or p2 <= 0:
            return 0.0
        return float(-np.log(p2 / p1))
    except Exception:
        return 0.0


# ============================================================================
# ORIGINAL 90-FEATURE EXTRACTION (PRESERVED EXACTLY)
# ============================================================================

def extract_original_features(w: np.ndarray) -> List[float]:
    """
    Extract ORIGINAL 90 features (30 per channel) from a (window_size, 3) array.
    Preserved exactly from SD_EMG_7G_WS_OL_Q1_Full_Final_Pipeline_5_ML.py.
    """
    from scipy.stats import kurtosis
    from scipy.signal import hilbert, welch
    import pywt
    from statsmodels.tsa.ar_model import AutoReg

    feats: List[float] = []
    fs = SAMPLING_RATE
    
    for ch in range(N_CHANNELS):
        sig = w[:, ch].astype(np.float64)
        sig -= np.mean(sig)

        if np.std(sig) < 1e-8:
            feats.extend([0.0] * 30)
            continue

        # ── Time domain features ─────────────────────────────────────────
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

        # ── Hjorth parameters ────────────────────────────────────────────
        act  = float(np.var(sig))
        mob  = float(np.sqrt(np.mean(np.diff(sig) ** 2) / (act + 1e-10)))
        dsig = np.diff(sig)
        if len(dsig) > 1:
            mob_d1 = float(np.sqrt(np.mean(np.diff(dsig) ** 2) / (np.var(dsig) + 1e-10)))
            cplx   = mob_d1 / (mob + 1e-10)
        else:
            cplx = 0.0

        # ── Statistics ────────────────────────────────────────────────────
        skew_val = float(np.mean(((sig - np.mean(sig)) / (np.std(sig) + 1e-10)) ** 3))
        kurt_val = float(np.nan_to_num(kurtosis(sig, fisher=True, bias=False)))

        # ── AR coefficients ──────────────────────────────────────────────
        ar: List[float] = [0.0] * 4
        try:
            if np.std(sig) > 1e-4:
                res = AutoReg(sig, lags=4, old_names=False).fit()
                ar = list(res.params[1:5])
                if len(ar) < 4:
                    ar = (ar + [0.0] * 4)[:4]
        except Exception:
            ar = [0.0] * 4

        # ── Wavelet energy ────────────────────────────────────────────────
        try:
            max_level = min(3, pywt.dwt_max_level(len(sig), "db4"))
            coeffs = pywt.wavedec(sig, "db4", level=max_level)
            wav = [float(np.sum(np.abs(c) ** 2)) for c in coeffs[1:]]
            wav = (wav + [0.0] * 4)[:4]
        except Exception:
            wav = [0.0] * 4

        # ── PSD features ──────────────────────────────────────────────────
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

        # ── Shannon-Hilbert entropy ──────────────────────────────────────
        analytic = hilbert(sig)
        hist, _ = np.histogram(np.abs(analytic), bins=10, density=True)
        sh_hilb = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # ── Sample entropy ───────────────────────────────────────────────
        sampen = sample_entropy(sig)

        # ── Append all 30 features in EXACT order ──────────────────────
        feats.extend([
            mav, rms, wl, dasdv, wamp, logvar, tkeo, iemg, zc,
            act, mob, cplx,
            *ar,
            *wav,
            low, mid, high, mdf_val, mnf_val, spec_entropy,
            sh_hilb, sampen, skew_val, kurt_val,
        ])

    assert len(feats) == 90, f"Feature length mismatch: {len(feats)}"
    return feats


# ============================================================================
# BUILD FEATURE NAME MAPPING
# ============================================================================

def build_feature_names() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build feature names for the original 90 features.
    Uses explicit ordering to ensure names match features exactly.
    """
    channel_labels = [f"CH{ch+1}" for ch in range(N_CHANNELS)]
    
    all_feature_names = []
    feature_groups = {
        'time_domain': [],
        'hjorth': [],
        'ar': [],
        'wavelet': [],
        'psd': [],
        'entropy': [],
        'statistics': [],
    }
    
    # Per-channel features
    for ch_label in channel_labels:
        for group_name, feat_list in FEATURE_GROUP_DEFS.items():
            for feat in feat_list:
                name = f"{ch_label}_{feat}"
                all_feature_names.append(name)
                feature_groups[group_name].append(name)
    
    return all_feature_names, feature_groups


# ============================================================================
# FEATURE LOADING AND PROCESSING
# ============================================================================

def _load_subject_data(
    sub: str,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all gesture files for one subject and extract original 90 features.
    """
    sub_path = INPUT_PATH / sub
    X_list, y_list, g_list = [], [], []

    for gest, lbl in GESTURE_MAP.items():
        files = sorted(glob.glob(str(sub_path / gest / "*.csv")))
        for fi, fpath in enumerate(files):
            try:
                df_raw = pd.read_csv(fpath)
                emg_cols = _detect_emg_columns(df_raw)
                
                if len(emg_cols) < 3:
                    continue
                
                df = df_raw[emg_cols].astype(float)
                data = preprocess_emg(df.values)

                for i in range(0, len(data) - window_size + 1, step_size):
                    win = data[i: i + window_size]
                    if is_good_window(win):
                        X_list.append(extract_original_features(win))
                        y_list.append(lbl)
                        g_list.append(f"{sub}_{gest}_{fi}")
            except Exception as exc:
                logger.warning(f"  Skipping {fpath}: {exc}")

    if not X_list:
        return np.empty((0, 90), dtype=np.float32), np.empty(0, np.int32), np.empty(0)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(g_list),
    )


# ============================================================================
# DIAGNOSTIC MODULES (FE-07, FE-08, FE-09)
# ============================================================================

def analyze_feature_stability(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int = 5
) -> pd.DataFrame:
    """FE-07: Feature Stability Analysis (Diagnostic Only)."""
    logger.info("  FE-07: Feature Stability Analysis (Diagnostic)...")
    
    group_kfold = GroupKFold(n_splits=n_folds)
    feature_stability = []
    
    for fold_i, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, subjects)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        test_std = np.std(X_test, axis=0)
        
        mean_diff = np.abs(train_mean - test_mean) / (train_std + 1e-10)
        std_ratio = train_std / (test_std + 1e-10)
        
        for fi in range(X.shape[1]):
            feature_stability.append({
                'Feature_Index': fi,
                'Fold': fold_i,
                'Mean_Diff': mean_diff[fi],
                'Std_Ratio': std_ratio[fi],
                'Stability_Score': 1 / (1 + mean_diff[fi])
            })
    
    df_stability = pd.DataFrame(feature_stability)
    
    df_summary = df_stability.groupby('Feature_Index').agg({
        'Stability_Score': ['mean', 'std'],
        'Mean_Diff': 'mean',
        'Std_Ratio': 'mean'
    }).reset_index()
    df_summary.columns = ['Feature_Index', 'Stability_Mean', 'Stability_Std', 
                          'Mean_Diff', 'Std_Ratio']
    df_summary = df_summary.sort_values('Stability_Mean', ascending=False)
    
    save_csv(df_summary, OUT_DIR / "feature_stability.csv")
    logger.info(f"  Feature stability analysis complete (Diagnostic Only)")
    
    return df_summary


def analyze_feature_redundancy(
    X: np.ndarray,
    feature_names: List[str],
    y: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """FE-08: Feature Redundancy Analysis (Diagnostic Only)."""
    logger.info("  FE-08: Feature Redundancy Analysis (Diagnostic Only)...")
    
    corr_matrix = np.corrcoef(X.T)
    
    redundant_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            r = corr_matrix[i, j]
            if abs(r) > 0.9:
                redundant_pairs.append({
                    'Feature_1': feature_names[i],
                    'Feature_2': feature_names[j],
                    'Correlation': r,
                    'Abs_Correlation': abs(r)
                })
    
    df_redundant = pd.DataFrame(redundant_pairs).sort_values('Abs_Correlation', ascending=False)
    save_csv(df_redundant, OUT_DIR / "feature_redundancy.csv")
    
    mi_df = pd.DataFrame()
    if y is not None:
        mi_scores = mutual_info_classif(X, y, random_state=SEED)
        mi_df = pd.DataFrame({
            'Feature': feature_names,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        mi_df['Note'] = 'DIAGNOSTIC ONLY - Not used for feature selection'
        save_csv(mi_df, OUT_DIR / "feature_ranking.csv")
    
    logger.info(f"  Found {len(df_redundant)} redundant feature pairs (Diagnostic Only)")
    logger.info("  ⚠ WARNING: Diagnostic analysis only - NOT used for feature selection")
    
    return df_redundant, mi_df


def analyze_subject_separability(
    X: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """FE-09: Subject Separability Analysis (Diagnostic Only)."""
    logger.info("  FE-09: Subject Separability Analysis (Diagnostic Only)...")
    
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        return pd.DataFrame()
    
    from scipy.stats import f_oneway
    
    separability_scores = []
    
    for fi, fname in enumerate(feature_names):
        feature_vals = X[:, fi]
        
        subject_groups = [feature_vals[subjects == s] for s in unique_subjects]
        
        f_stat, p_val = f_oneway(*subject_groups)
        
        overall_mean = np.mean(feature_vals)
        between_var = np.sum([len(g) * (np.mean(g) - overall_mean) ** 2 for g in subject_groups])
        within_var = np.sum([np.sum((g - np.mean(g)) ** 2) for g in subject_groups])
        fisher_score = between_var / (within_var + 1e-10)
        
        separability_scores.append({
            'Feature': fname,
            'Feature_Index': fi,
            'ANOVA_F': f_stat,
            'ANOVA_p': p_val,
            'Fisher_Score': fisher_score,
            'Subject_Separability': fisher_score / (1 + fisher_score),
            'Note': 'DIAGNOSTIC ONLY - Not used for feature selection'
        })
    
    df_separability = pd.DataFrame(separability_scores).sort_values('Fisher_Score', ascending=False)
    save_csv(df_separability, OUT_DIR / "subject_separability.csv")
    
    logger.info("  Subject separability analysis complete (Diagnostic Only)")
    
    return df_separability


def _save_artifact_manifest(artifact_map: Dict[str, Path], out_dir: Path) -> None:
    """Save artifacts.json manifest."""
    manifest = {
        "block": BLOCK_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pipeline": PIPELINE_NAME,
        "artifacts": {name: str(path) for name, path in artifact_map.items()},
    }
    manifest_path = out_dir / "artifacts.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Artifact manifest saved → {manifest_path}")


# ============================================================================
# TABLE S21: FEATURE GROUP SUMMARY
# ============================================================================

def generate_table_s21(
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    X: np.ndarray
) -> pd.DataFrame:
    """
    Generate Table S21: Feature Group Summary.
    Provides comprehensive summary statistics per feature group.
    """
    logger.info("  Generating Table S21: Feature Group Summary...")
    
    group_summary = []
    
    for group_name, feat_list in feature_groups.items():
        # Find indices of features in this group
        indices = [i for i, f in enumerate(feature_names) if f in feat_list]
        
        if not indices:
            continue
        
        # Extract feature values for this group
        group_X = X[:, indices]
        
        # Compute statistics
        group_mean = np.mean(group_X)
        group_std = np.std(group_X)
        group_min = np.min(group_X)
        group_max = np.max(group_X)
        
        # Compute correlation within group (average pairwise correlation)
        if len(indices) > 1:
            corr_matrix = np.corrcoef(group_X.T)
            # Average of upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
        else:
            avg_corr = 1.0
        
        # Compute stability (if available from diagnostic)
        stability_path = OUT_DIR / "feature_stability.csv"
        avg_stability = 0.0
        if stability_path.exists():
            try:
                df_stab = pd.read_csv(stability_path)
                stab_indices = df_stab[df_stab['Feature_Index'].isin(indices)]
                if not stab_indices.empty:
                    avg_stability = stab_indices['Stability_Mean'].mean()
            except Exception:
                pass
        
        group_summary.append({
            'Group': group_name,
            'N_Features': len(indices),
            'Mean_Value': float(group_mean),
            'Std_Value': float(group_std),
            'Min_Value': float(group_min),
            'Max_Value': float(group_max),
            'Avg_Correlation': float(avg_corr),
            'Avg_Stability': float(avg_stability),
            'Example_Feature': feat_list[0] if feat_list else '',
        })
    
    df_s21 = pd.DataFrame(group_summary)
    save_csv(df_s21, OUT_DIR / "Table_S21_Feature_Group_Summary.csv")
    logger.info("  Table S21 saved.")
    
    return df_s21


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def _generate_figures(
    df_group_counts: pd.DataFrame,
    df_stability: Optional[pd.DataFrame] = None,
    corr_matrix: Optional[np.ndarray] = None,
    mi_df: Optional[pd.DataFrame] = None,
    df_separability: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    ws: int = WINDOW_SIZE,
    st: int = STEP_SIZE
) -> None:
    """Generate Figures 8-15."""
    
    # Figure 8 – Feature Engineering Workflow
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    
    modules = {
        "Time Domain\n(MAV,RMS,WL,\nDASDV,WAMP,\nLOGVAR,TKEO,\nIEMG,ZC)": "#E3F2FD",
        "Hjorth/\nStatistics\n(Activity,\nMobility,\nComplexity,\nSkew,Kurt)": "#E8F5E9",
        "AR/Wavelet\n(AR1-4,\nWAV1-4)": "#FFF9C4",
        "Frequency/\nEntropy\n(MDF,MNF,\nPSD_LOW/MID,\nSPEC_ENT,\nSH_HILB,\nSAMPEN)": "#FCE4EC",
    }
    
    xp = np.linspace(0.07, 0.93, len(modules))
    for i, ((label, color), x) in enumerate(zip(modules.items(), xp)):
        ax.add_patch(FancyBboxPatch(
            (x - 0.1, 0.15), 0.2, 0.70,
            facecolor=color, edgecolor="#555", lw=1.5,
            transform=ax.transAxes,
        ))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8,
                transform=ax.transAxes)
    
    ax.text(0.5, 0.95,
            f"Figure 8 – EMG Feature Engineering Workflow ({90} features × {N_CHANNELS} channels)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            transform=ax.transAxes)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig8_Feature_Engineering_Workflow.png")
    logger.info("  Figure 8 saved.")

    # Figure 9 – Window Segmentation Example
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 20 * t) * np.exp(-t / 5) + 0.5 * np.random.randn(1000)
    signal = signal / np.max(np.abs(signal))
    
    ax.plot(t, signal, 'b-', alpha=0.6, linewidth=1.5)
    
    window_size_samples = ws
    step_size_samples = st
    
    for i in range(0, 800, int(step_size_samples / 100 * 100)):
        start = i / 100
        end = (i + window_size_samples) / 100
        if end > 10:
            break
        ax.axvspan(start, end, alpha=0.15, color='#FF6B6B')
        ax.axvline(start, color='#FF6B6B', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Figure 9 – Window Segmentation and Feature Extraction Example\n"
                f"(Window Size = {window_size_samples} samples, Step Size = {step_size_samples} samples)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig9_Window_Segmentation_Example.png")
    logger.info("  Figure 9 saved.")

    # Figure 10 – Feature Category Distribution
    if df_group_counts is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("Set2", len(df_group_counts))
        ax.pie(df_group_counts['Count'], labels=df_group_counts['Group'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f"Figure 10 – Feature Category Distribution\n({df_group_counts['Count'].sum()} Total Features)")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig10_Feature_Category_Distribution.png")
        logger.info("  Figure 10 saved.")

    # Figure 11 – Feature Stability Analysis
    if df_stability is not None and not df_stability.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = df_stability.head(20)
        ax.barh(range(len(top_features)), top_features['Stability_Mean'], 
                color='#4CAF50', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Stability Score (Higher = More Stable)")
        ax.set_ylabel("Feature Index")
        ax.set_title("Figure 11 – Feature Stability Analysis (Diagnostic Only - Top 20)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig11_Feature_Stability_Analysis.png")
        logger.info("  Figure 11 saved.")

    # Figure 12 – Feature Correlation Heatmap
    if corr_matrix is not None and feature_names is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        n_show = min(90, len(feature_names))
        corr_show = corr_matrix[:n_show, :n_show]
        sns.heatmap(corr_show, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=feature_names[:n_show], yticklabels=feature_names[:n_show],
                   ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Figure 12 – Feature Correlation Heatmap (Diagnostic)")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig12_Feature_Correlation_Heatmap.png")
        logger.info("  Figure 12 saved.")

    # Figure 13 – Mutual Information Ranking
    if mi_df is not None and not mi_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_mi = mi_df.head(30)
        ax.barh(range(len(top_mi)), top_mi['Mutual_Information'], 
                color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Mutual Information")
        ax.set_ylabel("Feature")
        ax.set_yticks(range(len(top_mi)))
        ax.set_yticklabels([f[:20] for f in top_mi['Feature']], fontsize=8)
        ax.set_title("Figure 13 – Mutual Information Ranking (Diagnostic Only - Top 30)\n"
                    "NOT Used for Feature Selection")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig13_Mutual_Information_Ranking.png")
        logger.info("  Figure 13 saved.")

    # Figure 14 – Feature Redundancy Analysis (generated separately)
    
    # Figure 15 – Subject Separability Analysis
    if df_separability is not None and not df_separability.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_sep = df_separability.head(20)
        ax.barh(range(len(top_sep)), top_sep['Fisher_Score'], 
                color='#9C27B0', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Fisher Score")
        ax.set_ylabel("Feature")
        ax.set_yticks(range(len(top_sep)))
        ax.set_yticklabels([f[:20] for f in top_sep['Feature']], fontsize=8)
        ax.set_title("Figure 15 – Subject Separability Analysis (Diagnostic Only - Top 20)\n"
                    "NOT Used for Feature Selection")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig15_Subject_Separability_Analysis.png")
        logger.info("  Figure 15 saved.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_fe1() -> Dict:
    """Main entry point for BLOCK 2."""
    logger.info("=" * 70)
    logger.info("BLOCK 2 | FE1: Signal Processing and Feature Engineering Framework")
    logger.info("=" * 70)
    
    # ── Load optimal config if available ──────────────────────────────────
    opt_config_path = BLOCK_DIRS["BLOCK_0"] / "best_config.json"
    ws, st = WINDOW_SIZE, STEP_SIZE
    
    if opt_config_path.exists():
        try:
            opt_config = load_json(opt_config_path)
            ws = opt_config.get("ws", WINDOW_SIZE)
            ol = opt_config.get("overlap", 0.5)
            st = max(1, int(ws * (1 - ol)))
            logger.info(f"  Using optimal config from OPT1: WS={ws}, Step={st}")
        except Exception:
            logger.warning("  Could not load OPT1 config, using defaults")
    
    # ── Build feature names BEFORE processing ──────────────────────────────
    feature_names, feature_groups = build_feature_names()
    EXPECTED_FEATURES = len(feature_names)  # Should be 90
    logger.info(f"  Expected features: {EXPECTED_FEATURES}")
    
    # ── Process all subjects ──────────────────────────────────────────────
    subjects = sorted(
        s for s in os.listdir(INPUT_PATH)
        if os.path.isdir(INPUT_PATH / s)
    )
    logger.info(f"  Processing {len(subjects)} subjects...")
    
    X_all, y_all, sub_all, groups_all = [], [], [], []
    
    for si, sub in enumerate(subjects, 1):
        logger.info(f"  [{si}/{len(subjects)}] Processing {sub}")
        X_s, y_s, g_s = _load_subject_data(sub, ws, st)
        
        if len(X_s) < 20:
            logger.warning(f"    {sub}: too few windows, skipping")
            continue
        
        X_all.append(X_s)
        y_all.append(y_s)
        sub_all.append(np.full(len(y_s), sub))
        groups_all.append(g_s)
        gc.collect()
    
    if not X_all:
        raise RuntimeError("No data loaded from any subject.")
    
    # ── Combine all subjects ──────────────────────────────────────────────
    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int32)
    subjects_arr = np.concatenate(sub_all)
    groups = np.concatenate(groups_all)
    
    logger.info(f"  Total windows: {len(X)}")
    logger.info(f"  Feature dimension: {X.shape[1]}")
    
    # ── Feature Validation ──────────────────────────────────────────────────
    n_features_actual = X.shape[1]
    
    logger.info("=" * 60)
    logger.info("VALIDATING FEATURE MATRIX")
    logger.info("=" * 60)
    logger.info(f"  Expected features: {EXPECTED_FEATURES}")
    logger.info(f"  Actual features:   {n_features_actual}")
    
    assert n_features_actual == EXPECTED_FEATURES, (
        f"❌ FEATURE MISMATCH: X has {n_features_actual}, expected {EXPECTED_FEATURES}"
    )
    logger.info(f"  ✅ Feature validation passed: {n_features_actual} features")
    logger.info("=" * 60)
    
    # ── Feature Statistics ─────────────────────────────────────────────────
    df_stats = pd.DataFrame({
        'Feature_Index': range(X.shape[1]),
        'Feature_Name': feature_names,
        'Mean': np.mean(X, axis=0),
        'Std': np.std(X, axis=0),
        'Min': np.min(X, axis=0),
        'Max': np.max(X, axis=0),
        'Q25': np.percentile(X, 25, axis=0),
        'Q75': np.percentile(X, 75, axis=0)
    })
    save_csv(df_stats, OUT_DIR / "feature_statistics.csv")
    
    # ── Diagnostic Analyses ──────────────────────────────────────────────
    df_stability = analyze_feature_stability(X, y, subjects_arr)
    df_redundant, mi_df = analyze_feature_redundancy(X, feature_names, y)
    df_separability = analyze_subject_separability(X, subjects_arr, feature_names)
    
    # ── Table S21: Feature Group Summary ──────────────────────────────────
    df_s21 = generate_table_s21(feature_names, feature_groups, X)
    
    # ── Save artifacts ──────────────────────────────────────────────────────
    logger.info("  Saving artifacts...")
    
    save_npy(X, ARTIFACTS["X_features_npy"])
    save_npy(y, ARTIFACTS["y_labels_npy"])
    save_npy(subjects_arr, ARTIFACTS["subjects_npy"])
    save_npy(groups, OUT_DIR / "groups.npy")
    save_json(feature_names, ARTIFACTS["feature_names_json"])
    save_json(feature_groups, OUT_DIR / "feature_groups.json")
    
    feature_metadata = {
        "n_features": X.shape[1],
        "n_channels": N_CHANNELS,
        "sampling_rate": SAMPLING_RATE,
        "window_size": ws,
        "step_size": st,
        "feature_names": feature_names,
        "feature_groups": {k: list(v) for k, v in feature_groups.items()},
        "diagnostic_note": "FE-07, FE-08, FE-09 are DIAGNOSTIC ONLY - Not used for feature selection",
        "original_pipeline": "SD_EMG_7G_WS_OL_Q1_Full_Final_Pipeline_5_ML.py"
    }
    save_json(feature_metadata, OUT_DIR / "feature_metadata.json")
    
    # ── Tables ──────────────────────────────────────────────────────────────
    # Table 2 – Feature Library Summary
    df_group_counts = pd.DataFrame([
        {"Group": g, "Count": len(fs), "Example": list(fs)[0] if fs else ""}
        for g, fs in feature_groups.items()
    ])
    save_csv(df_group_counts, OUT_DIR / "Table2_Feature_Library_Summary.csv")
    logger.info("  Table 2 saved.")
    
    # Table S4 – Complete Feature Dictionary
    feat_dict = []
    for fi, fname in enumerate(feature_names):
        grp = "Unknown"
        for g, fs in feature_groups.items():
            if fname in fs:
                grp = g
                break
        feat_dict.append({
            "Index": fi,
            "Feature": fname,
            "Group": grp,
        })
    df_ts4 = pd.DataFrame(feat_dict)
    save_csv(df_ts4, OUT_DIR / "Table_S4_Complete_Feature_Dictionary.csv")
    logger.info("  Table S4 saved.")
    
    # Table S16 – Feature Stability Statistics
    if not df_stability.empty:
        df_ts16 = df_stability.head(50).copy()
        df_ts16['Feature'] = [feature_names[i] if i < len(feature_names) else f"F{i:03d}" 
                             for i in df_ts16['Feature_Index']]
        save_csv(df_ts16, OUT_DIR / "Table_S16_Feature_Stability_Statistics.csv")
        logger.info("  Table S16 saved.")
    
    # Table S17 – Feature Correlation Summary
    if not df_redundant.empty:
        df_ts17 = df_redundant.head(50).copy()
        save_csv(df_ts17, OUT_DIR / "Table_S17_Feature_Correlation_Summary.csv")
        logger.info("  Table S17 saved.")
    
    # Table S18 – Feature Robustness Ranking
    if not mi_df.empty:
        df_ts18 = mi_df.head(50).copy()
        df_ts18['Rank'] = range(1, len(df_ts18) + 1)
        df_ts18['Note'] = 'DIAGNOSTIC ONLY - Not used for feature selection'
        save_csv(df_ts18, OUT_DIR / "Table_S18_Feature_Robustness_Ranking.csv")
        logger.info("  Table S18 saved.")
    
    # Table S19 – Subject Separability Scores
    if not df_separability.empty:
        df_ts19 = df_separability.head(50).copy()
        df_ts19['Rank'] = range(1, len(df_ts19) + 1)
        df_ts19['Note'] = 'DIAGNOSTIC ONLY - Not used for feature selection'
        save_csv(df_ts19, OUT_DIR / "Table_S19_Subject_Separability_Scores.csv")
        logger.info("  Table S19 saved.")
    
    # Table S21 is already saved in generate_table_s21()
    
    # ── Correlation matrix for figures ─────────────────────────────────────
    corr_matrix = np.corrcoef(X.T)
    
    # ── Figures ──────────────────────────────────────────────────────────────
    _generate_figures(
        df_group_counts=df_group_counts,
        df_stability=df_stability,
        corr_matrix=corr_matrix,
        mi_df=mi_df,
        df_separability=df_separability,
        feature_names=feature_names,
        ws=ws,
        st=st
    )
    
    # ── Figure 14 – Feature Redundancy Analysis ──────────────────────────
    if not df_redundant.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_pairs = df_redundant.head(30)
        ax.barh(range(len(top_pairs)), top_pairs['Abs_Correlation'], 
                color='#FF9800', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Absolute Correlation")
        ax.set_ylabel("Feature Pair")
        ax.set_yticks(range(len(top_pairs)))
        ax.set_yticklabels([f"{row['Feature_1'][:12]}×{row['Feature_2'][:12]}" 
                           for _, row in top_pairs.iterrows()], fontsize=7)
        ax.set_title("Figure 14 – Feature Redundancy Analysis (Diagnostic - Top 30 Pairs)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig14_Feature_Redundancy_Analysis.png")
        logger.info("  Figure 14 saved.")
    
    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "X_features": Path(ARTIFACTS["X_features_npy"]),
        "y_labels": Path(ARTIFACTS["y_labels_npy"]),
        "subjects": Path(ARTIFACTS["subjects_npy"]),
        "groups": OUT_DIR / "groups.npy",
        "feature_names": Path(ARTIFACTS["feature_names_json"]),
        "feature_groups": OUT_DIR / "feature_groups.json",
        "feature_metadata": OUT_DIR / "feature_metadata.json",
        "feature_statistics": OUT_DIR / "feature_statistics.csv",
        "feature_stability": OUT_DIR / "feature_stability.csv",
        "feature_redundancy": OUT_DIR / "feature_redundancy.csv",
        "subject_separability": OUT_DIR / "subject_separability.csv",
        "table2": OUT_DIR / "Table2_Feature_Library_Summary.csv",
        "table_s4": OUT_DIR / "Table_S4_Complete_Feature_Dictionary.csv",
        "table_s16": OUT_DIR / "Table_S16_Feature_Stability_Statistics.csv",
        "table_s17": OUT_DIR / "Table_S17_Feature_Correlation_Summary.csv",
        "table_s18": OUT_DIR / "Table_S18_Feature_Robustness_Ranking.csv",
        "table_s19": OUT_DIR / "Table_S19_Subject_Separability_Scores.csv",
        "table_s21": OUT_DIR / "Table_S21_Feature_Group_Summary.csv",
        "fig8": OUT_DIR / "Fig8_Feature_Engineering_Workflow.png",
        "fig9": OUT_DIR / "Fig9_Window_Segmentation_Example.png",
        "fig10": OUT_DIR / "Fig10_Feature_Category_Distribution.png",
        "fig11": OUT_DIR / "Fig11_Feature_Stability_Analysis.png",
        "fig12": OUT_DIR / "Fig12_Feature_Correlation_Heatmap.png",
        "fig13": OUT_DIR / "Fig13_Mutual_Information_Ranking.png",
        "fig14": OUT_DIR / "Fig14_Feature_Redundancy_Analysis.png",
        "fig15": OUT_DIR / "Fig15_Subject_Separability_Analysis.png",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)
    
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total windows: {len(X)}")
    logger.info(f"  Total features: {X.shape[1]} (90 original features)")
    logger.info(f"  Total subjects: {len(np.unique(subjects_arr))}")
    logger.info(f"  Total gestures: {len(GESTURE_MAP)}")
    logger.info(f"  Redundant pairs: {len(df_redundant)}")
    logger.info("")
    logger.info("  Tables Generated:")
    logger.info("    - Table 2: Feature Library Summary")
    logger.info("    - Table S4: Complete Feature Dictionary")
    logger.info("    - Table S16: Feature Stability Statistics")
    logger.info("    - Table S17: Feature Correlation Summary")
    logger.info("    - Table S18: Feature Robustness Ranking")
    logger.info("    - Table S19: Subject Separability Scores")
    logger.info("    - Table S21: Feature Group Summary  (NEW)")
    logger.info("")
    logger.info("  ⚠ DIAGNOSTIC WARNING:")
    logger.info("  FE-07 (Stability), FE-08 (Redundancy), FE-09 (Separability)")
    logger.info("  are DIAGNOSTIC ONLY and NOT used for feature selection")
    logger.info("=" * 60)
    
    return {
        "n_windows": len(X),
        "n_features": X.shape[1],
        "n_subjects": len(np.unique(subjects_arr)),
        "n_gestures": len(GESTURE_MAP),
        "n_redundant_pairs": len(df_redundant),
        "pipeline": PIPELINE_NAME,
    }


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return

    try:
        result = run_fe1()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.error(f"❌  {BLOCK_NAME} FAILED: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_2 WITH TABLE S21
# ✓ ORIGINAL 90 FEATURES PRESERVED (30 per channel × 3 channels)
# ✓ FIX 1: Explicit feature ordering (PER_CHANNEL_FEATURE_ORDER)
# ✓ FIX 2: Feature names match extraction order exactly
# ✓ FIX 3: Feature count validation uses dynamic EXPECTED_FEATURES
# ✓ FIX 4: Table S4 uses correct feature name mapping
# ✓ FIX 5: Real feature names (CH1_MAV, CH1_RMS, etc.)
# ✓ FIX 6: MI and Subject Separability labeled DIAGNOSTIC ONLY
# ✓ FIX 7: Figure 9 uses actual WS/ST from OPT1
# ✓ NEW: Table S21 - Feature Group Summary with statistics
# ✓ No new features added - strictly original 90 features
# ✓ All artifacts saved with real feature names
# ✓ Figures 8-15
# ✓ Tables 2, S4, S16, S17, S18, S19, S21
# ✓ artifacts.json
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────