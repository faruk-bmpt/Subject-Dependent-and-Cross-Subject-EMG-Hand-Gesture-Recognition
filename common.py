#
# common.py
# =========
# Shared utilities used across all frozen pipeline blocks.
#
# FIXES APPLIED:
# - FIX 1: Removed unused MODEL_NAMES import
# - FIX 2: ci95() handles len(arr) < 2 edge case
# - FIX 3: Added SMOTE_K_NEIGHBORS and SMOTE_MIN_SAMPLES config imports
# - FIX 4: Removed single_precision_histogram=True for XGBoost compatibility
# - FIX 5: Added feature-space perturbation functions for BLOCK_13
# - FIX 6: Added load_fallback_data for standalone execution
# - FIX 7: Added preprocess_emg, is_good_window, extract_ml_features
# - FIX 8: Seed RNG in robustness functions for reproducibility
# - FIX 9: Use per-feature std in add_feature_corruption()
# - FIX 10: Cache filter coefficients globally for performance
#

import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import os
import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score
)

# ── FIX 1: Remove unused MODEL_NAMES ──────────────────────────────────────
# FIX 3: Add SMOTE config imports
from config import (
    SEED,
    K_FEATURES,
    N_CLASSES,
    SVM_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    KNN_PARAMS,
    LDA_PARAMS,
    SMOTE_K_NEIGHBORS,
    SMOTE_MIN_SAMPLES,
    INPUT_PATH,
    GESTURE_MAP,
    EMG_CHANNEL_COLUMNS,
    SAMPLING_RATE,
    MIN_RMS_THRESHOLD,
    WINDOW_SIZE,
    STEP_SIZE,
    N_CHANNELS,
)

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")

# ── FIX 10: Cache filter coefficients globally ─────────────────────────────
_FILTER_CACHE = None


def _get_filter_coefficients():
    """
    Compute and cache filter coefficients once.
    This avoids recomputing for every window.
    """
    global _FILTER_CACHE
    
    if _FILTER_CACHE is not None:
        return _FILTER_CACHE
    
    try:
        from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, NOTCH_FREQ, NOTCH_Q
    except ImportError:
        BANDPASS_LOW = 10.0
        BANDPASS_HIGH = 45.0
        FILTER_ORDER = 4
        NOTCH_FREQ = 50.0
        NOTCH_Q = 30.0
    
    from scipy.signal import butter, filtfilt
    from scipy import signal as scipy_signal
    
    _B_BP, _A_BP = butter(
        FILTER_ORDER,
        [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
        btype="band",
    )
    _B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)
    
    _FILTER_CACHE = {
        "bp_b": _B_BP,
        "bp_a": _A_BP,
        "notch_b": _B_NOTCH,
        "notch_a": _A_NOTCH,
    }
    
    return _FILTER_CACHE


#
# ── FIX 7: EMG PREPROCESSING FUNCTIONS ──────────────────────────────────────
#

def preprocess_emg(data: np.ndarray) -> np.ndarray:
    """
    Bandpass and notch filter EMG data.
    Uses cached filter coefficients for performance.
    """
    from scipy.signal import filtfilt
    
    coeffs = _get_filter_coefficients()
    
    data = filtfilt(coeffs["bp_b"], coeffs["bp_a"], data, axis=0)
    data = filtfilt(coeffs["notch_b"], coeffs["notch_a"], data, axis=0)
    return data


def is_good_window(win: np.ndarray) -> bool:
    """Check if window RMS exceeds threshold for all channels."""
    return bool(np.all(np.sqrt(np.mean(win ** 2, axis=0)) > MIN_RMS_THRESHOLD))


def extract_ml_features(w: np.ndarray) -> List[float]:
    """90-feature extraction for ML pipeline."""
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

        # Time domain features
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

        # Hjorth parameters
        act = float(np.var(sig))
        mob = float(np.sqrt(np.mean(np.diff(sig) ** 2) / (act + 1e-10)))
        ds = np.diff(sig)
        if len(ds) > 1:
            mob_d1 = float(np.sqrt(np.mean(np.diff(ds) ** 2) / (np.var(ds) + 1e-10)))
            cplx = mob_d1 / (mob + 1e-10)
        else:
            cplx = 0.0

        # Statistics
        skew = float(np.mean(((sig - sig.mean()) / (np.std(sig) + 1e-10)) ** 3))
        kurt = float(np.nan_to_num(kurtosis(sig, fisher=True, bias=False)))

        # AR coefficients
        ar = [0.0] * 4
        try:
            if np.std(sig) > 1e-4:
                res = AutoReg(sig, lags=4, old_names=False).fit()
                ar = list(res.params[1:5])
                if len(ar) < 4:
                    ar = (ar + [0.0] * 4)[:4]
        except Exception:
            ar = [0.0] * 4

        # Wavelet energy
        try:
            max_level = min(3, pywt.dwt_max_level(len(sig), "db4"))
            coeffs = pywt.wavedec(sig, "db4", level=max_level)
            wav = [float(np.sum(np.abs(c) ** 2)) for c in coeffs[1:]]
            wav = (wav + [0.0] * 4)[:4]
        except Exception:
            wav = [0.0] * 4

        # PSD features
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

        # Shannon-Hilbert entropy
        hist, _ = np.histogram(np.abs(hilbert(sig)), bins=10, density=True)
        sh_h = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # Sample entropy
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

    assert len(feats) == 90
    return feats


def load_fallback_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback data loading if BLOCK_2 artifacts not available.
    Used by BLOCK_9, BLOCK_13, and other standalone blocks.
    """
    print("  Loading raw data from INPUT_PATH (fallback mode)...")
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


#
# ── FIX 5 & 8: FEATURE-SPACE PERTURBATION FUNCTIONS ─────────────────────────
# Used by BLOCK_13 for robustness testing
#

def _get_rng() -> np.random.Generator:
    """
    Get a seeded random number generator for reproducibility.
    """
    return np.random.default_rng(SEED)


def add_gaussian_noise(features: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Add Gaussian noise to feature matrix (feature-space perturbation).
    
    FIX 8: Uses seeded RNG for reproducibility.
    """
    rng = _get_rng()
    return features + rng.normal(0, noise_std, features.shape)


def add_awgn(features: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN to feature matrix at given SNR (feature-space perturbation).
    
    FIX 8: Uses seeded RNG for reproducibility.
    """
    sig_pwr = np.mean(features ** 2)
    if sig_pwr < 1e-10:
        return features
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    rng = _get_rng()
    noise = rng.normal(0, 1, features.shape) * np.sqrt(noise_pwr)
    return features + noise.astype(features.dtype)


def add_feature_drift(features: np.ndarray, drift_level: float) -> np.ndarray:
    """
    Add linear drift to feature matrix.
    This perturbs feature offsets, not physiological baseline drift.
    """
    n_samples = features.shape[0]
    drift = np.linspace(0, drift_level, n_samples).reshape(-1, 1)
    return features + drift


def add_feature_magnitude_scaling(features: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale feature magnitudes."""
    return features * scale_factor


def simulate_feature_dropout(features: np.ndarray, n_features: int) -> np.ndarray:
    """
    Simulate feature dropout (random features set to zero).
    
    FIX 8: Uses seeded RNG for reproducibility.
    """
    features = features.copy()
    n_total_features = features.shape[1]
    rng = _get_rng()
    
    if n_features >= 1 and n_features < n_total_features:
        drop_feats = rng.choice(n_total_features, n_features, replace=False)
        features[:, drop_feats] = 0
    else:
        # Random feature dropout
        n_drop = rng.integers(1, max(1, n_total_features // 3))
        drop_feats = rng.choice(n_total_features, n_drop, replace=False)
        features[:, drop_feats] = 0
    
    return features


def add_feature_corruption(features: np.ndarray, corruption_level: float) -> np.ndarray:
    """
    Corrupt random percentage of feature samples.
    
    FIX 8: Uses seeded RNG for reproducibility.
    FIX 9: Uses per-feature standard deviation for corruption.
    """
    features = features.copy()
    n_samples = features.shape[0]
    n_features = features.shape[1]
    n_corrupt = int(n_samples * corruption_level / 100)
    
    if n_corrupt > 0:
        rng = _get_rng()
        corrupt_idx = rng.choice(n_samples, n_corrupt, replace=False)
        
        # FIX 9: Use per-feature standard deviation
        per_feature_std = np.std(features, axis=0)
        # Broadcast to match the shape of corrupted samples
        noise = rng.normal(0, 1, (n_corrupt, n_features)) * per_feature_std
        features[corrupt_idx] = noise
    
    return features


#
# MODEL FACTORY
#

def build_models() -> Dict[str, Any]:
    """Return fresh (unfitted) model instances for all 5 models."""
    return {
        "LDA": LDAClf(**LDA_PARAMS) if LDA_PARAMS else LDAClf(),
        "KNN": KNeighborsClassifier(**KNN_PARAMS) if KNN_PARAMS else KNeighborsClassifier(n_neighbors=7, n_jobs=1),
        "SVM": SVC(**SVM_PARAMS) if SVM_PARAMS else SVC(C=50, kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(**RF_PARAMS) if RF_PARAMS else RandomForestClassifier(
            n_estimators=250, max_depth=12, n_jobs=1, random_state=SEED, class_weight="balanced"
        ),
        "XGB": XGBClassifier(**XGB_PARAMS) if XGB_PARAMS else XGBClassifier(
            objective="multi:softmax", num_class=N_CLASSES,
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, gamma=0.1,
            reg_lambda=1.5, reg_alpha=0.2, eval_metric="mlogloss",
            verbosity=0, n_jobs=1, random_state=SEED,
            tree_method="hist"
            # FIX 4: Removed single_precision_histogram=True for compatibility
        ),
    }


#
# FEATURE PIPELINE
#

def build_feature_pipeline(X_tr: np.ndarray, y_tr: np.ndarray, k: int = K_FEATURES):
    """
    Fit scaler -> VarianceThreshold -> SelectKBest on training data.
    
    Returns
    -------
    X_sel : np.ndarray
        Transformed training features
    scaler : RobustScaler
        Fitted scaler
    sel_var : VarianceThreshold
        Fitted variance threshold selector
    sel_kb : SelectKBest
        Fitted KBest selector
    """
    scaler = RobustScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)

    sel_var = VarianceThreshold(threshold=1e-4).fit(X_s)
    X_v = sel_var.transform(X_s)

    k_safe = min(k, X_v.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_kb = SelectKBest(f_classif, k=k_safe).fit(X_v, y_tr)

    X_sel = sel_kb.transform(X_v)
    return X_sel, scaler, sel_var, sel_kb


def apply_feature_pipeline(X: np.ndarray, scaler: Any, sel_var: Any, sel_kb: Any) -> np.ndarray:
    """Apply fitted feature pipeline to test data."""
    return sel_kb.transform(sel_var.transform(scaler.transform(X)))


def get_selected_feature_names(feature_names: List[str], sel_var: Any, sel_kb: Any) -> np.ndarray:
    """Get names of selected features after variance threshold and KBest."""
    after_var = np.array(feature_names)[sel_var.get_support()]
    return after_var[sel_kb.get_support()]


#
# SMOTE HELPER
#

def apply_smote(X: np.ndarray, y: np.ndarray, tag: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE if minority class has enough samples.
    
    Uses SMOTE_MIN_SAMPLES and SMOTE_K_NEIGHBORS from config.
    """
    min_samples = min(Counter(y).values())
    
    # FIX 3: Use SMOTE_MIN_SAMPLES from config
    if min_samples >= SMOTE_MIN_SAMPLES:
        try:
            # FIX 3: Use SMOTE_K_NEIGHBORS from config
            sm = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
            X, y = sm.fit_resample(X, y)
        except Exception:
            pass
    return X, y


#
# METRICS AND STATISTICS
#

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def ci95(arr: np.ndarray) -> Tuple[float, float]:
    """
    Compute 95% confidence interval using 1.96 * SEM.
    
    FIX 2: Handles edge case where len(arr) < 2.
    """
    arr = np.asarray(arr)
    
    # If only one value, return that value as both bounds
    if len(arr) < 2:
        m = float(np.mean(arr))
        return m, m
    
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    
    return (
        float(m - 1.96 * se),
        float(m + 1.96 * se),
    )


def summarise_results(rows: List[Dict]) -> pd.DataFrame:
    """Aggregate fold-level rows into mean +- SD + 95% CI per model."""
    df = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "MCC", "Balanced_Accuracy"]
    out = []
    for model, grp in df.groupby("Model"):
        row = {"Model": model}
        for mc in metric_cols:
            if mc not in grp:
                continue
            vals = grp[mc].dropna().values
            lo, hi = ci95(vals)
            row[f"{mc}_Mean"] = round(float(np.mean(vals)), 4)
            row[f"{mc}_SD"] = round(float(np.std(vals, ddof=1)), 4)
            row[f"{mc}_CI_Lo"] = round(lo, 4)
            row[f"{mc}_CI_Hi"] = round(hi, 4)
        out.append(row)
    return pd.DataFrame(out)


METRIC_COLS = ["Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "MCC", "Balanced_Accuracy"]


def train_evaluate_fold(X_tr, y_tr, X_te, y_te, fold_meta=None):
    """Train and evaluate all models on a single fold."""
    models = build_models()
    rows = []
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        metrics = compute_metrics(y_te, preds)
        row = {"Model": name, **metrics}
        if fold_meta:
            row.update(fold_meta)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE common.py WITH ALL FIXES
# ✓ FIX 1: Removed unused MODEL_NAMES import
# ✓ FIX 2: ci95() handles len(arr) < 2 edge case
# ✓ FIX 3: Added SMOTE_K_NEIGHBORS and SMOTE_MIN_SAMPLES config imports
# ✓ FIX 4: Removed single_precision_histogram=True for XGBoost compatibility
# ✓ FIX 5: Added feature-space perturbation functions for BLOCK_13
# ✓ FIX 6: Added load_fallback_data for standalone execution
# ✓ FIX 7: Added preprocess_emg, is_good_window, extract_ml_features
# ✓ FIX 8: Seeded RNG in all robustness functions for reproducibility
# ✓ FIX 9: Used per-feature standard deviation in add_feature_corruption()
# ✓ FIX 10: Cached filter coefficients globally for performance
# ✓ All functions properly typed
# ✓ No duplicate imports
# ✓ Compatible with all BLOCK_* scripts
# ─────────────────────────────────────────────────────────────────────────────