# BLOCK_7_STAT1.py - UPDATED VERSION WITH ALL FIXES
# ================
# FILE PURPOSE:
#     BLOCK 7 — STAT1: Statistical Validation and Reliability Assessment Framework.
#     Validates the statistical significance, reliability, calibration, and reproducibility
#     of all machine learning results obtained throughout the pipeline.
#
#     STATISTICAL MODULES:
#     - STAT-01: Permutation Significance Testing (Across ALL LOSO Folds)
#     - STAT-02: Friedman Ranking Analysis
#     - STAT-03: Nemenyi Post-Hoc Analysis
#     - STAT-04: Wilcoxon Signed-Rank Analysis (with Effect Size + Holm Correction)
#     - STAT-05: Bootstrap Confidence Interval Analysis (Accuracy, F1, MCC, Balanced Acc)
#     - STAT-06: Calibration Assessment
#     - STAT-07: Model Reliability Analysis
#     - STAT-08: Subject-Wise Statistical Analysis (with Confidence Intervals)
#     - STAT-09: LOSO Fold Stability Analysis
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_3/Table_S5_PerFold_LOSO.csv (optional)
#     - outputs/BLOCK_4/Table_S17_SubjectWise_Model_Comparison.csv (optional)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_7/Table9_Statistical_Validation_Summary.csv
#     - outputs/BLOCK_7/Table10_Calibration_Metrics.csv
#     - outputs/BLOCK_7/Table_S39_Complete_Friedman_Rankings.csv
#     - outputs/BLOCK_7/Table_S40_Complete_Nemenyi_Comparisons.csv
#     - outputs/BLOCK_7/Table_S41_Complete_Wilcoxon_Results.csv
#     - outputs/BLOCK_7/Table_S42_Bootstrap_Confidence_Intervals.csv
#     - outputs/BLOCK_7/Table_S43_SubjectWise_Statistical_Summary.csv
#     - outputs/BLOCK_7/Table_S44_Calibration_Statistics_by_Model.csv
#     - outputs/BLOCK_7/Table_S45_LOSO_Fold_Stability_Statistics.csv
#     - outputs/BLOCK_7/Fig37_Permutation_Distribution.png
#     - outputs/BLOCK_7/Fig38_Critical_Difference_Diagram.png
#     - outputs/BLOCK_7/Fig39_Reliability_Diagram.png
#     - outputs/BLOCK_7/Fig40_Calibration_Curve.png
#     - outputs/BLOCK_7/Fig41_Calibration_Histogram.png
#     - outputs/BLOCK_7/FigS28_Bootstrap_Confidence_Intervals.png
#     - outputs/BLOCK_7/FigS29_Model_Rank_Distribution.png
#     - outputs/BLOCK_7/FigS30_Wilcoxon_Significance_Heatmap.png
#     - outputs/BLOCK_7/FigS31_SubjectWise_Statistical_Variability.png
#     - outputs/BLOCK_7/FigS32_Prediction_Confidence_Distribution.png
#     - outputs/BLOCK_7/FigS33_Calibration_Error_Distribution.png
#     - outputs/BLOCK_7/FigS34_ModelWise_ECE_Comparison.png
#     - outputs/BLOCK_7/FigS35_ModelWise_Brier_Comparison.png
#     - outputs/BLOCK_7/FigS36_LOSO_Fold_Accuracy_Distribution.png
#     - outputs/BLOCK_7/FigS37_LOSO_Fold_Variability_Across_Models.png
#     - outputs/BLOCK_7/Statistical_Validation_Report.txt
#     - outputs/BLOCK_7/environment_versions.json
#     - outputs/BLOCK_7/permutation_archive/ (Permutation test results)
#     - outputs/BLOCK_7/bootstrap_archive/ (Bootstrap samples)
#     - outputs/BLOCK_7/wilcoxon_archive/ (Pairwise comparisons)
#     - outputs/BLOCK_7/calibration_archive/ (Calibration metrics per model)
#     - outputs/BLOCK_7/subject_statistics_archive/ (Subject-wise stats)
#     - outputs/BLOCK_7/loso_stability_archive/ (LOSO fold stability)
#     - outputs/BLOCK_7/artifacts.json
#     - state/BLOCK_7_STAT1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_7_STAT1.py

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
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import scipy
from scipy.stats import friedmanchisquare, wilcoxon, bootstrap, ttest_rel, norm, t
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss, f1_score, 
                             matthews_corrcoef, balanced_accuracy_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import label_binarize, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.neighbors import KNeighborsClassifier
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
    RF_PARAMS,
    XGB_PARAMS,
    SVM_PARAMS,
    LDA_PARAMS,
    KNN_PARAMS,
    REQUIRED_CHANNELS,
    EXPECTED_SUBJECTS,  # UPDATE 6: Import N_SUBJECTS
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_feature_matrix, load_csv, save_csv, save_txt, save_fig, save_json,
    load_npy
)

# ── FIX: Calibration reference class explicitly defined ──────────────────────
CALIBRATION_REFERENCE_CLASS = 0
CALIBRATION_REFERENCE_GESTURE = GESTURE_SHORT[CALIBRATION_REFERENCE_CLASS] if GESTURE_SHORT else "HOC"

# ── Try to import scikit-posthocs with graceful fallback ─────────────────────
try:
    import scikit_posthocs as sp
    POSTHOCS_OK = True
    print("scikit-posthocs available for Nemenyi test")
except ImportError:
    POSTHOCS_OK = False
    print("WARNING: scikit-posthocs not installed — Nemenyi test skipped")

# ── Try to import from config with fallbacks ─────────────────────────────────
try:
    from config import CALIB_CV_FOLDS
except ImportError:
    CALIB_CV_FOLDS = 3
    print("WARNING: CALIB_CV_FOLDS not in config — using default 3")

try:
    from config import N_PERMUTATIONS
except ImportError:
    N_PERMUTATIONS = 1000
    print("WARNING: N_PERMUTATIONS not in config — using default 1000")

try:
    from config import N_BOOTSTRAP
except ImportError:
    N_BOOTSTRAP = 1000
    print("WARNING: N_BOOTSTRAP not in config — using default 1000")

try:
    from config import MAX_CALIBRATION_FOLDS
except ImportError:
    MAX_CALIBRATION_FOLDS = 18
    print("WARNING: MAX_CALIBRATION_FOLDS not in config — using default 18")

try:
    from config import VARIANCE_THRESHOLD
except ImportError:
    VARIANCE_THRESHOLD = 1e-4
    print("WARNING: VARIANCE_THRESHOLD not in config — using default 1e-4")

try:
    from config import EXPECTED_SUBJECTS
except ImportError:
    EXPECTED_SUBJECTS = 40
    print("WARNING: EXPECTED_SUBJECTS not in config — using default 40")


# ============================================================================
# UPDATE 1: Centralized model builder (identical to BLOCK_3 and BLOCK_4)
# ============================================================================

def _build_model(model_name: str, use_probability: bool = False):
    """
    UPDATE 1: Centralized model builder using config.py parameters.
    Identical to BLOCK_3_GEN1.py and BLOCK_4_PER1.py.
    
    Args:
        model_name: Name of the model ('LDA', 'KNN', 'SVM', 'RF', 'XGB')
        use_probability: If True, set probability=True for SVM
    
    Returns:
        Instantiated sklearn model
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


# ── Try to import from common with fallback ──────────────────────────────────
try:
    from common import (
        preprocess_emg,
        is_good_window,
        extract_ml_features,
        build_feature_pipeline,
        apply_feature_pipeline,
        apply_smote,
        compute_metrics,
        build_models,
    )
    print("Using centralized feature extraction from common.py")
except ImportError:
    # ── Fallback: Define locally if common.py not available ────────────────
    from scipy.signal import butter, filtfilt
    from scipy import signal as scipy_signal

    print("WARNING: common.py not found - using fallback feature extraction")

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
        """90-feature extraction — identical to SD pipeline."""
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

        assert len(feats) == 90
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
        if min_samples > 6:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=5)
                X, y = sm.fit_resample(X, y)
            except Exception as e:
                print(f"  SMOTE failed {tag}: {e}")
        return X, y

    def compute_metrics(y_true, y_pred):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, matthews_corrcoef, balanced_accuracy_score)
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

    def load_fallback_data():
        """Fallback data loading if BLOCK_2 artifacts not available."""
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

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
np.random.seed(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_7_STAT1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_7")

BLOCK_NAME = "BLOCK_7_STAT1"
OUT_DIR    = BLOCK_DIRS["BLOCK_7"]

# ── Archive directories ──────────────────────────────────────────────────────
PERMUTATION_ARCHIVE = OUT_DIR / "permutation_archive"
BOOTSTRAP_ARCHIVE = OUT_DIR / "bootstrap_archive"
WILCOXON_ARCHIVE = OUT_DIR / "wilcoxon_archive"
CALIBRATION_ARCHIVE = OUT_DIR / "calibration_archive"
SUBJECT_STATS_ARCHIVE = OUT_DIR / "subject_statistics_archive"
LOSO_STABILITY_ARCHIVE = OUT_DIR / "loso_stability_archive"

for d in [PERMUTATION_ARCHIVE, BOOTSTRAP_ARCHIVE, WILCOXON_ARCHIVE, 
          CALIBRATION_ARCHIVE, SUBJECT_STATS_ARCHIVE, LOSO_STABILITY_ARCHIVE]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UPDATE 3: Artifact manifest with verification (identical to BLOCK_4)
# ============================================================================

def _save_artifact_manifest(
    artifact_map: Dict[str, Path],
    out_dir: Path,
) -> None:
    """
    UPDATE 3: Save artifacts.json manifest with file verification.
    Identical to BLOCK_4 artifact verification.
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
        for mf in missing_files[:5]:  # Log first 5 missing
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
# UPDATE 4: Save environment versions (identical to BLOCK_4)
# ============================================================================

def _save_environment_versions():
    """UPDATE 4: Save package versions for reproducibility. Identical to BLOCK_4."""
    env_info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "imblearn": SMOTE.__module__.split('.')[0] if hasattr(SMOTE, '__module__') else "unknown",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    # Try to get shap version if available
    try:
        import shap
        env_info["shap"] = shap.__version__
    except ImportError:
        env_info["shap"] = "not_installed"
    
    # Try to get scikit-posthocs version
    try:
        import scikit_posthocs
        env_info["scikit_posthocs"] = scikit_posthocs.__version__
    except (ImportError, AttributeError):
        env_info["scikit_posthocs"] = "not_installed"
    
    save_json(env_info, OUT_DIR / "environment_versions.json")
    logger.info("  Environment versions saved.")


# ============================================================================
# STAT-01: Permutation Significance Testing (All LOSO Folds)
# ============================================================================

def _permutation_test(X, y, subjects, n_perm: int = N_PERMUTATIONS) -> Dict:
    """
    STAT-01: Perform permutation test across ALL LOSO folds using RF.
    UPDATE 1: Uses centralized _build_model()
    """
    logger.info("  STAT-01: Permutation Significance Testing (All LOSO Folds)...")
    
    loso = LeaveOneGroupOut()
    all_true_accs = []
    all_perm_accs = []
    
    for fold_i, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), 1):
        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_te_raw, y_te = X[te_idx], y[te_idx]
        
        X_sel, sc, sv, sk = build_feature_pipeline(X_tr_raw, y_tr)
        X_te = apply_feature_pipeline(X_te_raw, sc, sv, sk)
        X_sm, y_sm = apply_smote(X_sel, y_tr, tag=f"perm_fold{fold_i}")

        # UPDATE 1: Use centralized model builder
        clf = _build_model("RF")
        clf.fit(X_sm, y_sm)
        true_acc = accuracy_score(y_te, clf.predict(X_te))
        all_true_accs.append(true_acc)

        # Permutations for this fold
        fold_perm_accs = []
        for _ in range(min(n_perm // 10, 100)):
            yp = y_sm.copy()
            np.random.shuffle(yp)
            c2 = _build_model("RF")
            try:
                c2.fit(X_sm, yp)
                fold_perm_accs.append(accuracy_score(y_te, c2.predict(X_te)))
            except Exception:
                fold_perm_accs.append(0.0)
            gc.collect()
        
        all_perm_accs.extend(fold_perm_accs)
        logger.info(f"    Fold {fold_i}: true_acc={true_acc:.4f}, n_perm={len(fold_perm_accs)}")
        gc.collect()
    
    mean_true_acc = np.mean(all_true_accs)
    perm_arr = np.array(all_perm_accs)
    p_val = float(np.mean(perm_arr >= mean_true_acc))
    
    # Save to archive
    perm_df = pd.DataFrame({
        "Fold": range(1, len(all_true_accs) + 1),
        "True_Accuracy": all_true_accs
    })
    perm_df.to_csv(PERMUTATION_ARCHIVE / "true_accuracies_per_fold.csv", index=False)
    
    perm_dist_df = pd.DataFrame({
        "Permutation_Accuracy": perm_arr
    })
    perm_dist_df.to_csv(PERMUTATION_ARCHIVE / "permutation_distribution.csv", index=False)
    logger.info(f"  Permutation archive saved: {len(perm_arr)} samples")

    # Figure 37 – Permutation Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(perm_arr, bins=30, color="#90CAF9", edgecolor="white", label="Null distribution")
    ax.axvline(mean_true_acc, color="red", lw=2.5, ls="--", label=f"Mean True Acc={mean_true_acc:.4f}")
    ax.axvline(np.percentile(perm_arr, 95), color="orange", lw=1.5, ls=":",
               label="95th percentile")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Figure 37 – Permutation Distribution (All Folds, n={len(perm_arr)}) p={p_val:.4f}")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig37_Permutation_Distribution.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info(f"  Figure 37 saved.")

    return {
        "Test": "Permutation (RF, All Folds)",
        "Statistic": round(mean_true_acc, 4),
        "p_value": round(p_val, 4),
        "Significant": "Yes" if p_val < 0.05 else "No"
    }


# ============================================================================
# STAT-02, STAT-03, STAT-04: Friedman + Nemenyi + Wilcoxon
# ============================================================================

def _friedman_nemenyi_wilcoxon(df_loso: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    STAT-02: Friedman Ranking Analysis
    STAT-03: Nemenyi Post-Hoc Analysis
    STAT-04: Wilcoxon Signed-Rank Analysis (with Effect Size + Holm Correction)
    """
    logger.info("  STAT-02 to STAT-04: Friedman + Nemenyi + Wilcoxon Analysis...")
    
    rows: List[Dict] = []
    pivot = df_loso.pivot_table(index="Fold", columns="Model", values="Accuracy")
    pivot_c = pivot.dropna()

    if pivot_c.empty or pivot_c.shape[1] < 2:
        logger.warning("  Insufficient pivot data for Friedman test.")
        return rows, pd.DataFrame(), pd.DataFrame()

    try:
        pivot_c = pivot_c[[m for m in MODEL_NAMES if m in pivot_c.columns]]
        scores = [pivot_c[m].values for m in pivot_c.columns]
        stat, p = friedmanchisquare(*scores)
        rows.append({
            "Test": "Friedman",
            "Statistic": round(stat, 4),
            "p_value": round(p, 6),
            "Significant": "Yes" if p < 0.05 else "No"
        })
        logger.info(f"  Friedman: χ²={stat:.4f}, p={p:.6f}")

        # Table S39: Complete Friedman Rankings
        mean_ranks = pivot_c.rank(axis=1, ascending=False).mean().sort_values()
        rank_df = pd.DataFrame({
            "Model": mean_ranks.index,
            "Mean_Rank": mean_ranks.values,
            "Rank": range(1, len(mean_ranks) + 1)
        })
        rank_df = rank_df.sort_values(["Rank", "Model"], ascending=[True, True])
        save_csv(rank_df, OUT_DIR / "Table_S39_Complete_Friedman_Rankings.csv")
        logger.info("  Table S39 saved.")

        # Nemenyi post-hoc
        if p < 0.05 and POSTHOCS_OK:
            try:
                nem = sp.posthoc_nemenyi_friedman(pivot_c.values)
                nem_df = pd.DataFrame(nem, index=pivot_c.columns, columns=pivot_c.columns)
                save_csv(nem_df, OUT_DIR / "Table_S40_Complete_Nemenyi_Comparisons.csv")
                logger.info("  Table S40 saved (Nemenyi).")
                
                # Figure 38 – Critical Difference Diagram
                _critical_diagram(pivot_c)
            except Exception as e:
                logger.warning(f"  Nemenyi post-hoc failed: {e}")
    except Exception as exc:
        logger.warning(f"  Friedman failed: {exc}")

    # Wilcoxon pairwise comparisons with Effect Size and Holm Correction
    wilcoxon_rows = []
    all_pairs = []
    
    if not pivot_c.empty:
        models = list(pivot_c.columns)
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                m1, m2 = models[i], models[j]
                try:
                    diff = pivot_c[m1].values - pivot_c[m2].values
                    if len(diff) >= 5 and np.any(diff != 0):
                        w, wp = wilcoxon(pivot_c[m1].values, pivot_c[m2].values)
                        n = len(diff)
                        z = (w - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24) if n > 0 else 0
                        effect_size = abs(z) / np.sqrt(n) if n > 0 else 0
                        all_pairs.append({
                            "Model_1": m1,
                            "Model_2": m2,
                            "p_value": wp,
                            "statistic": w,
                            "effect_size": effect_size,
                            "z_score": z
                        })
                except Exception as e:
                    logger.warning(f"  Wilcoxon {m1} vs {m2} failed: {e}")
        
        # Holm-Bonferroni correction
        if all_pairs:
            all_pairs = sorted(all_pairs, key=lambda x: x["p_value"])
            m_holm = len(all_pairs)
            
            for idx, pair in enumerate(all_pairs):
                holm_alpha = 0.05 / (m_holm - idx)
                pair["holm_alpha"] = holm_alpha
                pair["holm_significant"] = pair["p_value"] < holm_alpha
                pair["bonferroni_alpha"] = 0.05 / m_holm
                pair["bonferroni_significant"] = pair["p_value"] < (0.05 / m_holm)
                
                wilcoxon_rows.append({
                    "Comparison": f"{pair['Model_1']} vs {pair['Model_2']}",
                    "Statistic": round(pair["statistic"], 4),
                    "p_value": round(pair["p_value"], 6),
                    "Effect_Size_r": round(pair["effect_size"], 4),
                    "Z_Score": round(pair["z_score"], 4),
                    "Holm_Significant": "Yes" if pair["holm_significant"] else "No",
                    "Bonferroni_Significant": "Yes" if pair["bonferroni_significant"] else "No",
                    "Significant_Original": "Yes" if pair["p_value"] < 0.05 else "No"
                })
                
                rows.append({
                    "Test": f"Wilcoxon {pair['Model_1']} vs {pair['Model_2']}",
                    "Statistic": round(pair["statistic"], 4),
                    "p_value": round(pair["p_value"], 6),
                    "Significant": "Yes" if pair["holm_significant"] else "No"
                })
        
        # Table S41: Complete Wilcoxon Results
        wilcoxon_df = pd.DataFrame(wilcoxon_rows)
        if not wilcoxon_df.empty:
            wilcoxon_df = wilcoxon_df.sort_values(["Comparison"], ascending=[True])
            save_csv(wilcoxon_df, OUT_DIR / "Table_S41_Complete_Wilcoxon_Results.csv")
            logger.info("  Table S41 saved.")
            
            wilcoxon_df.to_csv(WILCOXON_ARCHIVE / "wilcoxon_comparisons.csv", index=False)
            _wilcoxon_heatmap(wilcoxon_df)
    
    return rows, rank_df, wilcoxon_df


def _critical_diagram(pivot_c: pd.DataFrame) -> None:
    """Figure 38 – Critical Difference Diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if not pivot_c.empty:
        ranks = pivot_c.rank(axis=1, ascending=False).mean().sort_values()
        bars = ax.barh(ranks.index, ranks.values,
                       color=sns.color_palette("tab10", len(ranks)))
        ax.set_xlabel("Mean Rank (lower = better)")
        ax.set_title("Figure 38 – Critical Difference Diagram")
        ax.invert_yaxis()
        for bar, v in zip(bars, ranks.values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{v:.2f}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Figure 38 – Critical Difference Diagram")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig38_Critical_Difference_Diagram.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 38 saved.")


def _wilcoxon_heatmap(wilcoxon_df: pd.DataFrame) -> None:
    """Figure S30 – Wilcoxon Significance Heatmap (after Holm correction)."""
    if wilcoxon_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    models = list(set([c.split(" vs ")[0] for c in wilcoxon_df["Comparison"]] + 
                      [c.split(" vs ")[1] for c in wilcoxon_df["Comparison"]]))
    models = sorted(models)
    
    sig_matrix = np.zeros((len(models), len(models)))
    
    for _, row in wilcoxon_df.iterrows():
        m1, m2 = row["Comparison"].split(" vs ")
        i, j = models.index(m1), models.index(m2)
        sig_matrix[i, j] = 1 if row["Holm_Significant"] == "Yes" else 0
        sig_matrix[j, i] = sig_matrix[i, j]
    
    sns.heatmap(sig_matrix, annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=models, yticklabels=models, ax=ax,
                cbar_kws={"label": "Holm-Significant (1=Yes, 0=No)"})
    ax.set_title("Figure S30 – Wilcoxon Significance Heatmap (Holm Correction)")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS30_Wilcoxon_Significance_Heatmap.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S30 saved.")


# ============================================================================
# STAT-05: Bootstrap Confidence Interval Analysis
# ============================================================================

def _bootstrap_analysis(df_loso: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    """
    STAT-05: Bootstrap confidence interval analysis for multiple metrics.
    """
    logger.info("  STAT-05: Bootstrap Confidence Interval Analysis...")
    
    metric_cols = ["Accuracy", "Macro_F1", "MCC", "Balanced_Accuracy"]
    boot_rows = []
    
    for model in MODEL_NAMES:
        model_data = df_loso[df_loso["Model"] == model]
        if len(model_data) < 2:
            continue
        
        row = {"Model": model}
        
        for metric in metric_cols:
            if metric not in model_data.columns:
                continue
                
            values = model_data[metric].values
            
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                boot_means.append(np.mean(sample))
            
            boot_means = np.array(boot_means)
            mean_val = np.mean(boot_means)
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            
            row[f"{metric}_Mean"] = mean_val
            row[f"{metric}_CI_Lower"] = ci_lower
            row[f"{metric}_CI_Upper"] = ci_upper
            row[f"{metric}_CI_Width"] = ci_upper - ci_lower
        
        row["N_Samples"] = len(values)
        row["N_Bootstrap"] = n_bootstrap
        boot_rows.append(row)
    
    df_boot = pd.DataFrame(boot_rows)
    df_boot = df_boot.sort_values(["Model"], ascending=[True])
    save_csv(df_boot, OUT_DIR / "Table_S42_Bootstrap_Confidence_Intervals.csv")
    logger.info("  Table S42 saved.")
    
    df_boot.to_csv(BOOTSTRAP_ARCHIVE / "bootstrap_confidence_intervals.csv", index=False)
    
    # Figure S28 – Bootstrap Confidence Intervals (Accuracy)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(df_boot))
    ax.errorbar(df_boot["Accuracy_Mean"], y_pos, 
                xerr=[df_boot["Accuracy_Mean"] - df_boot["Accuracy_CI_Lower"], 
                      df_boot["Accuracy_CI_Upper"] - df_boot["Accuracy_Mean"]],
                fmt='o', capsize=5, capthick=2, color='#1976D2', elinewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_boot["Model"])
    ax.set_xlabel("Accuracy (95% CI)")
    ax.set_title("Figure S28 – Bootstrap Confidence Intervals (Accuracy)")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS28_Bootstrap_Confidence_Intervals.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S28 saved.")
    
    # Figure S29 – Model Rank Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    rank_data = []
    for model in MODEL_NAMES:
        model_data = df_loso[df_loso["Model"] == model]
        if not model_data.empty and "Accuracy" in model_data.columns:
            rank_data.append(model_data["Accuracy"].values)
    
    if rank_data:
        ax.boxplot(rank_data, labels=MODEL_NAMES, patch_artist=True)
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure S29 – Model Rank Distribution")
        ax.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS29_Model_Rank_Distribution.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S29 saved.")
    
    return df_boot


# ============================================================================
# STAT-06: Calibration Assessment
# ============================================================================

def _calibration_analysis(X, y, subjects) -> Tuple[List[Dict], Dict]:
    """
    STAT-06: Calibration assessment with Sigmoid calibration inside LOSO folds.
    UPDATE 1: Uses centralized _build_model() with probability=True for SVM.
    """
    logger.info("  STAT-06: Calibration Assessment...")
    
    loso = LeaveOneGroupOut()
    rows: List[Dict] = []
    
    all_splits = list(loso.split(X, y, subjects))
    n_splits = min(len(all_splits), MAX_CALIBRATION_FOLDS)
    splits = all_splits[:n_splits]
    logger.info(f"    Using {n_splits} LOSO folds for calibration (max {MAX_CALIBRATION_FOLDS})")
    
    # Per-model calibration metrics
    model_calibration = {m: {"ECE": [], "Brier": [], "LogLoss": []} for m in MODEL_NAMES}
    
    for fi, (tr, te) in enumerate(splits, 1):
        X_sel, sc, sv, sk = build_feature_pipeline(X[tr], y[tr])
        X_te = apply_feature_pipeline(X[te], sc, sv, sk)
        X_sm, y_sm = apply_smote(X_sel, y[tr], tag=f"calib_f{fi}")
        
        for model_name in MODEL_NAMES:
            # UPDATE 1: Use centralized model builder
            base_clf = _build_model(model_name, use_probability=True)
            
            cc = CalibratedClassifierCV(base_clf, method="sigmoid", cv=CALIB_CV_FOLDS)
            
            try:
                cc.fit(X_sm, y_sm)
                proba = cc.predict_proba(X_te)
                y_bin = label_binarize(y[te], classes=list(range(N_CLASSES)))
                
                brier = float(np.mean([
                    brier_score_loss(y_bin[:, c], proba[:, c])
                    for c in range(N_CLASSES)
                ]))
                
                logloss = log_loss(y[te], proba)
                ece = _compute_ece(y[te], proba)
                
                rows.append({
                    "Fold": fi,
                    "Model": model_name,
                    "ECE": round(ece, 4),
                    "Brier": round(brier, 4),
                    "LogLoss": round(logloss, 4)
                })
                
                model_calibration[model_name]["ECE"].append(ece)
                model_calibration[model_name]["Brier"].append(brier)
                model_calibration[model_name]["LogLoss"].append(logloss)
                
                if fi == 1 and model_name == "SVM":
                    _calibration_figures(cc, X_te, y[te])
                    
            except Exception as exc:
                logger.warning(f"  Calibration {model_name} fold {fi}: {exc}")
            gc.collect()
    
    df_calib = pd.DataFrame(rows)
    df_calib = df_calib.sort_values(["Model", "Fold"], ascending=[True, True])
    save_csv(df_calib, OUT_DIR / "Table10_Calibration_Metrics.csv")
    logger.info("  Table 10 saved.")
    
    # Table S44: Calibration Statistics by Model
    cal_stats = []
    for model, metrics in model_calibration.items():
        if metrics["ECE"]:
            cal_stats.append({
                "Model": model,
                "Mean_ECE": np.mean(metrics["ECE"]),
                "Std_ECE": np.std(metrics["ECE"], ddof=1),
                "Mean_Brier": np.mean(metrics["Brier"]),
                "Std_Brier": np.std(metrics["Brier"], ddof=1),
                "Mean_LogLoss": np.mean(metrics["LogLoss"]),
                "Std_LogLoss": np.std(metrics["LogLoss"], ddof=1),
                "N_Folds": len(metrics["ECE"])
            })
    
    df_cal_stats = pd.DataFrame(cal_stats)
    df_cal_stats = df_cal_stats.sort_values(["Model"], ascending=[True])
    save_csv(df_cal_stats, OUT_DIR / "Table_S44_Calibration_Statistics_by_Model.csv")
    logger.info("  Table S44 saved.")
    
    df_calib.to_csv(CALIBRATION_ARCHIVE / "calibration_metrics_all_folds.csv", index=False)
    df_cal_stats.to_csv(CALIBRATION_ARCHIVE / "calibration_statistics_by_model.csv", index=False)
    
    _calibration_error_distribution(df_cal_stats)
    _modelwise_ece_comparison(df_cal_stats)
    _modelwise_brier_comparison(df_cal_stats)
    
    return rows, model_calibration


def _compute_ece(y_true, proba, n_bins=10):
    """Compute Expected Calibration Error."""
    y_pred = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_conf - bin_acc) * np.sum(in_bin)
    
    ece = ece / len(confidences)
    return ece


def _calibration_figures(cc, X_te, y_te) -> None:
    """Figures 39–41: Reliability diagram, calibration curve, histogram."""
    try:
        proba = cc.predict_proba(X_te)
        y_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))

        ref_class = CALIBRATION_REFERENCE_CLASS
        ref_gesture = CALIBRATION_REFERENCE_GESTURE

        prob_ref = proba[:, ref_class]
        fp, mp = calibration_curve(y_bin[:, ref_class], prob_ref, n_bins=10)

        # Figure 39 – Reliability Diagram
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.plot(mp, fp, "s-", color="#1976D2", lw=2, ms=7, label="SVM Sigmoid")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Figure 39 – Reliability Diagram ({ref_gesture} class)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig39_Reliability_Diagram.png")
        # UPDATE 2: Close figure
        plt.close(fig)
        gc.collect()
        logger.info(f"  Figure 39 saved.")

        # Figure 40 – Calibration Curve
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(mp, fp, "o-", color="#E91E63", lw=2)
        ax.fill_between(mp, fp, alpha=0.15, color="#E91E63")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax.set_title(f"Figure 40 – Calibration Curve ({ref_gesture} class)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig40_Calibration_Curve.png")
        # UPDATE 2: Close figure
        plt.close(fig)
        gc.collect()
        logger.info(f"  Figure 40 saved.")

        # Figure 41 – Calibration Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(prob_ref, bins=20, color="#7E57C2", edgecolor="white", alpha=0.85)
        ax.set_xlabel(f"Predicted Probability ({ref_gesture})")
        ax.set_ylabel("Count")
        ax.set_title(f"Figure 41 – Calibration Histogram ({ref_gesture} class)")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig41_Calibration_Histogram.png")
        # UPDATE 2: Close figure
        plt.close(fig)
        gc.collect()
        logger.info(f"  Figure 41 saved.")
    except Exception as exc:
        logger.warning(f"  Calibration figures failed: {exc}")


def _calibration_error_distribution(df_cal_stats: pd.DataFrame) -> None:
    """Figure S33 – Calibration Error Distribution."""
    if df_cal_stats.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_cal_stats))
    width = 0.25
    
    ax.bar(x - width, df_cal_stats["Mean_ECE"], width, label="ECE", color='#4CAF50')
    ax.bar(x, df_cal_stats["Mean_Brier"], width, label="Brier", color='#FF9800')
    ax.bar(x + width, df_cal_stats["Mean_LogLoss"], width, label="LogLoss", color='#2196F3')
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Calibration Error")
    ax.set_title("Figure S33 – Calibration Error Distribution by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(df_cal_stats["Model"])
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS33_Calibration_Error_Distribution.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S33 saved.")


def _modelwise_ece_comparison(df_cal_stats: pd.DataFrame) -> None:
    """Figure S34 – Model-wise ECE Comparison."""
    if df_cal_stats.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_cal_stats["Model"], df_cal_stats["Mean_ECE"], 
           yerr=df_cal_stats["Std_ECE"], capsize=5, color='#4CAF50', edgecolor='black')
    ax.set_xlabel("Model")
    ax.set_ylabel("ECE (Expected Calibration Error)")
    ax.set_title("Figure S34 – Model-wise ECE Comparison")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS34_ModelWise_ECE_Comparison.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S34 saved.")


def _modelwise_brier_comparison(df_cal_stats: pd.DataFrame) -> None:
    """Figure S35 – Model-wise Brier Comparison."""
    if df_cal_stats.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_cal_stats["Model"], df_cal_stats["Mean_Brier"], 
           yerr=df_cal_stats["Std_Brier"], capsize=5, color='#FF9800', edgecolor='black')
    ax.set_xlabel("Model")
    ax.set_ylabel("Brier Score")
    ax.set_title("Figure S35 – Model-wise Brier Comparison")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS35_ModelWise_Brier_Comparison.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S35 saved.")


# ============================================================================
# STAT-07: Model Reliability Analysis
# ============================================================================

def _reliability_analysis(X, y, subjects) -> None:
    """
    STAT-07: Model reliability analysis - prediction confidence distribution.
    UPDATE 1: Uses centralized _build_model()
    """
    logger.info("  STAT-07: Model Reliability Analysis...")
    
    loso = LeaveOneGroupOut()
    tr0, te0 = next(iter(loso.split(X, y, subjects)))
    X_sel, sc, sv, sk = build_feature_pipeline(X[tr0], y[tr0])
    X_te = apply_feature_pipeline(X[te0], sc, sv, sk)
    X_sm, y_sm = apply_smote(X_sel, y[tr0], tag="reliability")
    
    rf = _build_model("RF")
    rf.fit(X_sm, y_sm)
    proba = rf.predict_proba(X_te)
    confidences = np.max(proba, axis=1)
    
    # Figure S32 – Prediction Confidence Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=20, color='#42A5F5', edgecolor='white', alpha=0.7)
    ax.axvline(np.mean(confidences), color='red', linestyle='--', 
               label=f"Mean: {np.mean(confidences):.3f}")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Figure S32 – Prediction Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS32_Prediction_Confidence_Distribution.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S32 saved.")


# ============================================================================
# STAT-08: Subject-Wise Statistical Analysis (with Confidence Intervals)
# ============================================================================

def _subjectwise_statistics(df_subjects: pd.DataFrame) -> pd.DataFrame:
    """
    STAT-08: Subject-wise statistical analysis with confidence intervals.
    UPDATE 7: Uses Student-t confidence intervals instead of 1.96×SEM.
    """
    logger.info("  STAT-08: Subject-Wise Statistical Analysis...")
    
    if df_subjects.empty:
        logger.warning("  No subject-wise data available.")
        return pd.DataFrame()
    
    sub_stats = []
    for sub in df_subjects["Subject"].unique():
        sub_data = df_subjects[df_subjects["Subject"] == sub]
        
        best_row = sub_data.loc[sub_data["Accuracy"].idxmax()]
        
        acc_values = sub_data["Accuracy"].values
        f1_values = sub_data["Macro_F1"].values
        mcc_values = sub_data["MCC"].values
        
        n_models = len(acc_values)
        mean_acc = np.mean(acc_values)
        std_acc = np.std(acc_values, ddof=1)
        
        # UPDATE 7: Student-t confidence intervals
        if n_models > 1:
            se = std_acc / np.sqrt(n_models)
            df_t = n_models - 1
            t_val = t.ppf(0.975, df_t)  # 95% confidence, two-tailed
            ci_lower = mean_acc - t_val * se
            ci_upper = mean_acc + t_val * se
        else:
            ci_lower = mean_acc
            ci_upper = mean_acc
        
        sub_stats.append({
            "Subject": sub,
            "Best_Model": best_row["Model"],
            "Best_Accuracy": best_row["Accuracy"],
            "Mean_Accuracy": mean_acc,
            "Std_Accuracy": std_acc,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "Best_F1": best_row["Macro_F1"],
            "Best_MCC": best_row["MCC"],
            "N_Models": n_models,
        })
    
    df_stats = pd.DataFrame(sub_stats)
    df_stats = df_stats.sort_values(["Best_Accuracy", "Subject"], ascending=[False, True])
    df_stats["Rank"] = range(1, len(df_stats) + 1)
    
    save_csv(df_stats, OUT_DIR / "Table_S43_SubjectWise_Statistical_Summary.csv")
    logger.info("  Table S43 saved.")
    
    df_stats.to_csv(SUBJECT_STATS_ARCHIVE / "subjectwise_statistics.csv", index=False)
    
    # Figure S31 – Subject-Wise Statistical Variability with CI
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_stats) * 0.4)))
    y_pos = range(len(df_stats))
    
    ax.errorbar(df_stats["Mean_Accuracy"], y_pos, 
                xerr=[df_stats["Mean_Accuracy"] - df_stats["CI_Lower"], 
                      df_stats["CI_Upper"] - df_stats["Mean_Accuracy"]],
                fmt='o', capsize=3, capthick=1.5, color='#1976D2', elinewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_stats["Subject"], fontsize=8)
    ax.set_xlabel("Mean Accuracy (95% CI)")
    ax.set_title("Figure S31 – Subject-Wise Statistical Variability")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS31_SubjectWise_Statistical_Variability.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S31 saved.")
    
    return df_stats


# ============================================================================
# STAT-09: LOSO Fold Stability Analysis
# ============================================================================

def _loso_fold_stability_analysis(df_loso: pd.DataFrame) -> pd.DataFrame:
    """
    STAT-09: LOSO Fold Stability Analysis.
    
    Computes:
    - Mean fold accuracy per model
    - Standard deviation
    - Coefficient of variation (%)
    - Min and Max fold accuracy
    - Fold range
    """
    logger.info("  STAT-09: LOSO Fold Stability Analysis...")
    
    if df_loso.empty:
        logger.warning("  No LOSO fold data available for stability analysis.")
        return pd.DataFrame()
    
    stability_rows = []
    
    for model in MODEL_NAMES:
        model_data = df_loso[df_loso["Model"] == model]
        if model_data.empty or "Accuracy" not in model_data.columns:
            continue
        
        accuracies = model_data["Accuracy"].values
        n_folds = len(accuracies)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        cv = (std_acc / (mean_acc + 1e-10)) * 100
        
        stability_rows.append({
            "Model": model,
            "Mean_Accuracy": round(mean_acc, 4),
            "Std_Accuracy": round(std_acc, 4),
            "CV_Percentage": round(cv, 2),
            "Min_Accuracy": round(np.min(accuracies), 4),
            "Max_Accuracy": round(np.max(accuracies), 4),
            "Fold_Range": round(np.max(accuracies) - np.min(accuracies), 4),
            "N_Folds": n_folds,
        })
    
    df_stability = pd.DataFrame(stability_rows)
    df_stability = df_stability.sort_values(["CV_Percentage", "Model"], ascending=[True, True])
    
    # Table S45: LOSO Fold Stability Statistics
    save_csv(df_stability, OUT_DIR / "Table_S45_LOSO_Fold_Stability_Statistics.csv")
    logger.info("  Table S45 saved.")
    
    # Save to archive
    df_stability.to_csv(LOSO_STABILITY_ARCHIVE / "loso_fold_stability.csv", index=False)
    
    # Figure S36 – LOSO Fold Accuracy Distribution (boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_data = []
    for model in MODEL_NAMES:
        model_data = df_loso[df_loso["Model"] == model]
        if not model_data.empty and "Accuracy" in model_data.columns:
            fold_data.append(model_data["Accuracy"].values)
    
    if fold_data:
        bp = ax.boxplot(fold_data, labels=MODEL_NAMES, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_ylabel("Fold Accuracy")
        ax.set_title("Figure S36 – LOSO Fold Accuracy Distribution")
        ax.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS36_LOSO_Fold_Accuracy_Distribution.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S36 saved.")
    
    # Figure S37 – LOSO Fold Variability Across Models
    fig, ax = plt.subplots(figsize=(10, 6))
    if not df_stability.empty:
        bars = ax.bar(df_stability["Model"], df_stability["CV_Percentage"],
                      color=sns.color_palette("Set2", len(df_stability)), edgecolor='black')
        ax.set_xlabel("Model")
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_title("Figure S37 – LOSO Fold Variability Across Models\n(Lower = More Stable)")
        ax.grid(True, alpha=0.2, axis="y")
        
        for bar, v in zip(bars, df_stability["CV_Percentage"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "FigS37_LOSO_Fold_Variability_Across_Models.png")
    # UPDATE 2: Close figure
    plt.close(fig)
    gc.collect()
    logger.info("  Figure S37 saved.")
    
    return df_stability


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_block7() -> Dict:
    """Main entry point for BLOCK 7."""
    
    # UPDATE 5: Runtime logging
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 7 | STAT1: Statistical Validation and Reliability Assessment Framework")
    logger.info("=" * 70)

    # UPDATE 4: Log software versions
    logger.info(f"Python : {sys.version}")
    logger.info(f"Numpy  : {np.__version__}")
    logger.info(f"Pandas : {pd.__version__}")
    logger.info(f"SciPy  : {scipy.__version__}")
    logger.info(f"Sklearn: {sklearn.__version__}")
    logger.info(f"XGBoost: {xgboost.__version__}")
    logger.info(f"PID    : {os.getpid()}")

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        X, y, subjects, _ = load_feature_matrix()
        logger.info(f"  Loaded {len(X)} windows from BLOCK_2 artifacts")
    except (FileNotFoundError, KeyError, TypeError) as e:
        logger.warning(f"  BLOCK_2 artifacts not found: {e}")
        logger.info("  Falling back to raw data loading...")
        X, y, subjects = load_fallback_data()

    # UPDATE 6: Verify subject count
    n_subjects_detected = len(np.unique(subjects))
    if n_subjects_detected != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
            f"but detected {n_subjects_detected} subjects in the data."
        )
    logger.info(f"  Verified {n_subjects_detected} subjects (matches EXPECTED_SUBJECTS)")

    # ── Load LOSO fold results ────────────────────────────────────────────
    try:
        df_loso = load_csv(BLOCK_DIRS["BLOCK_3"] / "Table_S5_PerFold_LOSO.csv")
        logger.info(f"  Loaded LOSO results: {len(df_loso)} rows")
    except Exception as e:
        logger.warning(f"  LOSO fold results not found: {e}")
        df_loso = pd.DataFrame()

    # ── Load subject-wise results ────────────────────────────────────────
    try:
        df_subjects = load_csv(BLOCK_DIRS["BLOCK_4"] / "Table_S17_SubjectWise_Model_Comparison.csv")
        logger.info(f"  Loaded subject-wise results: {len(df_subjects)} rows")
    except Exception as e:
        logger.warning(f"  Subject-wise results not found: {e}")
        df_subjects = pd.DataFrame()

    stat_rows: List[Dict] = []

    # ── STAT-01: Permutation Test (All Folds) ────────────────────────────
    perm_row = _permutation_test(X, y, subjects, n_perm=N_PERMUTATIONS)
    stat_rows.append(perm_row)

    # ── STAT-02 to STAT-04: Friedman + Nemenyi + Wilcoxon ──────────────
    friedman_rows, rank_df, wilcoxon_df = _friedman_nemenyi_wilcoxon(df_loso)
    stat_rows.extend(friedman_rows)

    # ── STAT-05: Bootstrap Confidence Intervals ──────────────────────────
    df_boot = _bootstrap_analysis(df_loso)

    # ── STAT-06: Calibration Assessment ──────────────────────────────────
    calib_rows, model_calibration = _calibration_analysis(X, y, subjects)

    # ── STAT-07: Model Reliability Analysis ──────────────────────────────
    _reliability_analysis(X, y, subjects)

    # ── STAT-08: Subject-Wise Statistical Analysis ──────────────────────
    df_sub_stats = _subjectwise_statistics(df_subjects)

    # ── STAT-09: LOSO Fold Stability Analysis ────────────────────────────
    df_loso_stability = _loso_fold_stability_analysis(df_loso)

    # ── Table 9: Statistical Validation Summary ──────────────────────────
    df_stat = pd.DataFrame(stat_rows)
    df_stat = df_stat.sort_values(["Test"], ascending=[True])
    save_csv(df_stat, OUT_DIR / "Table9_Statistical_Validation_Summary.csv")
    logger.info("  Table 9 saved.")

    # ── Text report ──────────────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        f"{PIPELINE_NAME} | Statistical Validation Report",
        "=" * 70,
        f"Calibration Reference Class: {CALIBRATION_REFERENCE_GESTURE} (Class {CALIBRATION_REFERENCE_CLASS})",
        f"Permutations: {N_PERMUTATIONS}",
        f"Bootstrap Samples: {N_BOOTSTRAP}",
        f"Calibration Folds: {MAX_CALIBRATION_FOLDS}",
        f"Subjects Verified: {n_subjects_detected}",
        "-" * 70,
        "",
        "STATISTICAL TESTS SUMMARY",
        "=" * 70,
        df_stat.to_string(index=False),
        "",
        "=" * 70,
        "BOOTSTRAP CONFIDENCE INTERVALS",
        "=" * 70,
        df_boot.to_string(index=False) if not df_boot.empty else "No data",
        "",
        "=" * 70,
        "WILCOXON EFFECT SIZES",
        "=" * 70,
        wilcoxon_df.to_string(index=False) if not wilcoxon_df.empty else "No data",
        "",
        "=" * 70,
        "LOSO FOLD STABILITY",
        "=" * 70,
        df_loso_stability.to_string(index=False) if not df_loso_stability.empty else "No data",
    ]
    save_txt("\n".join(report_lines), OUT_DIR / "Statistical_Validation_Report.txt")
    logger.info("  Statistical Validation Report saved.")

    # ── UPDATE 4: Save environment versions ──────────────────────────────────
    _save_environment_versions()

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "table9": OUT_DIR / "Table9_Statistical_Validation_Summary.csv",
        "table10": OUT_DIR / "Table10_Calibration_Metrics.csv",
        "table_s39": OUT_DIR / "Table_S39_Complete_Friedman_Rankings.csv",
        "table_s40": OUT_DIR / "Table_S40_Complete_Nemenyi_Comparisons.csv",
        "table_s41": OUT_DIR / "Table_S41_Complete_Wilcoxon_Results.csv",
        "table_s42": OUT_DIR / "Table_S42_Bootstrap_Confidence_Intervals.csv",
        "table_s43": OUT_DIR / "Table_S43_SubjectWise_Statistical_Summary.csv",
        "table_s44": OUT_DIR / "Table_S44_Calibration_Statistics_by_Model.csv",
        "table_s45": OUT_DIR / "Table_S45_LOSO_Fold_Stability_Statistics.csv",
        "fig37": OUT_DIR / "Fig37_Permutation_Distribution.png",
        "fig38": OUT_DIR / "Fig38_Critical_Difference_Diagram.png",
        "fig39": OUT_DIR / "Fig39_Reliability_Diagram.png",
        "fig40": OUT_DIR / "Fig40_Calibration_Curve.png",
        "fig41": OUT_DIR / "Fig41_Calibration_Histogram.png",
        "fig_s28": OUT_DIR / "FigS28_Bootstrap_Confidence_Intervals.png",
        "fig_s29": OUT_DIR / "FigS29_Model_Rank_Distribution.png",
        "fig_s30": OUT_DIR / "FigS30_Wilcoxon_Significance_Heatmap.png",
        "fig_s31": OUT_DIR / "FigS31_SubjectWise_Statistical_Variability.png",
        "fig_s32": OUT_DIR / "FigS32_Prediction_Confidence_Distribution.png",
        "fig_s33": OUT_DIR / "FigS33_Calibration_Error_Distribution.png",
        "fig_s34": OUT_DIR / "FigS34_ModelWise_ECE_Comparison.png",
        "fig_s35": OUT_DIR / "FigS35_ModelWise_Brier_Comparison.png",
        "fig_s36": OUT_DIR / "FigS36_LOSO_Fold_Accuracy_Distribution.png",
        "fig_s37": OUT_DIR / "FigS37_LOSO_Fold_Variability_Across_Models.png",
        "stat_report": OUT_DIR / "Statistical_Validation_Report.txt",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    # UPDATE 5: Runtime logging
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK 7 runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    logger.info("=" * 70)
    logger.info("BLOCK 7 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Tests performed: {len(stat_rows)}")
    logger.info(f"  Calibration metrics: {len(calib_rows)}")
    logger.info(f"  Bootstrap samples: {N_BOOTSTRAP}")
    logger.info(f"  Permutations: {N_PERMUTATIONS}")
    logger.info(f"  Calibration folds: {MAX_CALIBRATION_FOLDS}")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return {
        "n_tests": len(stat_rows),
        "n_calib_folds": len(calib_rows),
        "calibration_reference_class": CALIBRATION_REFERENCE_GESTURE,
        "bootstrap_samples": N_BOOTSTRAP,
        "permutation_samples": N_PERMUTATIONS,
        "calibration_folds": MAX_CALIBRATION_FOLDS,
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
        result = run_block7()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.error(f"❌  {BLOCK_NAME} FAILED: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_7 WITH ALL UPDATES
# ✓ UPDATE 1: Centralized _build_model() using config.py parameters
# ✓ UPDATE 2: plt.close(fig) and gc.collect() after every figure
# ✓ UPDATE 3: Artifact manifest with verification (identical to BLOCK_4)
# ✓ UPDATE 4: Environment versions saved (environment_versions.json)
# ✓ UPDATE 5: Runtime logging (elapsed time, minutes, seconds)
# ✓ UPDATE 6: Subject count verification (40 subjects from config)
# ✓ UPDATE 7: Student-t confidence intervals (95% CI)
# ✓ STAT-01: Permutation Significance Testing
# ✓ STAT-02: Friedman Ranking Analysis
# ✓ STAT-03: Nemenyi Post-Hoc Analysis
# ✓ STAT-04: Wilcoxon Signed-Rank Analysis
# ✓ STAT-05: Bootstrap Confidence Interval Analysis
# ✓ STAT-06: Calibration Assessment
# ✓ STAT-07: Model Reliability Analysis
# ✓ STAT-08: Subject-Wise Statistical Analysis
# ✓ STAT-09: LOSO Fold Stability Analysis
# ✓ Figures 37-41
# ✓ Figures S28-S37
# ✓ Tables 9, 10, S39-S45
# ✓ All archives
# ✓ Artifact manifest
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────
