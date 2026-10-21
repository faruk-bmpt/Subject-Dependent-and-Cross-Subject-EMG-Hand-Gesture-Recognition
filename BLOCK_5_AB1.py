# BLOCK_5_AB1.py - COMPLETE PIPELINE WITH THREE IMPROVEMENTS
# ===============
# FILE PURPOSE:
#     BLOCK 5 — AB1: Feature Contribution and Systematic Ablation Analysis.
#     Evaluates accuracy drop when each feature group is removed (LOGO ablation)
#     and measures incremental accuracy gain as features are added.
#
#     ABLATION MODULES:
#     - AB-01: Leave-One-Group-Out (LOGO) Ablation
#     - AB-02: Feature Group Contribution Analysis
#     - AB-03: Incremental Feature Addition Analysis
#     - AB-04: Feature Count Sensitivity Analysis
#     - AB-05: Per-Group Ablation Impact Ranking
#     - AB-06: Feature Group Complementarity Analysis
#     - AB-07: Model-Wise Feature Analysis
#     - AB-08: Subject-Wise Feature Stability
#     - AB-09: LOSO Fold Stability Analysis
#
#     RESUME SYSTEM (Two-Level):
#     - Analysis-level checkpoints for AB-01 through AB-09
#     - Model-level checkpoints inside AB-07 (LDA, KNN, SVM, RF, XGB)
#     - Checkpoints are removed after successful completion with CSV verification
#     - AB-07 loads existing rows from CSV on resume (preserves all results)
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_2/feature_names.json (optional)
#     - outputs/BLOCK_2/feature_groups.json (optional)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_5/Table6_Feature_Ablation_Results.csv
#     - outputs/BLOCK_5/Table7_Feature_Group_Contribution.csv
#     - outputs/BLOCK_5/Table_S6_Complete_Ablation_Stats.csv
#     - outputs/BLOCK_5/Table_S7_Feature_Group_Ranking.csv
#     - outputs/BLOCK_5/Table_S8_Feature_Count_Sensitivity.csv
#     - outputs/BLOCK_5/Table_S9_Feature_Group_Complementarity.csv
#     - outputs/BLOCK_5/Table_S29_ModelWise_Feature_Ranking.csv
#     - outputs/BLOCK_5/Table_S29_ModelWise_Feature_Ranking_Long.csv
#     - outputs/BLOCK_5/Table_S30_SubjectWise_Feature_Ranking.csv
#     - outputs/BLOCK_5/Table_S31_Subject_Feature_Stability.csv
#     - outputs/BLOCK_5/Table_S32_Fold_Feature_Stability.csv
#     - outputs/BLOCK_5/checkpoints/ (Checkpoint directory - cleaned after success)
#     - outputs/BLOCK_5/environment_versions.json
#     - outputs/BLOCK_5/artifacts.json
#     - state/BLOCK_5_AB1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_5_AB1.py

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
import time
import warnings
import json
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
from matplotlib.patches import FancyBboxPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── FIX: Import from config ──────────────────────────────────────────────────
from config import (
    BLOCK_DIRS,
    INPUT_PATH,
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
    FEATURE_NAMES,
    MODEL_NAMES,
    MODEL_COLORS,
    LDA_PARAMS,
    KNN_PARAMS,
    SVM_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    N_CLASSES,
    EXPECTED_SUBJECTS,
    VARIANCE_THRESHOLD,
    SMOTE_K_NEIGHBORS,
    FEATURE_GROUPS,
)

from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_feature_matrix, load_json, save_csv, save_fig, save_json as save_json_artifact,
    load_npy
)

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")

# FIX: Use deterministic random generator
RNG = np.random.default_rng(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_5_AB1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_5")

BLOCK_NAME = "BLOCK_5_AB1"
OUT_DIR    = BLOCK_DIRS["BLOCK_5"]
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── Try to import from common with fallback ──────────────────────────────────
try:
    from common import (
        preprocess_emg,
        is_good_window,
        extract_ml_features,
        build_feature_pipeline,
        apply_feature_pipeline,
        apply_smote,
        build_models,
        compute_metrics,
        summarise_results,
        get_feature_group,
    )
    logger.info("Using centralized feature extraction from common.py")
except ImportError:
    # ── Fallback: Define locally if common.py not available ────────────────
    from scipy.signal import butter, filtfilt
    from scipy import signal as scipy_signal

    logger.warning("common.py not found - using fallback feature extraction")

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

        if len(feats) != 90:
            raise ValueError(
                f"Expected 90 features, got {len(feats)}. Feature extraction failed."
            )
        return feats

    def apply_smote(X, y, tag=""):
        min_samples = min(Counter(y).values())
        if min_samples > SMOTE_K_NEIGHBORS:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
                X, y = sm.fit_resample(X, y)
            except Exception:
                pass
        return X, y

    def get_feature_group(feature_name):
        """Determine feature group from feature name."""
        groups = {
            "TimeDomain": ["MAV", "RMS", "WL", "DASDV", "WAMP", "LogVar", "TKEO", "IEMG", "ZC", "Hjorth", "AR", "Skew", "Kurt"],
            "FreqDomain": ["PSD", "MDF", "MNF", "Spec_Entropy"],
            "TimeFreq": ["WAV_E"],
            "Complexity": ["Shannon_Hilb_Ent", "SampEn"],
        }
        for group, patterns in groups.items():
            if any(feature_name.startswith(p) for p in patterns):
                return group
        return "Other"

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

        feature_names = [f"feat_{i}" for i in range(len(X_list[0]))]
        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(sub_list),
            feature_names,
        )

# ── END OF FALLBACK IMPORTS ──────────────────────────────────────────────────


# ============================================================================
# TWO-LEVEL RESUME SYSTEM (IMPROVED)
# ============================================================================

def _get_checkpoint_path(ab_module: str) -> Path:
    """Get checkpoint path for a specific AB module."""
    return CHECKPOINT_DIR / f"{ab_module.lower()}_checkpoint.json"


def _save_ab_checkpoint(ab_module: str, data: Dict[str, Any]) -> None:
    """Save checkpoint for an AB module."""
    checkpoint_path = _get_checkpoint_path(ab_module)
    checkpoint_data = {
        "module": ab_module,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"  Checkpoint saved: {ab_module}")
    except Exception as e:
        logger.warning(f"  Failed to save checkpoint for {ab_module}: {e}")


def _load_ab_checkpoint(ab_module: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint for an AB module if it exists."""
    checkpoint_path = _get_checkpoint_path(ab_module)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            logger.info(f"  Loaded checkpoint for {ab_module} from {checkpoint_path}")
            return data.get("data", {})
        except Exception as e:
            logger.warning(f"  Failed to load checkpoint for {ab_module}: {e}")
    return None


def _remove_ab_checkpoint(ab_module: str) -> None:
    """Remove checkpoint for an AB module after successful completion."""
    checkpoint_path = _get_checkpoint_path(ab_module)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug(f"  Checkpoint removed: {ab_module}")
        except Exception as e:
            logger.warning(f"  Failed to remove checkpoint for {ab_module}: {e}")


def _is_ab_completed(ab_module: str, expected_outputs: List[Path]) -> bool:
    """
    IMPROVEMENT 1: Check BOTH checkpoint status AND CSV existence.
    
    Args:
        ab_module: Module name (e.g., "AB01")
        expected_outputs: List of Paths that must exist for completion
    
    Returns:
        True if checkpoint says complete AND all expected outputs exist
    """
    data = _load_ab_checkpoint(ab_module)
    if not data or data.get("status") != "complete":
        return False
    
    # Check that all expected outputs exist
    for output_path in expected_outputs:
        if not output_path.exists():
            logger.warning(
                f"  {ab_module}: checkpoint says complete but {output_path.name} is missing. "
                f"Will re-run."
            )
            return False
    
    return True


def _mark_ab_completed(ab_module: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Mark an AB module as completed by saving a checkpoint with 'complete' status."""
    data = {"status": "complete"}
    if metadata:
        data.update(metadata)
    _save_ab_checkpoint(ab_module, data)


def _is_model_completed(ab_module: str, model_name: str) -> bool:
    """Check if a specific model has been completed inside AB-07."""
    data = _load_ab_checkpoint(ab_module)
    if data and data.get("status") == "complete":
        completed_models = data.get("completed_models", [])
        return model_name in completed_models
    return False


def _mark_model_completed(ab_module: str, model_name: str, result_data: Dict[str, Any]) -> None:
    """
    IMPROVEMENT 2: Store only completed_models in checkpoint, not large result tables.
    Detailed results go to CSV files.
    """
    data = _load_ab_checkpoint(ab_module) or {}
    completed_models = data.get("completed_models", [])
    if model_name not in completed_models:
        completed_models.append(model_name)
    
    # Store only progress tracking, not large result tables
    data["completed_models"] = completed_models
    data["total_models"] = len(MODEL_NAMES)
    
    # Optionally store small metadata like which model was completed
    # but NOT the full modelwise_rows
    _save_ab_checkpoint(ab_module, data)


def _cleanup_checkpoint_after_success(
    ab_module: str, 
    expected_outputs: List[Path],
    should_remove: bool = True
) -> None:
    """
    IMPROVEMENT 3: Remove checkpoint after verifying all outputs exist.
    
    Args:
        ab_module: Module name (e.g., "AB01")
        expected_outputs: List of Paths that must exist
        should_remove: If True, remove checkpoint; if False, keep it
    """
    if not should_remove:
        return
    
    # Verify all outputs exist before removing checkpoint
    all_exist = all(p.exists() for p in expected_outputs)
    
    if all_exist:
        _remove_ab_checkpoint(ab_module)
        logger.debug(f"  {ab_module}: checkpoint removed after successful completion")
    else:
        missing = [p.name for p in expected_outputs if not p.exists()]
        logger.warning(
            f"  {ab_module}: not removing checkpoint - missing outputs: {missing}"
        )


def _cleanup_baseline_checkpoint() -> None:
    """Remove AB00 baseline checkpoint after successful completion."""
    ab00_checkpoint = CHECKPOINT_DIR / "ab00_checkpoint.json"
    if ab00_checkpoint.exists():
        try:
            ab00_checkpoint.unlink()
            logger.debug("  AB00 checkpoint removed after successful completion")
        except Exception as e:
            logger.warning(f"  Failed to remove AB00 checkpoint: {e}")


# ============================================================================
# Save environment versions (identical to other blocks)
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
    
    save_json_artifact(env_info, OUT_DIR / "environment_versions.json")
    logger.info("  Environment versions saved.")


# ============================================================================
# Artifact manifest with verification (identical to other blocks)
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


# =============================================================================
# Internal Model Factory
# =============================================================================

def _build_model(model_type: str):
    """
    Build a classifier using centralized hyperparameters from config.py.
    """

    model_type = model_type.upper()

    if model_type == "LDA":
        return LDAClf(**LDA_PARAMS)

    elif model_type == "KNN":
        return KNeighborsClassifier(**KNN_PARAMS)

    elif model_type == "SVM":
        return SVC(**SVM_PARAMS)

    elif model_type == "RF":
        return RandomForestClassifier(**RF_PARAMS)

    elif model_type == "XGB":
        return XGBClassifier(**XGB_PARAMS)

    raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# LOSO Accuracy (with explicit cleanup)
# =============================================================================

def _loso_model_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feat_mask: np.ndarray,
    model_type: str = "LDA",
    model_cache: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    Compute LOSO accuracy using a selected feature subset.

    Pipeline
    --------
    LOSO
        ↓
    RobustScaler(train)
        ↓
    Transform(train/test)
        ↓
    SMOTE(train only)
        ↓
    Train classifier
        ↓
    Evaluate test subject

    Returns
    -------
    mean_accuracy
    std_accuracy
    """

    if feat_mask.sum() == 0:
        logger.warning("No features selected.")
        return 0.0, 0.0

    X_sub = X[:, feat_mask]

    if X_sub.shape[1] == 0:
        logger.warning("Feature subset is empty.")
        return 0.0, 0.0

    loso = LeaveOneGroupOut()

    accs = []

    for fold, (tr, te) in enumerate(loso.split(X_sub, y, subjects), start=1):

        model = None
        scaler = None
        Xtr = None
        Xte = None
        Xtr_sm = None
        ytr_sm = None
        pred = None

        try:

            scaler = RobustScaler()

            Xtr = scaler.fit_transform(X_sub[tr])
            Xte = scaler.transform(X_sub[te])

            Xtr_sm, ytr_sm = apply_smote(
                Xtr,
                y[tr],
                tag=f"AB_{model_type}_Fold{fold}",
            )

            model = _build_model(model_type)

            model.fit(Xtr_sm, ytr_sm)

            pred = model.predict(Xte)

            acc = accuracy_score(y[te], pred)

            accs.append(acc)

        except Exception as exc:

            logger.warning(
                f"{model_type} failed on LOSO fold {fold}: {exc}"
            )

        finally:
            # Explicit cleanup of all large objects
            try:
                del model
            except (NameError, UnboundLocalError, Exception):
                pass

            try:
                del scaler
            except (NameError, UnboundLocalError, Exception):
                pass

            try:
                del Xtr
                del Xte
            except (NameError, UnboundLocalError, Exception):
                pass

            try:
                del Xtr_sm
                del ytr_sm
            except (NameError, UnboundLocalError, Exception):
                pass

            try:
                del pred
            except (NameError, UnboundLocalError, Exception):
                pass

            gc.collect()

    if len(accs) == 0:

        logger.warning(f"No successful LOSO folds for {model_type}")

        return 0.0, 0.0

    mean_acc = float(np.mean(accs))

    std_acc = (
        float(np.std(accs, ddof=1))
        if len(accs) > 1
        else 0.0
    )

    return mean_acc, std_acc


# =============================================================================
# LDA Wrapper
# =============================================================================

def _loso_lda_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feat_mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Backward-compatible wrapper for legacy code.
    """

    return _loso_model_accuracy(
        X,
        y,
        subjects,
        feat_mask,
        model_type="LDA",
    )


# =============================================================================
# AB-01: Leave-One-Group-Out Ablation (WITH IMPROVED RESUME)
# =============================================================================

def run_ab01_logo(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    base_acc: float,
    base_std: float,
) -> pd.DataFrame:
    """
    AB-01: Leave-One-Group-Out Ablation with improved resume support.
    """
    logger.info("  AB-01: Leave-One-Group-Out Ablation...")
    
    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table7_Feature_Group_Contribution.csv",
        OUT_DIR / "Table6_Feature_Ablation_Results.csv",
    ]
    
    if _is_ab_completed("AB01", expected_outputs):
        logger.info("  AB-01: Already completed, loading from CSV...")
        df_logo = pd.read_csv(OUT_DIR / "Table7_Feature_Group_Contribution.csv")
        return df_logo
    
    # Check for partial progress (in_progress checkpoint)
    checkpoint_data = _load_ab_checkpoint("AB01")
    completed_groups = checkpoint_data.get("completed_groups", []) if checkpoint_data else []
    logo_rows = checkpoint_data.get("logo_rows", []) if checkpoint_data else []
    
    if completed_groups:
        logger.info(f"    Resuming AB-01: {len(completed_groups)} groups already completed")
    
    for grp_name, grp_feats in feature_groups.items():
        # Skip if already completed
        if grp_name in completed_groups:
            logger.debug(f"    Skipping {grp_name} (already completed)")
            continue
        
        logger.info(f"    Processing group: {grp_name}")
        
        mask = np.array([f not in grp_feats for f in feature_names])
        acc, std = _loso_lda_accuracy(X, y, subjects, mask)
        drop = base_acc - acc
        
        logo_rows.append({
            "Dropped_Group": grp_name,
            "N_Features_Used": int(mask.sum()),
            "N_Dropped": int((~mask).sum()),
            "Accuracy_LOGO": round(acc, 4),
            "Accuracy_Std": round(std, 4),
            "Accuracy_Drop": round(drop, 4),
            "Drop_Percentage": round(drop / (base_acc + 1e-10) * 100, 2),
        })
        
        logger.info(f"    Drop {grp_name:15s}: acc={acc:.4f}  drop={drop:+.4f}")
        
        # Save checkpoint after each group
        completed_groups.append(grp_name)
        _save_ab_checkpoint("AB01", {
            "status": "in_progress",
            "completed_groups": completed_groups,
            "logo_rows": logo_rows,
            "total_groups": len(feature_groups),
        })
    
    df_logo = pd.DataFrame(logo_rows)
    df_logo = df_logo.sort_values(["Accuracy_Drop", "Dropped_Group"], ascending=[False, True])
    
    # Save CSV before marking complete
    save_csv(df_logo, OUT_DIR / "Table7_Feature_Group_Contribution.csv")
    
    # IMPROVEMENT 3: Mark complete then remove checkpoint after verifying CSV
    _mark_ab_completed("AB01", {
        "n_groups": len(feature_groups),
        "n_rows": len(logo_rows),
    })
    
    # Remove checkpoint after verifying outputs exist
    _cleanup_checkpoint_after_success("AB01", expected_outputs, should_remove=True)
    
    return df_logo


# =============================================================================
# AB-03: Incremental Feature Addition (WITH IMPROVED RESUME)
# =============================================================================

def run_ab03_incremental(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    k_values: List[int],
) -> Tuple[List[float], List[float], int, float]:
    """
    AB-03: Incremental Feature Addition with improved resume support.
    """
    logger.info("  AB-03: Incremental Feature Addition...")
    
    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table_S8_Feature_Count_Sensitivity.csv",
    ]
    
    if _is_ab_completed("AB03", expected_outputs):
        logger.info("  AB-03: Already completed, loading from CSV...")
        df_fc = pd.read_csv(OUT_DIR / "Table_S8_Feature_Count_Sensitivity.csv")
        # Extract results from CSV
        incr_accs = df_fc["Accuracy"].tolist()
        incr_stds = df_fc["Std"].tolist()
        best_k = k_values[np.argmax(incr_accs)] if incr_accs else 0
        best_acc = max(incr_accs) if incr_accs else 0.0
        return incr_accs, incr_stds, best_k, best_acc
    
    # Check for partial progress
    checkpoint_data = _load_ab_checkpoint("AB03")
    completed_k = checkpoint_data.get("completed_k", []) if checkpoint_data else []
    incr_accs = checkpoint_data.get("incr_accs", []) if checkpoint_data else []
    incr_stds = checkpoint_data.get("incr_stds", []) if checkpoint_data else []
    
    if completed_k:
        logger.info(f"    Resuming AB-03: {len(completed_k)} K values already completed")
    
    # Fit SelectKBest on first LOSO training fold
    loso_splits = list(LeaveOneGroupOut().split(X, y, subjects))
    tr0, _ = loso_splits[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_ref = SelectKBest(f_classif, k=min(K_FEATURES, X.shape[1]))
        sel_ref.fit(X[tr0], y[tr0])
        feature_scores = np.nan_to_num(
            sel_ref.scores_,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rank_idx = np.argsort(-feature_scores)
    
    for k in k_values:
        # Skip if already completed
        if k in completed_k:
            logger.debug(f"    Skipping K={k} (already completed)")
            continue
        
        logger.info(f"    Processing K={k}")
        
        mask_k = np.zeros(len(feature_names), dtype=bool)
        mask_k[rank_idx[:k]] = True
        acc_k, std_k = _loso_lda_accuracy(X, y, subjects, mask_k)
        incr_accs.append(acc_k)
        incr_stds.append(std_k)
        
        logger.info(f"    K={k:3d}: acc={acc_k:.4f}")
        
        # Save checkpoint after each K value
        completed_k.append(k)
        _save_ab_checkpoint("AB03", {
            "status": "in_progress",
            "completed_k": completed_k,
            "incr_accs": incr_accs,
            "incr_stds": incr_stds,
            "total_k": len(k_values),
        })
    
    # Table S8: Feature Count Sensitivity
    fc_rows = []
    for i, k in enumerate(k_values):
        fc_rows.append({
            "N_Features": k,
            "Accuracy": round(incr_accs[i], 4),
            "Std": round(incr_stds[i], 4),
            "Gain_from_Previous": round(incr_accs[i] - (incr_accs[i-1] if i > 0 else 0), 4),
            "Cumulative_Percentage": round(incr_accs[i] / (incr_accs[-1] + 1e-10) * 100, 2),
        })
    df_fc = pd.DataFrame(fc_rows)
    save_csv(df_fc, OUT_DIR / "Table_S8_Feature_Count_Sensitivity.csv")
    
    best_k = k_values[np.argmax(incr_accs)] if incr_accs else 0
    best_acc = max(incr_accs) if incr_accs else 0.0
    
    # Mark complete and remove checkpoint
    _mark_ab_completed("AB03", {
        "n_k_values": len(k_values),
        "best_k": best_k,
        "best_acc": best_acc,
    })
    _cleanup_checkpoint_after_success("AB03", expected_outputs, should_remove=True)
    
    return incr_accs, incr_stds, best_k, best_acc


# =============================================================================
# AB-06: Feature Group Complementarity (WITH IMPROVED RESUME)
# =============================================================================

def _compute_group_complementarity(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    base_mask: np.ndarray,
    base_acc: float,
) -> pd.DataFrame:
    """
    AB-06: Feature Group Complementarity with improved resume support.
    """
    logger.info("  AB-06: Feature Group Complementarity Analysis...")

    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table_S9_Feature_Group_Complementarity.csv",
    ]
    
    if _is_ab_completed("AB06", expected_outputs):
        logger.info("  AB-06: Already completed, loading from CSV...")
        df = pd.read_csv(OUT_DIR / "Table_S9_Feature_Group_Complementarity.csv")
        return df

    feature_names = np.asarray(feature_names)

    group_masks = {}

    for group_name, group_features in feature_groups.items():
        group_masks[group_name] = np.isin(feature_names, group_features)

    # Check for partial progress
    checkpoint_data = _load_ab_checkpoint("AB06")
    completed_groups = checkpoint_data.get("completed_groups", []) if checkpoint_data else []
    rows = checkpoint_data.get("rows", []) if checkpoint_data else []

    if completed_groups:
        logger.info(f"    Resuming AB-06: {len(completed_groups)} groups already completed")

    for group_name, mask in group_masks.items():
        # Skip if already completed
        if group_name in completed_groups:
            logger.debug(f"    Skipping {group_name} (already completed)")
            continue

        if mask.sum() == 0:
            logger.warning(f"{group_name}: contains no features.")
            continue

        logger.info(f"    Processing group: {group_name}")

        complement_mask = base_mask & (~mask)

        acc_without, std_without = _loso_lda_accuracy(
            X, y, subjects, complement_mask,
        )

        acc_only, std_only = _loso_lda_accuracy(
            X, y, subjects, mask,
        )

        rows.append({
            "Group": group_name,
            "N_Features": int(mask.sum()),
            "Only_This_Group_Acc": round(acc_only, 4),
            "Only_This_Group_STD": round(std_only, 4),
            "All_Except_This_Acc": round(acc_without, 4),
            "All_Except_This_STD": round(std_without, 4),
            "Complementarity_Drop": round(base_acc - acc_without, 4),
            "Unique_Contribution": round(acc_only / max(base_acc, 1e-10), 4),
        })

        # Save checkpoint after each group
        completed_groups.append(group_name)
        _save_ab_checkpoint("AB06", {
            "status": "in_progress",
            "completed_groups": completed_groups,
            "rows": rows,
            "total_groups": len(group_masks),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df.sort_values("Complementarity_Drop", ascending=False, inplace=True)

    save_csv(df, OUT_DIR / "Table_S9_Feature_Group_Complementarity.csv")
    logger.info(f"  Table S9 saved ({len(df)} feature groups).")

    # Mark complete and remove checkpoint
    _mark_ab_completed("AB06", {
        "n_groups": len(group_masks),
        "n_rows": len(rows),
    })
    _cleanup_checkpoint_after_success("AB06", expected_outputs, should_remove=True)

    return df


# ============================================================================
# AB-07: Model-Wise Feature Analysis (WITH COMPLETE RESUME FIX)
# ============================================================================

def compute_modelwise_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    AB-07: Compute feature importance rankings for all 5 models.
    
    TWO-LEVEL RESUME:
    - Analysis-level: AB-07 checkpoint tracks which models are completed
    - Model-level: Each model's results are saved independently
    - IMPROVEMENT 2: Checkpoint stores only completed_models, not large result tables
    - FIX: When resuming, load existing rows from CSV to preserve all results
    """
    logger.info("  AB-07: Model-Wise Feature Analysis...")
    
    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table_S29_ModelWise_Feature_Ranking.csv",
    ]
    
    if _is_ab_completed("AB07", expected_outputs):
        logger.info("  AB-07: Already completed, loading from CSV...")
        df_pivot = pd.read_csv(OUT_DIR / "Table_S29_ModelWise_Feature_Ranking.csv")
        # Also load the long format if needed
        df_modelwise = pd.read_csv(OUT_DIR / "Table_S29_ModelWise_Feature_Ranking_Long.csv") \
            if (OUT_DIR / "Table_S29_ModelWise_Feature_Ranking_Long.csv").exists() else pd.DataFrame()
        return df_pivot, df_modelwise
    
    # Check for partial progress (only completed_models list, not full results)
    checkpoint_data = _load_ab_checkpoint("AB07")
    completed_models = checkpoint_data.get("completed_models", []) if checkpoint_data else []
    
    # ================================================================
    # FIX: Load existing results from CSV if it exists
    # ================================================================
    modelwise_rows = []
    long_csv_path = OUT_DIR / "Table_S29_ModelWise_Feature_Ranking_Long.csv"
    
    if long_csv_path.exists():
        try:
            existing_df = pd.read_csv(long_csv_path)
            # Only keep rows for models that are marked as completed
            if not existing_df.empty and "Model" in existing_df.columns:
                existing_rows = existing_df[existing_df["Model"].isin(completed_models)]
                modelwise_rows = existing_rows.to_dict("records")
                logger.info(f"    Loaded {len(modelwise_rows)} existing rows from CSV")
        except Exception as e:
            logger.warning(f"    Could not load existing rows from CSV: {e}")
    
    if completed_models:
        logger.info(f"    Resuming AB-07: {len(completed_models)} models already completed")
        logger.info(f"    Completed models: {completed_models}")
    
    for model_name in MODEL_NAMES:
        # SKIP if model already completed
        if model_name in completed_models:
            logger.info(f"    Skipping {model_name} (already completed)")
            continue
        
        logger.info("=" * 60)
        logger.info(f"    Running model: {model_name}")
        logger.info("=" * 60)
        
        # Compute baseline for this model
        all_mask = np.ones(len(feature_names), dtype=bool)
        base_acc, base_std = _loso_model_accuracy(X, y, subjects, all_mask, model_name)
        logger.info(f"      Baseline accuracy: {base_acc:.4f} ± {base_std:.4f}")
        
        # Compute LOGO drops for each group
        for grp_name, grp_feats in feature_groups.items():
            mask = ~np.isin(feature_names, grp_feats)
            acc, std = _loso_model_accuracy(X, y, subjects, mask, model_name)
            drop = base_acc - acc
            logger.info(f"      Drop {grp_name}: {drop:+.4f} (acc={acc:.4f})")
            
            modelwise_rows.append({
                "Model": model_name,
                "Feature_Group": grp_name,
                "Baseline_Accuracy": round(base_acc, 4),
                "LOGO_Accuracy": round(acc, 4),
                "Accuracy_Drop": round(drop, 4),
                "Drop_Percentage": round(drop / (base_acc + 1e-10) * 100, 2),
            })
        
        # Save intermediate results to CSV after each model
        df_intermediate = pd.DataFrame(modelwise_rows)
        save_csv(df_intermediate, long_csv_path)
        logger.info(f"    Intermediate results saved to {long_csv_path.name}")
        
        # Mark model as completed (IMPROVEMENT 2: store only completed_models)
        _mark_model_completed("AB07", model_name, {
            "baseline_acc": base_acc,
            "baseline_std": base_std,
        })
        completed_models.append(model_name)
        
        # Save checkpoint after each model (only progress tracking)
        _save_ab_checkpoint("AB07", {
            "status": "in_progress",
            "completed_models": completed_models,
            "total_models": len(MODEL_NAMES),
        })
    
    df_modelwise = pd.DataFrame(modelwise_rows)
    
    # Pivot for table format
    df_pivot = df_modelwise.pivot_table(
        index="Feature_Group", 
        columns="Model", 
        values="Accuracy_Drop"
    ).reset_index()
    df_pivot.columns = ["Feature_Group"] + [f"{m}_Drop" for m in MODEL_NAMES]
    
    # Add average drop across models
    drop_cols = [f"{m}_Drop" for m in MODEL_NAMES]
    df_pivot["Avg_Drop"] = df_pivot[drop_cols].mean(axis=1)
    df_pivot = df_pivot.sort_values(["Avg_Drop", "Feature_Group"], ascending=[False, True])
    
    save_csv(df_pivot, OUT_DIR / "Table_S29_ModelWise_Feature_Ranking.csv")
    save_csv(df_modelwise, long_csv_path)
    logger.info(f"  Table S29 saved: {len(df_pivot)} groups × {len(MODEL_NAMES)} models")
    
    # Mark complete and remove checkpoint
    _mark_ab_completed("AB07", {
        "n_models": len(MODEL_NAMES),
        "n_rows": len(modelwise_rows),
    })
    _cleanup_checkpoint_after_success("AB07", expected_outputs, should_remove=True)
    
    return df_pivot, df_modelwise


# ============================================================================
# AB-08: Subject-Wise Feature Stability (WITH IMPROVED RESUME)
# ============================================================================

def compute_subjectwise_feature_stability(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    AB-08: Compute feature stability across subjects with improved resume support.
    
    Saves to Table_S31_Subject_Feature_Stability.csv
    """
    logger.info("  AB-08: Subject-Wise Feature Stability Analysis...")
    
    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table_S31_Subject_Feature_Stability.csv",
        OUT_DIR / "Table_S30_SubjectWise_Feature_Ranking.csv",
    ]
    
    if _is_ab_completed("AB08", expected_outputs):
        logger.info("  AB-08: Already completed, loading from CSV...")
        df_stability = pd.read_csv(OUT_DIR / "Table_S31_Subject_Feature_Stability.csv")
        return df_stability
    
    unique_subs = np.unique(subjects)
    
    # Check for existing checkpoint
    checkpoint_data = _load_ab_checkpoint("AB08")
    completed_subjects = checkpoint_data.get("completed_subjects", []) if checkpoint_data else []
    subject_importance = checkpoint_data.get("subject_importance", {}) if checkpoint_data else {}
    
    if completed_subjects:
        logger.info(f"    Resuming AB-08: {len(completed_subjects)} subjects already completed")
    
    # For each subject, compute feature importance using SelectKBest
    for sub in unique_subs:
        # Skip if already completed
        if sub in completed_subjects:
            logger.debug(f"    Skipping subject {sub} (already completed)")
            continue

        sub_mask = (subjects == sub)

        X_sub = X[sub_mask]
        y_sub = y[sub_mask]

        # Need enough samples and at least two classes
        if len(X_sub) < 20 or len(np.unique(y_sub)) < 2:
            logger.warning(f"Skipping subject {sub}: insufficient samples or classes.")
            completed_subjects.append(sub)
            continue

        try:
            # Use the same preprocessing as the main pipeline
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_sub)

            selector = SelectKBest(
                score_func=f_classif,
                k=min(K_FEATURES, X_scaled.shape[1]),
            )

            selector.fit(X_scaled, y_sub)

            # Replace NaN/Inf scores with zero
            scores = np.nan_to_num(
                selector.scores_,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            subject_importance[sub] = scores.tolist()  # Store as list for JSON

        except Exception as exc:
            logger.warning(
                f"Subject-wise stability failed for subject {sub}: {exc}"
            )
            completed_subjects.append(sub)
            continue
        
        completed_subjects.append(sub)
        
        # Save checkpoint after each subject
        _save_ab_checkpoint("AB08", {
            "status": "in_progress",
            "completed_subjects": completed_subjects,
            "subject_importance": subject_importance,
            "total_subjects": len(unique_subs),
        })

    if not subject_importance:
        logger.warning("No valid subject-wise feature importance could be computed.")
        return pd.DataFrame()
    
    # Compute stability metrics across subjects
    stability_rows = []
    for fi, fname in enumerate(feature_names):
        scores = []
        for sub, imp in subject_importance.items():
            if fi < len(imp):
                scores.append(imp[fi])
        
        if scores:
            stability_rows.append({
                "Feature_Index": fi,
                "Feature_Name": fname,
                "Mean_Importance": np.mean(scores),
                "Std_Importance": np.std(scores, ddof=1),
                "CV_Importance": np.std(scores, ddof=1) / (np.mean(scores) + 1e-10),
                "Stability_Score": 1 / (1 + np.std(scores, ddof=1) / (np.mean(scores) + 1e-10)),
                "N_Subjects": len(scores),
            })
    
    df_stability = pd.DataFrame(stability_rows)
    df_stability = df_stability.sort_values(["Stability_Score", "Feature_Name"], ascending=[False, True])
    
    # Save to Table_S31_Subject_Feature_Stability.csv
    save_csv(df_stability, OUT_DIR / "Table_S31_Subject_Feature_Stability.csv")
    logger.info(f"  Table S31 saved: {len(df_stability)} features")
    
    # Also create Table S30: Subject-wise ranking (top features per subject)
    ranking_rows = []
    for sub, imp in subject_importance.items():
        # Get top 10 features
        top_idx = np.argsort(imp)[-10:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            if idx < len(feature_names):
                ranking_rows.append({
                    "Subject": sub,
                    "Rank": rank,
                    "Feature_Index": idx,
                    "Feature_Name": feature_names[idx] if idx < len(feature_names) else f"F{idx}",
                    "Importance_Score": imp[idx],
                })
    
    df_ranking = pd.DataFrame(ranking_rows)
    if not df_ranking.empty:
        df_ranking = df_ranking.sort_values(["Subject", "Rank"], ascending=[True, True])
        save_csv(df_ranking, OUT_DIR / "Table_S30_SubjectWise_Feature_Ranking.csv")
        logger.info(f"  Table S30 saved: {len(df_ranking)} rows")
    
    # Mark complete and remove checkpoint
    _mark_ab_completed("AB08", {
        "n_subjects": len(unique_subs),
        "n_features": len(feature_names),
    })
    _cleanup_checkpoint_after_success("AB08", expected_outputs, should_remove=True)
    
    return df_stability


# ============================================================================
# AB-09: LOSO Fold Stability Analysis (WITH IMPROVED RESUME)
# ============================================================================

def compute_fold_stability(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    AB-09: Analyze feature stability across LOSO folds with improved resume support.
    
    Saves to Table_S32_Fold_Feature_Stability.csv
    """
    logger.info("  AB-09: LOSO Fold Stability Analysis...")
    
    # IMPROVEMENT 1: Check BOTH checkpoint AND CSV existence
    expected_outputs = [
        OUT_DIR / "Table_S32_Fold_Feature_Stability.csv",
    ]
    
    if _is_ab_completed("AB09", expected_outputs):
        logger.info("  AB-09: Already completed, loading from CSV...")
        df_fold_stability = pd.read_csv(OUT_DIR / "Table_S32_Fold_Feature_Stability.csv")
        return df_fold_stability
    
    loso = LeaveOneGroupOut()
    
    # Check for existing checkpoint
    checkpoint_data = _load_ab_checkpoint("AB09")
    completed_folds = checkpoint_data.get("completed_folds", []) if checkpoint_data else []
    fold_importances = checkpoint_data.get("fold_importances", []) if checkpoint_data else []
    
    if completed_folds:
        logger.info(f"    Resuming AB-09: {len(completed_folds)} folds already completed")
    
    for fold_i, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), 1):
        # Skip if already completed
        if fold_i in completed_folds:
            logger.debug(f"    Skipping Fold {fold_i} (already completed)")
            continue
        
        logger.info(f"    Processing Fold {fold_i}")
        
        X_train = X[tr_idx]
        y_train = y[tr_idx]
        
        # Compute feature importance for this fold
        try:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_train)
            selector = SelectKBest(f_classif, k=min(K_FEATURES, X_scaled.shape[1]))
            selector.fit(X_scaled, y_train)
            
            # NaN/Inf handling
            scores = np.nan_to_num(
                selector.scores_,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            
            fold_importances.append({
                "Fold": fold_i,
                "Scores": scores.tolist()  # Store as list for JSON
            })
            
        except Exception as exc:
            logger.warning(f"Fold {fold_i} stability computation failed: {exc}")
            fold_importances.append({
                "Fold": fold_i,
                "Scores": [0.0] * len(feature_names)
            })
        
        completed_folds.append(fold_i)
        
        # Save checkpoint after each fold
        _save_ab_checkpoint("AB09", {
            "status": "in_progress",
            "completed_folds": completed_folds,
            "fold_importances": fold_importances,
            "total_folds": len(list(loso.split(X, y, subjects))),
        })
    
    if not fold_importances:
        logger.warning("No valid fold stability could be computed.")
        return pd.DataFrame()
    
    # Compute stability across folds
    stability_rows = []
    for fi, fname in enumerate(feature_names):
        scores = [imp["Scores"][fi] for imp in fold_importances if fi < len(imp["Scores"])]
        
        if scores:
            stability_rows.append({
                "Feature_Index": fi,
                "Feature_Name": fname,
                "Fold_Mean": np.mean(scores),
                "Fold_Std": np.std(scores, ddof=1),
                "Fold_CV": np.std(scores, ddof=1) / (np.mean(scores) + 1e-10),
                "Fold_Stability": 1 / (1 + np.std(scores, ddof=1) / (np.mean(scores) + 1e-10)),
                "N_Folds": len(scores),
            })
    
    df_fold_stability = pd.DataFrame(stability_rows)
    df_fold_stability = df_fold_stability.sort_values(["Fold_Stability", "Feature_Name"], ascending=[False, True])
    
    # Save to Table_S32_Fold_Feature_Stability.csv
    save_csv(df_fold_stability, OUT_DIR / "Table_S32_Fold_Feature_Stability.csv")
    logger.info(f"  Table S32 saved: {len(df_fold_stability)} features")
    
    # Mark complete and remove checkpoint
    _mark_ab_completed("AB09", {
        "n_folds": len(completed_folds),
        "n_features": len(feature_names),
    })
    _cleanup_checkpoint_after_success("AB09", expected_outputs, should_remove=True)
    
    return df_fold_stability


# ============================================================================
# MAIN ABLATION ANALYSIS
# ============================================================================

def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
) -> None:
    """Run ablation analysis with two-level resume support and all three improvements."""
    
    # FIX: Use specific K values for consistency
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # Only keep values that are <= total features
    k_values = [k for k in k_values if k <= len(feature_names)]
    
    # ── Baseline (all features) ───────────────────────────────────────────
    # IMPROVEMENT 1: Check both checkpoint and CSV for baseline
    baseline_data = _load_ab_checkpoint("AB00")
    if baseline_data and baseline_data.get("status") == "complete":
        base_acc = baseline_data.get("base_acc", 0.0)
        base_std = baseline_data.get("base_std", 0.0)
        logger.info(f"  Baseline (all {len(feature_names)} features): {base_acc:.4f} ± {base_std:.4f} (from checkpoint)")
    else:
        all_mask = np.ones(len(feature_names), dtype=bool)
        base_acc, base_std = _loso_lda_accuracy(X, y, subjects, all_mask)
        logger.info(f"  Baseline (all {len(feature_names)} features): {base_acc:.4f} ± {base_std:.4f}")
        _save_ab_checkpoint("AB00", {
            "status": "complete",
            "base_acc": base_acc,
            "base_std": base_std,
            "n_features": len(feature_names),
        })

    # ── AB-01: Leave-One-Group-Out Ablation ──────────────────────────────
    df_logo = run_ab01_logo(X, y, subjects, feature_names, feature_groups, base_acc, base_std)

    # ── AB-02: Feature Group Contribution ──────────────────────────────────
    # Already computed in logo_rows
    
    # ── AB-03: Incremental Feature Addition ──────────────────────────────
    incr_accs, incr_stds, best_k, best_acc = run_ab03_incremental(
        X, y, subjects, feature_names, k_values
    )

    # ── AB-04: Feature Count Sensitivity ──────────────────────────────────
    # Already computed in AB-03 (Table_S8_Feature_Count_Sensitivity.csv)

    # ── AB-05: Per-Group Ablation Impact Ranking ──────────────────────────
    # Already computed from AB-01 (Table_S7_Feature_Group_Ranking.csv)
    df_ranking = df_logo[["Dropped_Group", "Accuracy_Drop", "Drop_Percentage"]].copy()
    df_ranking.columns = ["Group", "Accuracy_Drop", "Drop_Percentage"]
    df_ranking["Rank"] = range(1, len(df_ranking) + 1)
    save_csv(df_ranking, OUT_DIR / "Table_S7_Feature_Group_Ranking.csv")
    logger.info(f"  Table S7 saved: {len(df_ranking)} groups ranked")

    # ── Table 6: Ablation Results ─────────────────────────────────────────
    ablation_rows = [{"Config": "All Features", "N_Features": len(feature_names),
                      "Accuracy": round(base_acc, 4), "Std": round(base_std, 4)}]
    for _, r in df_logo.iterrows():
        ablation_rows.append({
            "Config": f"Drop {r['Dropped_Group']}",
            "N_Features": r["N_Features_Used"],
            "Accuracy": r["Accuracy_LOGO"],
            "Std": r["Accuracy_Std"],
        })
    df_ablation = pd.DataFrame(ablation_rows)

    save_csv(df_ablation, OUT_DIR / "Table6_Feature_Ablation_Results.csv")
    save_csv(df_logo, OUT_DIR / "Table7_Feature_Group_Contribution.csv")
    save_csv(df_ablation, OUT_DIR / "Table_S6_Complete_Ablation_Stats.csv")

    # ── AB-06: Feature Group Complementarity ──────────────────────────────
    df_complementarity = _compute_group_complementarity(
        X, y, subjects, feature_names, feature_groups, 
        np.ones(len(feature_names), dtype=bool), base_acc
    )

    # ── AB-07: Model-Wise Feature Analysis ────────────────────────────────
    logger.info("  AB-07: Preparing model-wise feature analysis...")
    logger.info(f"  Models: {MODEL_NAMES}")
    logger.info(f"  Feature groups: {list(feature_groups.keys())}")
    logger.info("  Calling compute_modelwise_feature_ranking()...")

    df_modelwise_pivot, df_modelwise_long = compute_modelwise_feature_ranking(
        X,
        y,
        subjects,
        feature_names,
        feature_groups,
    )

    logger.info("  Model-wise feature analysis completed successfully.")

    # ── AB-08: Subject-Wise Feature Stability ─────────────────────────────
    df_subject_stability = compute_subjectwise_feature_stability(
        X, y, subjects, feature_names, feature_groups
    )

    # ── AB-09: LOSO Fold Stability ────────────────────────────────────────
    df_fold_stability = compute_fold_stability(
        X, y, subjects, feature_names
    )

    # ── Figures 18–24 ─────────────────────────────────────────────────────
    palette_g = sns.color_palette("Set2", len(df_logo))

    # Fig 18 – Feature Group Contribution
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df_logo["Dropped_Group"], df_logo["Accuracy_Drop"],
                  color=palette_g, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, df_logo["Accuracy_Drop"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy Drop (Baseline − LOGO)")
    ax.set_title("Figure 18 – Feature Group Contribution (LOGO Ablation)")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig18_Feature_Group_Contribution.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 18 saved.")

    # Fig 19 – Ablation Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#66BB6A" if c=="All Features" else "#EF9A9A"
              for c in df_ablation["Config"]]
    ax.barh(df_ablation["Config"], df_ablation["Accuracy"],
            color=colors, edgecolor="white")
    ax.axvline(base_acc, color="navy", ls="--", lw=1.5,
               label=f"Baseline={base_acc:.4f}")
    ax.set_xlabel("LOSO LDA Accuracy")
    ax.set_title("Figure 19 – Feature Ablation Performance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig19_Ablation_Comparison.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 19 saved.")

    # Fig 20 – LOGO Performance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_logo["Dropped_Group"], df_logo["Accuracy_LOGO"],
           color=sns.color_palette("coolwarm", len(df_logo)), edgecolor="white")
    ax.axhline(base_acc, color="red", ls="--", lw=1.5, label=f"Baseline={base_acc:.4f}")
    ax.set_ylim(max(0, base_acc-0.15), min(1, base_acc+0.05))
    ax.set_ylabel("LOSO Accuracy")
    ax.set_title("Figure 20 – Leave-One-Group-Out Performance")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig20_LOGO_Performance.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 20 saved.")

    # Fig 21 – Incremental Feature Addition
    incr_arr = np.array(incr_accs)
    std_arr  = np.array(incr_stds)
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, incr_arr, marker="o", color="#1976D2", lw=2)
    ax.fill_between(k_values, incr_arr-std_arr, incr_arr+std_arr,
                    alpha=0.2, color="#1976D2")
    ax.set_xlabel("Number of Features (Top-K by F-score)")
    ax.set_ylabel("LOSO LDA Accuracy")
    ax.set_title("Figure 21 – Incremental Feature Addition Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig21_Incremental_Feature_Addition.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 21 saved.")

    # Fig 22 – Accuracy vs Feature Count
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(k_values, incr_arr, alpha=0.25, color="#42A5F5")
    ax.plot(k_values, incr_arr, marker="s", color="#0D47A1", lw=2)
    ax.axvline(best_k, color="red", ls="--", lw=1.5,
               label=f"Best K={best_k}  Acc={best_acc:.4f}")
    ax.set_xlabel("Feature Count")
    ax.set_ylabel("LOSO LDA Accuracy")
    ax.set_title("Figure 22 – Accuracy vs Feature Count Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig22_Accuracy_vs_FeatureCount.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 22 saved.")

    # Fig 23 – Feature Group Impact Heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_logo) * 0.4)))
    heat_data = df_logo.set_index("Dropped_Group")[["Accuracy_Drop", "Drop_Percentage"]]
    sns.heatmap(heat_data.T, annot=True, fmt=".3f", cmap="RdYlGn_r",
                cbar_kws={"label": "Accuracy Drop"}, ax=ax)
    ax.set_title("Figure 23 – Feature Group Impact Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Metric")
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig23_Feature_Group_Impact_Heatmap.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 23 saved.")

    # Fig 24 – Feature Group Complementarity
    if not df_complementarity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_comp = df_complementarity.sort_values("Complementarity_Drop", ascending=False)
        bars = ax.barh(sorted_comp["Group"], sorted_comp["Complementarity_Drop"],
                       color=sns.color_palette("viridis", len(sorted_comp)), edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        for bar, v in zip(bars, sorted_comp["Complementarity_Drop"]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Accuracy Drop When Group is Removed")
        ax.set_title("Figure 24 – Feature Group Complementarity Analysis\n"
                     "(How much each group uniquely contributes)")
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig24_Feature_Group_Complementarity.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure 24 saved.")

    # ── Figure 30 – Feature Contribution Radar Chart ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    group_names = df_logo["Dropped_Group"].tolist()
    drop_values = df_logo["Accuracy_Drop"].tolist()
    
    # Normalize drop values for radar (0 to 1 scale)
    max_drop = max(drop_values) if drop_values else 1.0
    normalized_drops = [v / (max_drop + 1e-10) for v in drop_values]
    
    # Close the loop
    group_names += group_names[:1]
    normalized_drops += normalized_drops[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(group_names), endpoint=True)
    
    ax.plot(angles, normalized_drops, 'o-', linewidth=2, color='#FF6B6B')
    ax.fill(angles, normalized_drops, alpha=0.25, color='#FF6B6B')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(group_names[:-1], size=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.set_title("Figure 30 – Feature Contribution Radar Chart\n"
                 "(Normalized Accuracy Drop)", size=12, pad=20)
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig30_Feature_Contribution_Radar.png")
    plt.close(fig)
    gc.collect()
    logger.info("  Figure 30 saved.")

    # ── Figure S20 – Model-Wise Feature Importance ──────────────────────────
    if not df_modelwise_long.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_data = df_modelwise_long.pivot_table(
            index="Feature_Group", 
            columns="Model", 
            values="Accuracy_Drop"
        ).fillna(0)
        
        pivot_data.plot(
            kind="bar",
            ax=ax,
            color=[MODEL_COLORS[m] for m in pivot_data.columns],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xlabel("Feature Group")
        ax.set_ylabel("Accuracy Drop")
        ax.set_title(
            "Figure S20 – Model-wise Feature Importance Comparison\n"
            "(Leave-One-Group-Out Accuracy Drop)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(
            title="Model",
            frameon=True,
            loc="best",
        )
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS20_ModelWise_Feature_Importance.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S20 saved.")

    # ── Figure S21 – Subject-Wise Feature Stability Heatmap ────────────────
    if not df_subject_stability.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_stable = df_subject_stability.head(20)
        
        heat_data = top_stable[["Feature_Name", "Stability_Score"]].set_index("Feature_Name").T
        sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlGnBu",
                    ax=ax, cbar_kws={"label": "Stability Score"})
        ax.set_title("Figure S21 – Subject-Wise Feature Stability Heatmap\n"
                     "(Top 20 Most Stable Features)", fontsize=12)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Stability")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS21_SubjectWise_Feature_Stability.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S21 saved.")

    # ── Figure S22 – Feature Stability Across LOSO Folds ────────────────────
    if not df_fold_stability.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_fold_stable = df_fold_stability.head(30)
        ax.barh(range(len(top_fold_stable)), top_fold_stable["Fold_Stability"],
                color='#4CAF50', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_fold_stable)))
        ax.set_yticklabels([f[:15] for f in top_fold_stable["Feature_Name"]], fontsize=8)
        ax.set_xlabel("Fold Stability Score")
        ax.set_title("Figure S22 – Feature Stability Across LOSO Folds\n"
                     "(Top 30 Most Stable Features)", fontsize=12)
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "FigS22_Feature_Stability_Across_Folds.png")
        plt.close(fig)
        gc.collect()
        logger.info("  Figure S22 saved.")

    logger.info(f"  Best K={best_k}, Best Accuracy={best_acc:.4f}")
    logger.info(f"  Baseline Accuracy={base_acc:.4f}, Best Accuracy={best_acc:.4f}")


def run_block5() -> Dict:
    """Main entry point for BLOCK 5."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 5 | AB1: Feature Contribution and Systematic Ablation Analysis")
    logger.info("=" * 70)
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

    # Log software versions
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
        X, y, subjects, feature_names = load_fallback_data()

    # Verify subject count
    n_subjects_detected = len(np.unique(subjects))
    if n_subjects_detected != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
            f"but detected {n_subjects_detected} subjects in the data."
        )
    logger.info(f"  Verified {n_subjects_detected} subjects (matches EXPECTED_SUBJECTS)")

    # ── Load or build feature groups ──────────────────────────────────────
    try:
        feature_groups = load_json(BLOCK_DIRS["BLOCK_2"] / "feature_groups.json")
        logger.info(f"  Loaded {len(feature_groups)} feature groups from BLOCK_2")
    except FileNotFoundError:
        logger.warning("  feature_groups.json not found - building from feature names")
        feature_groups = {}
        for fn in feature_names:
            g = get_feature_group(fn)
            feature_groups.setdefault(g, []).append(fn)
        logger.info(f"  Built {len(feature_groups)} feature groups")

    # ── Run ablation ──────────────────────────────────────────────────────
    run_ablation(X, y, subjects, feature_names, feature_groups)

    # ── Clean up AB00 baseline checkpoint ──────────────────────────────────
    _cleanup_baseline_checkpoint()

    # ── Save environment versions ──────────────────────────────────────────
    _save_environment_versions()

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "table6": OUT_DIR / "Table6_Feature_Ablation_Results.csv",
        "table7": OUT_DIR / "Table7_Feature_Group_Contribution.csv",
        "table_s6": OUT_DIR / "Table_S6_Complete_Ablation_Stats.csv",
        "table_s7": OUT_DIR / "Table_S7_Feature_Group_Ranking.csv",
        "table_s8": OUT_DIR / "Table_S8_Feature_Count_Sensitivity.csv",
        "table_s9": OUT_DIR / "Table_S9_Feature_Group_Complementarity.csv",
        "table_s29": OUT_DIR / "Table_S29_ModelWise_Feature_Ranking.csv",
        "table_s29_long": OUT_DIR / "Table_S29_ModelWise_Feature_Ranking_Long.csv",
        "table_s30": OUT_DIR / "Table_S30_SubjectWise_Feature_Ranking.csv",
        "table_s31": OUT_DIR / "Table_S31_Subject_Feature_Stability.csv",
        "table_s32": OUT_DIR / "Table_S32_Fold_Feature_Stability.csv",
        "env_versions": OUT_DIR / "environment_versions.json",
        "fig18": OUT_DIR / "Fig18_Feature_Group_Contribution.png",
        "fig19": OUT_DIR / "Fig19_Ablation_Comparison.png",
        "fig20": OUT_DIR / "Fig20_LOGO_Performance.png",
        "fig21": OUT_DIR / "Fig21_Incremental_Feature_Addition.png",
        "fig22": OUT_DIR / "Fig22_Accuracy_vs_FeatureCount.png",
        "fig23": OUT_DIR / "Fig23_Feature_Group_Impact_Heatmap.png",
        "fig24": OUT_DIR / "Fig24_Feature_Group_Complementarity.png",
        "fig30": OUT_DIR / "Fig30_Feature_Contribution_Radar.png",
        "fig_s20": OUT_DIR / "FigS20_ModelWise_Feature_Importance.png",
        "fig_s21": OUT_DIR / "FigS21_SubjectWise_Feature_Stability.png",
        "fig_s22": OUT_DIR / "FigS22_Feature_Stability_Across_Folds.png",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK 5 runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    summary = {
        "status": "complete",
        "n_groups": len(feature_groups),
        "n_features": len(feature_names),
        "runtime_sec": elapsed,
        "runtime_minutes": minutes,
        "runtime_seconds": seconds,
        "n_subjects_verified": n_subjects_detected,
        "pipeline": PIPELINE_NAME,
    }
    
    logger.info("=" * 70)
    logger.info("BLOCK 5 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Feature groups analyzed: {len(feature_groups)}")
    logger.info(f"  Total features: {len(feature_names)}")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)
    
    return summary


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return

    try:
        result = run_block5()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_5 WITH THREE IMPROVEMENTS
# ✓ IMPROVEMENT 1: _is_ab_completed() checks BOTH checkpoint status AND CSV existence
# ✓ IMPROVEMENT 2: AB-07 checkpoint stores only completed_models, not large result tables
# ✓ IMPROVEMENT 3: _cleanup_checkpoint_after_success() removes checkpoints after verification
# ✓ FIX: AB-07 loads existing rows from CSV on resume (preserves all results)
# ✓ FIX: AB-07 saves intermediate CSV after each model
# ✓ FIX: AB00 baseline checkpoint cleaned up after success
# ✓ FIX: Removed unused base_masks parameter
# ✓ TWO-LEVEL RESUME: AB-01 through AB-09 analysis-level checkpoints
# ✓ TWO-LEVEL RESUME: AB-07 model-level checkpoints (LDA, KNN, SVM, RF, XGB)
# ✓ Explicit cleanup in _loso_model_accuracy()
# ✓ Table S31 → Subject Stability, Table S32 → Fold Stability
# ✓ Per-model logging in AB-07
# ✓ NaN/Inf handling for feature_scores in AB-03
# ✓ NaN/Inf handling for selector.scores_ in AB-09
# ✓ Thread settings changed from "2" to "1"
# ✓ plt.close(fig) and gc.collect() after every figure
# ✓ Runtime logging with time.perf_counter()
# ✓ Environment versions saved
# ✓ Artifact manifest with verification
# ✓ Subject count verification
# ✓ print() replaced with logger
# ✓ assert replaced with ValueError
# ✓ AB-01 to AB-09 all complete with resume support
# ✓ Figures 18-24, 30, S20-S22
# ✓ Tables 6, 7, S6, S7, S8, S9, S29, S30, S31, S32
# ✓ Checkpoint directory created and cleaned after success
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────