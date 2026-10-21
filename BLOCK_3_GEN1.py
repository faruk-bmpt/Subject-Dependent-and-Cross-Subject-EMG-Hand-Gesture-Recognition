# BLOCK_3_GEN1.py - FULLY REFINED WITH CONFUSION MATRIX VISUALIZATIONS
# ================
# FILE PURPOSE:
#     BLOCK 3 — GEN1: Subject-Independent Generalization and Calibration Framework.
#     Evaluates real-world cross-subject generalization using strict Leave-One-Subject-Out (LOSO)
#     validation and quantifies the effectiveness of calibration for adapting models to unseen subjects.
#
#     CALIBRATION LEVELS: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 80%, 90%
#     (100% removed as it cannot be evaluated - no test samples remain)
#
#     GEN MODULES:
#     - GEN-01: LOSO Performance Evaluation
#     - GEN-02: Fold-Level Analysis
#     - GEN-03: Subject-Wise Analysis
#     - GEN-04: Inter-Subject Variability Analysis
#     - GEN-05: Calibration Performance Analysis
#     - GEN-06: Subject-Wise Calibration Analysis
#     - GEN-07: Model-Wise Calibration Analysis
#     - GEN-08: Gesture-Wise Calibration Analysis
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_0/grid_search_results.csv (for WS/OL sensitivity analysis)
#     - outputs/BLOCK_4/Table5_SubjectDependent_Summary.csv (optional, for fusion readiness)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_3/Table3_LOSO_Summary.csv
#     - outputs/BLOCK_3/Table4_SubjectWise_Metrics.csv
#     - outputs/BLOCK_3/Table_S5_PerFold_LOSO.csv
#     - outputs/BLOCK_3/Table_S5b_Fold_Statistics.csv
#     - outputs/BLOCK_3/PredictionArchive_LOSO.csv
#     - outputs/BLOCK_3/PredictionArchive_Calibration.csv
#     - outputs/BLOCK_3/loso_checkpoint.json
#     - outputs/BLOCK_3/calibration_checkpoint.json
#     - outputs/BLOCK_3/Table_S15_Calibration_Results_By_Subject.csv
#     - outputs/BLOCK_3/Table_S16_Calibration_Results_By_Model.csv
#     - outputs/BLOCK_3/Table_S21_WindowSize_Sensitivity.csv
#     - outputs/BLOCK_3/Table_S22_Overlap_Sensitivity.csv
#     - outputs/BLOCK_3/Table_S23_WS_OL_Sensitivity.csv
#     - outputs/BLOCK_3/Table_S24_Gesture_Calibration_Improvement.csv
#     - outputs/BLOCK_3/Table_S25_Calibration_Gain_Summary.csv
#     - outputs/BLOCK_3/Table_S26_Model_Calibration_Ranking.csv
#     - outputs/BLOCK_3/Table_S27_Subject_Calibration_Ranking.csv
#     - outputs/BLOCK_3/Table_S28_Gesture_Calibration_Ranking.csv
#     - outputs/BLOCK_3/Table_S29_Optimal_Calibration_Level.csv
#     - outputs/BLOCK_3/Table_S30_Calibration_Statistical_Comparison.csv
#     - outputs/BLOCK_3/Table_S31_Calibration_Metrics.csv
#     - outputs/BLOCK_3/Fig10_LOSO_Performance_Comparison.png
#     - outputs/BLOCK_3/Fig11_LOSO_Fold_Distribution.png
#     - outputs/BLOCK_3/Fig12_SubjectWise_Accuracy.png
#     - outputs/BLOCK_3/Fig13_SubjectWise_F1.png
#     - outputs/BLOCK_3/Fig14_SubjectWise_MCC.png
#     - outputs/BLOCK_3/Fig15_InterSubject_Variability.png
#     - outputs/BLOCK_3/Fig16_Accuracy_vs_Calibration.png
#     - outputs/BLOCK_3/Fig17_F1_vs_Calibration.png
#     - outputs/BLOCK_3/Fig18_MCC_vs_Calibration.png
#     - outputs/BLOCK_3/Fig19_Subject_Calibration_Heatmap.png
#     - outputs/BLOCK_3/Fig20_Model_Calibration_Heatmap.png
#     - outputs/BLOCK_3/Fig21_Gesture_Calibration_Improvement.png
#     - outputs/BLOCK_3/FigS14_WindowSize_Sensitivity.png
#     - outputs/BLOCK_3/FigS15_Overlap_Sensitivity.png
#     - outputs/BLOCK_3/FigS16_WS_OL_Sensitivity_Heatmap.png
#     - outputs/BLOCK_3/Fig22_Calibration_Gain_Curve.png
#     - outputs/BLOCK_3/Fig23_Calibration_Saturation_Curve.png
#     - outputs/BLOCK_3/Fig24_Subject_Gain_Distribution.png
#     - outputs/BLOCK_3/Fig25_Gesture_Gain_Distribution.png
#     - outputs/BLOCK_3/Fig26_Model_Calibration_Ranking.png
#     - outputs/BLOCK_3/Fig27_Reliability_Diagram.png
#     - outputs/BLOCK_3/Fig28_Calibration_Curve.png
#     - outputs/BLOCK_3/Fig29_Normalized_Difference_Confusion_Matrix_Heatmap.png
#     - outputs/BLOCK_3/Fig30_Subject_F1_Gain_Heatmap.png
#     - outputs/BLOCK_3/Fig31_Model_F1_Gain_Heatmap.png
#     - outputs/BLOCK_3/Fig32_Gesture_F1_Gain_Heatmap.png
#     - outputs/BLOCK_3/calibration_cm/ (organized by calibration level)
#       - Calibration_0/
#         - cm_XGB.csv, cm_XGB_normalized.csv, Fig_CM_XGB_0.png
#         - cm_LDA.csv, cm_LDA_normalized.csv, Fig_CM_LDA_0.png
#         - ... (for all models)
#       - Calibration_10/ (same structure)
#       - ... Calibration_90/ (same structure)
#     - outputs/BLOCK_3/calibration_report.txt
#     - outputs/BLOCK_3/artifacts.json
#     - state/BLOCK_3_GEN1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py, common.py
# STANDALONE RUN   : python BLOCK_3_GEN1.py

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import os
# ── Anti-segfault: use single-threaded math libraries ─────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── CRITICAL FIX: Disable Numba BEFORE any imports ──────────────────────────
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

# ── XGBoost Memory Safety ──────────────────────────────────────────────────
os.environ["XGBOOST_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XGBOOST_USE_CUDA"] = "0"
os.environ["XGBOOST_VERBOSITY"] = "0"

import faulthandler
faulthandler.enable()
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
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, brier_score_loss
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import calibration_curve
from scipy.stats import friedmanchisquare, rankdata
from scipy.interpolate import interp1d
import psutil

from config import (
    BLOCK_DIRS,
    GESTURE_SHORT,
    GESTURE_MAP,
    N_CLASSES,
    MODEL_NAMES,
    MODEL_COLORS,
    PALETTE,
    SEED,
    PIPELINE_NAME,
    N_CHANNELS,
    SAMPLING_RATE,
    MIN_RMS_THRESHOLD,
    VARIANCE_THRESHOLD,
    SMOTE_K_NEIGHBORS,
    WINDOW_SIZE,
    STEP_SIZE,
    INPUT_PATH,
    K_FEATURES,
    WINDOW_SIZES_GRID,
    OVERLAPS_GRID,
)
from state_manager import is_completed, mark_completed
from artifact_manager import (
    load_npy, save_csv, save_fig, save_json, load_feature_matrix, load_csv
)

# ── Import filter parameters with fallbacks ──────────────────────────────────
try:
    from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, NOTCH_FREQ, NOTCH_Q
except ImportError:
    BANDPASS_LOW = 10.0
    BANDPASS_HIGH = 45.0
    FILTER_ORDER = 4
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    logger = logging.getLogger("BLOCK_3")
    logger.warning("Using fallback filter parameters (not found in config)")

# ── EMG channel columns from config with fallback ──────────────────────────
try:
    from config import EMG_CHANNEL_COLUMNS
except ImportError:
    EMG_CHANNEL_COLUMNS = ["EMG_ch-01", "EMG_ch-02", "EMG_ch-03"]

# ── Try to import ARTIFACTS with fallback ────────────────────────────────────
try:
    from config import ARTIFACTS
except ImportError:
    ARTIFACTS = {}

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_3_GEN1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_3")

BLOCK_NAME = "BLOCK_3_GEN1"
OUT_DIR = BLOCK_DIRS["BLOCK_3"]

# ── Calibration directories ──────────────────────────────────────────────────
CALIBRATION_CM_DIR = OUT_DIR / "calibration_cm"
CALIBRATION_CM_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
for level in CALIBRATION_LEVELS:
    level_dir = CALIBRATION_CM_DIR / f"Calibration_{int(level*100)}"
    level_dir.mkdir(parents=True, exist_ok=True)


# ── Filters for fallback data loading ──────────────────────────────────────
from scipy.signal import butter, filtfilt
from scipy import signal as scipy_signal

_B_BP, _A_BP = butter(
    FILTER_ORDER,
    [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
    btype="band",
)
_B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)


def _preprocess(data: np.ndarray) -> np.ndarray:
    """Apply bandpass + notch filter."""
    data = filtfilt(_B_BP, _A_BP, data, axis=0)
    data = filtfilt(_B_NOTCH, _A_NOTCH, data, axis=0)
    return data


def _is_good(win: np.ndarray) -> bool:
    """Check if window meets RMS threshold."""
    return bool(np.all(np.sqrt(np.mean(win ** 2, axis=0)) > MIN_RMS_THRESHOLD))


def _get_quick_features(w: np.ndarray) -> List[float]:
    """Quick 9-feature proxy for fallback mode."""
    feats = []
    for ch in range(N_CHANNELS):
        s = w[:, ch].astype(np.float64)
        s -= s.mean()
        feats.extend([
            float(np.mean(np.abs(s))),
            float(np.sqrt(np.mean(s ** 2))),
            float(np.sum(np.abs(np.diff(s)))),
        ])
    return feats


def _load_fallback_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Fallback data loading if BLOCK_2 artifacts are not available."""
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
            for fpath in files[:5]:
                try:
                    df = pd.read_csv(
                        fpath,
                        usecols=EMG_CHANNEL_COLUMNS,
                        dtype=np.float32,
                    )
                    data = _preprocess(df.values)
                    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                        win = data[i: i + WINDOW_SIZE]
                        if _is_good(win):
                            feats = _get_quick_features(win)
                            X_list.append(feats)
                            y_list.append(lbl)
                            sub_list.append(sub)
                except Exception as e:
                    logger.debug(f"    Skipping file {fpath}: {e}")
                    continue

    if not X_list:
        raise RuntimeError("No data could be loaded from INPUT_PATH")

    logger.info(f"  Loaded {len(X_list)} windows from {len(subjects)} subjects")
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(sub_list),
        [f"feat_{i}" for i in range(len(X_list[0]))],
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESUME LOGIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_loso_checkpoint(
    completed_folds: List[int],
    completed_subjects: List[str],
    checkpoint_path: Path,
) -> None:
    """Save LOSO checkpoint for resume support."""
    checkpoint = {
        "completed_folds": completed_folds,
        "completed_subjects": completed_subjects,
        "last_updated": datetime.now().isoformat(),
        "total_folds": len(completed_folds),
    }
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"  LOSO checkpoint saved: {len(completed_folds)} folds completed")
    except Exception as e:
        logger.exception(f"  Failed to save LOSO checkpoint: {e}")


def _load_loso_checkpoint(checkpoint_path: Path) -> Tuple[List[int], List[str]]:
    """Load LOSO checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            completed_folds = checkpoint.get("completed_folds", [])
            completed_subjects = checkpoint.get("completed_subjects", [])
            logger.info(f"  Loaded LOSO checkpoint: {len(completed_folds)} folds previously completed")
            logger.debug(f"    Completed subjects: {completed_subjects}")
            return completed_folds, completed_subjects
        except Exception as e:
            logger.exception(f"  Could not load LOSO checkpoint: {e}")
    return [], []


def _save_calibration_checkpoint(
    completed_levels: List[float],
    checkpoint_path: Path,
) -> None:
    """Save calibration checkpoint for resume support."""
    checkpoint = {
        "completed_levels": completed_levels,
        "last_updated": datetime.now().isoformat(),
        "total_levels": len(completed_levels),
        "levels_pct": [int(l * 100) for l in completed_levels],
    }
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"  Calibration checkpoint saved: {len(completed_levels)} levels completed")
    except Exception as e:
        logger.exception(f"  Failed to save calibration checkpoint: {e}")


def _load_calibration_checkpoint(checkpoint_path: Path) -> List[float]:
    """Load calibration checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            completed_levels = checkpoint.get("completed_levels", [])
            logger.info(f"  Loaded calibration checkpoint: {len(completed_levels)} levels previously completed")
            logger.debug(f"    Completed levels: {[int(l*100) for l in completed_levels]}%")
            return completed_levels
        except Exception as e:
            logger.exception(f"  Could not load calibration checkpoint: {e}")
    return []


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
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"  Artifact manifest saved → {manifest_path}")
    except Exception as e:
        logger.exception(f"  Failed to save artifact manifest: {e}")


def _build_feature_pipeline(X_tr, y_tr):
    """Build feature pipeline: scaling -> variance threshold -> KBest."""
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

    logger.info("      Building feature pipeline")
    try:
        scaler = RobustScaler().fit(X_tr)
        X_s = scaler.transform(X_tr)
        sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD).fit(X_s)
        X_v = sel_var.transform(X_s)
        k_safe = min(K_FEATURES, X_v.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel_kb = SelectKBest(f_classif, k=k_safe).fit(X_v, y_tr)
        X_sel = sel_kb.transform(X_v)
        logger.info("      Feature pipeline OK")
        return X_sel, scaler, sel_var, sel_kb
    except Exception as e:
        logger.exception(f"      Feature pipeline failed: {e}")
        raise


def _apply_feature_pipeline(X, scaler, sel_var, sel_kb):
    """Apply fitted feature pipeline to test data."""
    try:
        return sel_kb.transform(sel_var.transform(scaler.transform(X)))
    except Exception as e:
        logger.exception(f"      Feature pipeline application failed: {e}")
        raise


def _apply_smote(X, y, tag=""):
    """Apply SMOTE if minority class has enough samples."""
    from imblearn.over_sampling import SMOTE

    logger.info("      Applying SMOTE")
    min_samples = min(Counter(y).values())
    if min_samples > SMOTE_K_NEIGHBORS:
        try:
            sm = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
            X, y = sm.fit_resample(X, y)
            logger.info(f"      SMOTE applied: {X.shape[0]} samples")
        except Exception as exc:
            logger.warning(f"  SMOTE failed {tag}: {exc}")
    else:
        logger.warning(
            f"  SMOTE skipped {tag} "
            f"(min_samples={min_samples}, "
            f"requires > {SMOTE_K_NEIGHBORS})"
        )
    return X, y


def _compute_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics with optional AUC."""
    try:
        metrics = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
    except Exception as e:
        logger.exception(f"      Metrics computation failed: {e}")
        return {}
    
    # Add ROC AUC if probabilities available
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            n_classes = len(np.unique(y_true))
            if y_prob.shape[1] >= n_classes:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                metrics["ROC_AUC"] = float(roc_auc)
            else:
                metrics["ROC_AUC"] = np.nan
        except Exception as e:
            logger.debug(f"      ROC-AUC computation failed: {e}")
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan
    
    # Add PR-AUC
    metrics["PR_AUC"] = np.nan
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            n_classes = len(np.unique(y_true))
            if y_prob.shape[1] >= n_classes:
                pr_aucs = []
                for i in range(n_classes):
                    y_true_binary = (y_true == i).astype(int)
                    precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
                    pr_auc = auc(recall_vals, precision_vals)
                    pr_aucs.append(pr_auc)
                metrics["PR_AUC"] = float(np.mean(pr_aucs))
        except Exception as e:
            logger.debug(f"      PR-AUC computation failed: {e}")
            metrics["PR_AUC"] = np.nan
    
    return metrics


def _compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean, lower CI, upper CI using SEM."""
    mean_val = np.mean(values)
    n = len(values)
    if n > 1:
        sem = np.std(values, ddof=1) / np.sqrt(n)
        z = 1.96  # 95% confidence
        ci_lower = mean_val - z * sem
        ci_upper = mean_val + z * sem
    else:
        ci_lower = mean_val
        ci_upper = mean_val
    return mean_val, ci_lower, ci_upper


def _compute_brier_score_multiclass(y_true, y_prob):
    """Compute Brier Score for multiclass (One-vs-Rest averaged)."""
    if y_prob is None or len(y_true) == 0:
        return np.nan
    
    try:
        n_classes = y_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        brier_scores = []
        for i in range(n_classes):
            bs = brier_score_loss(y_true_onehot[:, i], y_prob[:, i])
            brier_scores.append(bs)
        return float(np.mean(brier_scores))
    except Exception as e:
        logger.debug(f"      Brier score computation failed: {e}")
        return np.nan


def _compute_global_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute Global Expected Calibration Error (ECE) using pooled predictions."""
    if y_prob is None or len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    
    try:
        max_probs = np.max(y_prob, axis=1)
        pred_classes = np.argmax(y_prob, axis=1)
        
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(n_bins):
            bin_mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
            n_bin = np.sum(bin_mask)
            if n_bin == 0:
                continue
            
            bin_confidence = np.mean(max_probs[bin_mask])
            bin_accuracy = np.mean(pred_classes[bin_mask] == y_true[bin_mask])
            bin_weight = n_bin / n_samples
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    except Exception as e:
        logger.debug(f"      ECE computation failed: {e}")
        return np.nan


def _summarise_results(rows, metric_cols=None):
    """Aggregate fold-level rows into mean ± SD per model with CI."""
    if metric_cols is None:
        metric_cols = ["Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", 
                       "MCC", "Balanced_Accuracy", "ROC_AUC", "PR_AUC"]
    
    df = pd.DataFrame(rows)
    out = []
    for model, grp in df.groupby("Model"):
        row = {"Model": model}
        for mc in metric_cols:
            if mc not in grp:
                continue
            vals = grp[mc].dropna().values
            if len(vals) == 0:
                continue
            mean_val, ci_lower, ci_upper = _compute_confidence_interval(vals)
            row[f"{mc}_Mean"] = round(float(mean_val), 4)
            row[f"{mc}_SD"] = round(float(np.std(vals, ddof=1)), 4)
            row[f"{mc}_CI_Lower"] = round(float(ci_lower), 4)
            row[f"{mc}_CI_Upper"] = round(float(ci_upper), 4)
        out.append(row)
    return pd.DataFrame(out)


def _generate_fold_statistics(df_folds: pd.DataFrame) -> pd.DataFrame:
    """Generate fold-level statistics for Table S5b."""
    fold_stats = []
    for model in MODEL_NAMES:
        model_df = df_folds[df_folds["Model"] == model]
        if model_df.empty:
            continue
        accuracies = model_df["Accuracy"].values
        n_folds = len(accuracies)
        mean_val, ci_lower, ci_upper = _compute_confidence_interval(accuracies)
        fold_stats.append({
            "Model": model,
            "N_Folds": n_folds,
            "Mean_Accuracy": round(mean_val, 4),
            "Std_Accuracy": round(np.std(accuracies, ddof=1), 4),
            "CI_Lower": round(ci_lower, 4),
            "CI_Upper": round(ci_upper, 4),
            "Min_Accuracy": round(np.min(accuracies), 4),
            "Max_Accuracy": round(np.max(accuracies), 4),
            "Median_Accuracy": round(np.median(accuracies), 4),
            "Q1_Accuracy": round(np.percentile(accuracies, 25), 4),
            "Q3_Accuracy": round(np.percentile(accuracies, 75), 4),
            "IQR": round(np.percentile(accuracies, 75) - np.percentile(accuracies, 25), 4),
        })
    return pd.DataFrame(fold_stats)


def _train_models_with_calibration(X_tr, y_tr, X_cal, y_cal):
    """
    Train all models with optional calibration data.
    
    Returns:
        dict: trained models
    """
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    models = {
        "LDA": LDAClf(),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=1),
        "SVM": SVC(C=50, kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED, probability=True),
        "RF": RandomForestClassifier(n_estimators=250, max_depth=12, n_jobs=1, random_state=SEED, class_weight="balanced"),
        "XGB": XGBClassifier(
            objective="multi:softprob", num_class=N_CLASSES,
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, gamma=0.1,
            reg_lambda=1.5, reg_alpha=0.2, eval_metric="mlogloss",
            verbosity=0, n_jobs=1, random_state=SEED, tree_method="hist", predictor="cpu_predictor"
        ),
    }

    trained_models = {}
    
    # Log dataset composition for debugging
    logger.info(f"      Training data: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    if len(X_cal) > 0:
        logger.info(f"      Calibration data: {X_cal.shape[0]} samples, {X_cal.shape[1]} features")
        X_train_aug = np.vstack([X_tr, X_cal])
        y_train_aug = np.concatenate([y_tr, y_cal])
        logger.info(f"      Combined training: {X_train_aug.shape[0]} samples")
    else:
        X_train_aug = X_tr
        y_train_aug = y_tr
    
    # Log class distribution
    logger.info(f"      Class distribution: {Counter(y_train_aug)}")
    
    for name, clf in models.items():
        logger.info(f"      Training {name}...")
        
        try:
            t0 = time.perf_counter()
            clf.fit(X_train_aug, y_train_aug)
            elapsed = time.perf_counter() - t0
            logger.info(f"      {name} finished in {elapsed:.2f}s")
            trained_models[name] = clf
            gc.collect()
            
            # XGBoost releases native memory asynchronously
            if name == "XGB":
                time.sleep(0.2)
            
        except Exception as e:
            logger.exception(f"      {name} crashed: {e}")
            continue
        finally:
            gc.collect()
            time.sleep(0.02)
    
    # Clean up augmented data
    try:
        del X_train_aug
        del y_train_aug
    except Exception:
        pass
    gc.collect()
    
    return trained_models

# ============================================================================
# HELPER FUNCTION: PUBLICATION-QUALITY NORMALIZED CONFUSION MATRIX
# ============================================================================

def _plot_normalized_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    calibration_level: float,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """
    Generate and save a publication-quality normalized confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Raw confusion matrix (N_CLASSES × N_CLASSES).

    model_name : str
        Model name (e.g., LDA, KNN, SVM, RF, XGB).

    calibration_level : float
        Calibration percentage expressed as a fraction
        (0.0, 0.1, ..., 0.9).

    output_dir : Path
        Directory where the figure will be saved.

    title_prefix : str, optional
        Optional figure prefix (e.g., "Figure S17").
    """

    try:

        # --------------------------------------------------------------
        # Normalize confusion matrix using the project's helper function
        # --------------------------------------------------------------
        cm_norm = _normalize_cm(cm)

        level_pct = int(calibration_level * 100)

        # --------------------------------------------------------------
        # Create figure
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            square=True,
            xticklabels=GESTURE_SHORT,
            yticklabels=GESTURE_SHORT,
            cbar_kws={"label": "Row-Normalized Accuracy"},
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        # --------------------------------------------------------------
        # Labels
        # --------------------------------------------------------------
        ax.set_xlabel(
            "Predicted Gesture",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(
            "True Gesture",
            fontsize=12,
            fontweight="bold"
        )

        # --------------------------------------------------------------
        # Title
        # --------------------------------------------------------------
        if title_prefix:

            title = (
                f"{title_prefix}\n"
                f"{model_name} Normalized Confusion Matrix\n"
                f"{level_pct}% Calibration"
            )

        else:

            title = (
                f"{model_name} Normalized Confusion Matrix\n"
                f"{level_pct}% Calibration"
            )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        # --------------------------------------------------------------
        # Ensure output directory exists
        # --------------------------------------------------------------
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------------------
        # Save figure using centralized artifact manager
        # --------------------------------------------------------------
        fig_filename = f"Fig_CM_{model_name}_{level_pct}.png"
        fig_path = output_dir / fig_filename

        save_fig(fig, fig_path)

        plt.close(fig)

        logger.debug(
            f"    Normalized confusion matrix saved: {fig_path}"
        )

    except Exception as e:

        logger.exception(
            f"    Failed to generate normalized confusion matrix "
            f"for {model_name} @ {level_pct}% calibration: {e}"
        )

# ============================================================================
# GEN-01 to GEN-04: LOSO Evaluation
# ============================================================================

def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict Leave-One-Subject-Out evaluation.
    Feature pipeline fitted on training subjects only.
    
    Saves PredictionArchive_LOSO.csv with all predictions.
    Checkpoint resume support via loso_checkpoint.json.
    """
    loso = LeaveOneGroupOut()
    all_rows: List[Dict] = []
    cm_accum = {m: np.zeros((N_CLASSES, N_CLASSES), dtype=int) for m in MODEL_NAMES}
    subject_metrics = defaultdict(list)
    
    # Prediction archive for LOSO
    prediction_archive: List[Dict] = []

    unique_subs = np.unique(subjects)
    total_folds = len(unique_subs)
    
    # RESUME: Check for existing checkpoint
    checkpoint_path = OUT_DIR / "loso_checkpoint.json"
    completed_folds, completed_subjects = _load_loso_checkpoint(checkpoint_path)
    completed_set = set(completed_subjects)
    
    logger.info(f"  LOSO: {total_folds} subjects → {total_folds} folds")
    logger.info(f"  Dataset shape: X={X.shape}, y={y.shape}, subjects={len(unique_subs)}")
    if completed_set:
        logger.info(f"  RESUMING: {len(completed_set)} folds already completed, skipping")
        logger.debug(f"    Completed subjects: {completed_set}")

    fold_counter = 0
    total_start_time = time.perf_counter()
    
    for fold_i, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), 1):
        fold_start_time = time.perf_counter()
        test_sub = subjects[te_idx[0]]
        
        # RESUME: Skip already completed folds
        if test_sub in completed_set:
            logger.debug(f"  Fold {fold_i:2d}/{total_folds} | held-out: {test_sub} (already completed, skipping)")
            fold_counter += 1
            continue
            
        fold_counter += 1
        logger.info(f"  Fold {fold_i:2d}/{total_folds} | held-out: {test_sub}")

        X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
        X_te_raw, y_te = X[te_idx], y[te_idx]
        
        # Log dataset shapes for this fold
        logger.info(f"      Training set: {X_tr_raw.shape[0]} samples, {X_tr_raw.shape[1]} features")
        logger.info(f"      Test set: {X_te_raw.shape[0]} samples, {X_te_raw.shape[1]} features")
        logger.info(f"      Training classes: {Counter(y_tr)}")

        try:
            X_tr_sel, sc, sv, sk = _build_feature_pipeline(X_tr_raw, y_tr)
            X_te_sel = _apply_feature_pipeline(X_te_raw, sc, sv, sk)
        except Exception as e:
            logger.exception(f"      Feature pipeline failed for fold {fold_i} (subject {test_sub}): {e}")
            del X_tr_raw, y_tr, X_te_raw, y_te
            gc.collect()
            continue

        X_tr_final, y_tr_final = _apply_smote(X_tr_sel, y_tr, tag=f"fold{fold_i}")
        logger.info(f"      After SMOTE: {X_tr_final.shape[0]} samples")

        logger.info("      Training models")
        
        process = psutil.Process(os.getpid())
        logger.info(f"      RAM BEFORE TRAIN: {process.memory_info().rss / 1024**2:.1f} MB")
        
        try:
            models = _train_models_with_calibration(X_tr_final, y_tr_final, np.array([]), np.array([]))
        except Exception as e:
            logger.exception(f"      Training failed for fold {fold_i} (subject {test_sub}): {e}")
            del X_tr_raw, y_tr, X_te_raw, y_te, X_tr_sel, X_te_sel, X_tr_final, y_tr_final
            gc.collect()
            continue
        
        process = psutil.Process(os.getpid())
        logger.info(f"      RAM AFTER TRAIN : {process.memory_info().rss / 1024**2:.1f} MB")
        logger.info("      Models trained")
        
        for name, clf in models.items():
            logger.info(f"        Predicting with {name}")

            try:
                preds = clf.predict(X_te_sel)
                logger.info(f"        Prediction finished: {name}")
            except Exception as e:
                logger.exception(f"        Prediction FAILED for {name}: {e}")
                continue

            y_prob = None
            if hasattr(clf, "predict_proba"):
                try:
                    logger.info(f"        Computing probabilities: {name}")
                    y_prob = clf.predict_proba(X_te_sel)
                    logger.info(f"        Probability computation finished: {name}")
                except Exception as e:
                    logger.warning(f"        predict_proba failed for {name}: {e}")

            try:
                metrics = _compute_metrics(y_te, preds, y_prob)
            except Exception as e:
                logger.exception(f"        Metrics computation failed for {name}: {e}")
                continue

            # Store predictions in archive
            prediction_archive.append({
                "Subject": test_sub,
                "Fold": fold_i,
                "Model": name,
                "True_Label": y_te.tolist(),
                "Pred_Label": preds.tolist(),
            })

            try:
                cm = confusion_matrix(y_te, preds, labels=list(range(N_CLASSES)))
                cm_accum[name] += cm
            except Exception as e:
                logger.exception(f"        Confusion matrix failed for {name}: {e}")
                continue

            subject_metrics[test_sub].append({
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                "Macro_F1": metrics["Macro_F1"],
                "MCC": metrics["MCC"],
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "ROC_AUC": metrics.get("ROC_AUC", np.nan),
                "PR_AUC": metrics.get("PR_AUC", np.nan),
            })

            all_rows.append({
                "Model": name,
                "Fold": fold_i,
                "Test_Subject": test_sub,
                **metrics
            })

        # RESUME: Save checkpoint after each fold
        completed_folds.append(fold_i)
        completed_subjects.append(test_sub)
        _save_loso_checkpoint(completed_folds, completed_subjects, checkpoint_path)
        
        fold_elapsed = time.perf_counter() - fold_start_time
        total_elapsed = time.perf_counter() - total_start_time
        logger.info(f"      Fold {fold_i} completed in {fold_elapsed:.1f}s")
        logger.info(f"      Total elapsed: {total_elapsed/60:.1f} min, remaining folds: {total_folds - fold_i}")
        
        if fold_i < total_folds:
            avg_time_per_fold = total_elapsed / fold_i
            estimated_remaining = avg_time_per_fold * (total_folds - fold_i)
            logger.info(f"      Estimated time remaining: {estimated_remaining/60:.1f} min")

        # Explicit cleanup
        try:
            del X_tr_raw, y_tr, X_te_raw, y_te
            del X_tr_sel, X_te_sel, X_tr_final, y_tr_final
            del sc, sv, sk
            
            if "models" in locals():
                for model in models.values():
                    if hasattr(model, "_Booster"):
                        try:
                            del model._Booster
                        except Exception:
                            pass
                    del model
                del models
        except Exception as e:
            logger.debug(f"      Cleanup warning: {e}")
        
        gc.collect()
        time.sleep(0.02)
        
        process = psutil.Process(os.getpid())
        logger.debug(f"      RAM AFTER CLEANUP: {process.memory_info().rss / 1024**2:.1f} MB")

    # Remove checkpoint after completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info("  LOSO checkpoint removed (successful completion)")
        except Exception as e:
            logger.exception(f"  Failed to remove LOSO checkpoint: {e}")

    # Save Prediction Archive
    if prediction_archive:
        expanded_rows = []
        for entry in prediction_archive:
            for true_val, pred_val in zip(entry["True_Label"], entry["Pred_Label"]):
                expanded_rows.append({
                    "Subject": entry["Subject"],
                    "Fold": entry["Fold"],
                    "Model": entry["Model"],
                    "True_Label": true_val,
                    "Pred_Label": pred_val,
                })
        df_pred_archive = pd.DataFrame(expanded_rows)
        try:
            save_csv(df_pred_archive, OUT_DIR / "PredictionArchive_LOSO.csv")
            logger.info(f"  PredictionArchive_LOSO.csv saved: {len(df_pred_archive)} predictions")
        except Exception as e:
            logger.exception(f"  Failed to save PredictionArchive_LOSO.csv: {e}")

    # Save aggregated CMs
    for m, cm in cm_accum.items():
        try:
            pd.DataFrame(cm, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(
                OUT_DIR / f"loso_cm_{m}.csv"
            )
        except Exception as e:
            logger.exception(f"  Failed to save CM for {m}: {e}")

    df_folds = pd.DataFrame(all_rows)
    df_fold_stats = _generate_fold_statistics(df_folds)
    try:
        save_csv(df_fold_stats, OUT_DIR / "Table_S5b_Fold_Statistics.csv")
        logger.info(f"  Table S5b saved: {len(df_fold_stats)} model summaries")
    except Exception as e:
        logger.exception(f"  Failed to save Table S5b: {e}")

    # Subject-wise aggregation
    sub_rows = []
    for sub, metrics in subject_metrics.items():
        for m in MODEL_NAMES:
            m_metrics = [mm for mm in metrics if mm["Model"] == m]
            if m_metrics:
                sub_rows.append({
                    "Subject": sub,
                    "Model": m,
                    "Accuracy": np.mean([mm["Accuracy"] for mm in m_metrics]),
                    "Macro_F1": np.mean([mm["Macro_F1"] for mm in m_metrics]),
                    "MCC": np.mean([mm["MCC"] for mm in m_metrics]),
                    "Balanced_Accuracy": np.mean([mm["Balanced_Accuracy"] for mm in m_metrics]),
                    "ROC_AUC": np.nan if all(np.isnan([mm.get("ROC_AUC", np.nan) for mm in m_metrics])) else np.mean([mm.get("ROC_AUC", np.nan) for mm in m_metrics if not np.isnan(mm.get("ROC_AUC", np.nan))]),
                    "PR_AUC": np.nan if all(np.isnan([mm.get("PR_AUC", np.nan) for mm in m_metrics])) else np.mean([mm.get("PR_AUC", np.nan) for mm in m_metrics if not np.isnan(mm.get("PR_AUC", np.nan))]),
                })
    df_subjects = pd.DataFrame(sub_rows)
    try:
        save_csv(df_subjects, OUT_DIR / "Table4_SubjectWise_Metrics.csv")
    except Exception as e:
        logger.exception(f"  Failed to save Table4: {e}")

    return df_folds, df_fold_stats, df_subjects


# ============================================================================
# GEN-05 to GEN-08: CALIBRATION FRAMEWORK
# ============================================================================

def _compute_calibration_gain(df_cal_model: pd.DataFrame) -> pd.DataFrame:
    """Compute calibration gain metrics using Macro-F1 as primary criterion."""
    gain_rows = []
    
    for model in MODEL_NAMES:
        model_data = df_cal_model[df_cal_model["Model"] == model]
        if model_data.empty:
            continue
        
        model_data = model_data.sort_values("Calibration_Level")
        
        baseline = model_data[model_data["Calibration_Level"] == 0.0]
        if baseline.empty:
            baseline_f1 = 0.0
            baseline_acc = 0.0
            baseline_bal = 0.0
            baseline_mcc = 0.0
            baseline_roc = 0.0
            baseline_pr = 0.0
        else:
            baseline_f1 = baseline["F1_Mean"].values[0]
            baseline_acc = baseline["Accuracy_Mean"].values[0]
            baseline_bal = baseline["Balanced_Accuracy_Mean"].values[0] if "Balanced_Accuracy_Mean" in baseline.columns else 0.0
            baseline_mcc = baseline["MCC_Mean"].values[0]
            baseline_roc = baseline["ROC_AUC_Mean"].values[0] if "ROC_AUC_Mean" in baseline.columns else 0.0
            baseline_pr = baseline["PR_AUC_Mean"].values[0] if "PR_AUC_Mean" in baseline.columns else 0.0
        
        for _, row in model_data.iterrows():
            level = row["Calibration_Level"]
            f1 = row["F1_Mean"]
            acc = row["Accuracy_Mean"]
            bal = row["Balanced_Accuracy_Mean"] if "Balanced_Accuracy_Mean" in row else 0.0
            mcc = row["MCC_Mean"]
            roc = row["ROC_AUC_Mean"] if "ROC_AUC_Mean" in row else 0.0
            pr = row["PR_AUC_Mean"] if "PR_AUC_Mean" in row else 0.0
            
            gain_rows.append({
                "Model": model,
                "Calibration_Level": level,
                "F1_Gain": f1 - baseline_f1,
                "Accuracy_Gain": acc - baseline_acc,
                "Balanced_Accuracy_Gain": bal - baseline_bal,
                "MCC_Gain": mcc - baseline_mcc,
                "ROC_AUC_Gain": roc - baseline_roc,
                "PR_AUC_Gain": pr - baseline_pr,
                "F1_Value": f1,
                "Accuracy_Value": acc,
                "Balanced_Accuracy_Value": bal,
                "MCC_Value": mcc,
                "ROC_AUC_Value": roc,
                "PR_AUC_Value": pr,
                "Baseline_F1": baseline_f1,
                "Baseline_Accuracy": baseline_acc,
            })
    
    df_gain = pd.DataFrame(gain_rows)
    return df_gain


def _find_optimal_calibration_level(df_gain: pd.DataFrame) -> pd.DataFrame:
    """
    Find the optimal calibration level for each model using Macro-F1.

    Saturation criterion:
        Calibration is considered saturated when the incremental Macro-F1
        improvement is <1% for TWO consecutive calibration levels.
    """
    optimal_rows = []

    for model in MODEL_NAMES:
        model_gain = df_gain[df_gain["Model"] == model].copy()
        if model_gain.empty:
            continue

        model_gain = model_gain.sort_values("Calibration_Level").reset_index(drop=True)

        # Best calibration level (highest Macro-F1)
        best_row = model_gain.loc[model_gain["F1_Value"].idxmax()]

        # Saturation detection
        saturation_level = best_row["Calibration_Level"]

        for i in range(len(model_gain) - 2):
            current = model_gain.iloc[i]
            next_level = model_gain.iloc[i + 1]
            next_next_level = model_gain.iloc[i + 2]

            gain1 = next_level["F1_Value"] - current["F1_Value"]
            gain2 = next_next_level["F1_Value"] - next_level["F1_Value"]

            if (
                gain1 < 0.01 * current["F1_Value"]
                and
                gain2 < 0.01 * next_level["F1_Value"]
            ):
                saturation_level = current["Calibration_Level"]
                break

        # F1 score at saturation level
        sat_match = model_gain[model_gain["Calibration_Level"] == saturation_level]

        if not sat_match.empty:
            f1_at_saturation = sat_match.iloc[0]["F1_Value"]
        else:
            f1_at_saturation = best_row["F1_Value"]

        optimal_rows.append({
            "Model": model,
            "Optimal_Calibration_Level": best_row["Calibration_Level"],
            "Optimal_F1": best_row["F1_Value"],
            "Baseline_F1": best_row["Baseline_F1"],
            "Optimal_Accuracy": best_row["Accuracy_Value"],
            "Optimal_Balanced_Accuracy": best_row["Balanced_Accuracy_Value"],
            "Optimal_MCC": best_row["MCC_Value"],
            "Optimal_ROC_AUC": best_row.get("ROC_AUC_Value", np.nan),
            "Optimal_PR_AUC": best_row.get("PR_AUC_Value", np.nan),
            "F1_Gain": best_row["F1_Gain"],
            "Accuracy_Gain": best_row["Accuracy_Gain"],
            "Saturation_Level": saturation_level,
            "F1_at_Saturation": f1_at_saturation,
        })

    return pd.DataFrame(optimal_rows)


def _compute_statistical_tests_calibration(df_cal_subject: pd.DataFrame) -> pd.DataFrame:
    """Perform Friedman test across calibration levels using Subject as repeated measure."""
    try:
        try:
            import scikit_posthocs as sp
        except ImportError:
            logger.warning("  scikit_posthocs not available. Nemenyi post-hoc skipped.")
            return pd.DataFrame()
        
        subject_pivot = df_cal_subject.pivot_table(
            index="Subject",
            columns="Calibration_Level",
            values="Macro_F1",
            aggfunc="mean"
        ).dropna()
        
        if subject_pivot.shape[0] < 2 or subject_pivot.shape[1] < 2:
            return pd.DataFrame()
        
        # Friedman test across calibration levels
        scores = [subject_pivot[col].values for col in subject_pivot.columns]
        stat, p = friedmanchisquare(*scores)
        
        results = [{
            "Test": "Friedman_Calibration_Levels",
            "Statistic": stat,
            "P_Value": p,
            "Significant": "Yes" if p < 0.05 else "No"
        }]
        
        # Nemenyi post-hoc
        if p < 0.05:
            ranks = np.array([rankdata(-row) for row in subject_pivot.values])
            mean_ranks = ranks.mean(axis=0)
            
            for level, rank in zip(subject_pivot.columns, mean_ranks):
                results.append({
                    "Test": "Nemenyi_Ranking",
                    "Calibration_Level": level,
                    "Mean_Rank": rank,
                    "P_Value": np.nan,
                    "Significant": np.nan
                })
            
            nemenyi = sp.posthoc_nemenyi_friedman(subject_pivot.values)
            nemenyi_df = pd.DataFrame(nemenyi, index=[f"{int(c*100)}%" for c in subject_pivot.columns], 
                                     columns=[f"{int(c*100)}%" for c in subject_pivot.columns])
            try:
                nemenyi_df.to_csv(OUT_DIR / "nemenyi_calibration_levels.csv")
            except Exception as e:
                logger.exception(f"  Failed to save Nemenyi results: {e}")
        
        return pd.DataFrame(results)
    
    except Exception as e:
        logger.exception(f"  Statistical tests failed: {e}")
    
    return pd.DataFrame()


def _generate_multiclass_reliability_curve(y_true, y_prob, n_bins=10):
    """Generate reliability curve for multiclass using One-vs-Rest averaging."""
    if y_prob is None or len(y_true) == 0:
        return [], []
    
    try:
        n_classes = y_prob.shape[1]
        common_bins = np.linspace(0, 1, n_bins + 1)
        common_centers = (common_bins[:-1] + common_bins[1:]) / 2
        
        all_fraction_positives = []
        
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            class_probs = y_prob[:, i]
            
            fraction_positives, mean_predicted = calibration_curve(
                y_true_binary, 
                class_probs, 
                n_bins=n_bins,
                strategy='uniform'
            )
            
            if len(mean_predicted) < 2:
                continue
            
            interp_frac = interp1d(
                mean_predicted, fraction_positives, 
                kind='linear', fill_value='extrapolate'
            )
            frac_interp = interp_frac(common_centers)
            frac_interp = np.clip(frac_interp, 0, 1)
            
            all_fraction_positives.append(frac_interp)
        
        if len(all_fraction_positives) == 0:
            return [], []
        
        avg_fraction_positives = np.mean(all_fraction_positives, axis=0)
        return common_centers, avg_fraction_positives
    
    except Exception as e:
        logger.exception(f"  Multiclass reliability curve failed: {e}")
        return [], []


def _build_calibration_confusion_matrices_from_archive(
    prediction_archive_path: Path,
    calibration_level: float,
    model_name: str
) -> np.ndarray:
    """
    Build confusion matrix from PredictionArchive_Calibration.csv for a specific
    calibration level and model.
    
    This is the recommended approach: build CMs from the archive after all folds complete.
    """
    try:
        df = pd.read_csv(prediction_archive_path)
        df_filtered = df[
            (df["Calibration_Level"] == calibration_level) & 
            (df["Model"] == model_name)
        ]
        
        if len(df_filtered) == 0:
            return np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        
        y_true = df_filtered["True_Label"].values
        y_pred = df_filtered["Pred_Label"].values
        
        return confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    
    except Exception as e:
        logger.exception(f"  Failed to build CM from archive for {model_name} @ {calibration_level}: {e}")
        return np.zeros((N_CLASSES, N_CLASSES), dtype=int)


def _build_loso_confusion_matrices_from_archive(
    prediction_archive_path: Path,
    model_name: str
) -> np.ndarray:
    """Build LOSO confusion matrix from PredictionArchive_LOSO.csv."""
    try:
        df = pd.read_csv(prediction_archive_path)
        df_filtered = df[df["Model"] == model_name]
        
        if len(df_filtered) == 0:
            return np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        
        y_true = df_filtered["True_Label"].values
        y_pred = df_filtered["Pred_Label"].values
        
        return confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    
    except Exception as e:
        logger.exception(f"  Failed to build LOSO CM from archive for {model_name}: {e}")
        return np.zeros((N_CLASSES, N_CLASSES), dtype=int)


def run_calibration_analysis(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    calibration_levels: List[float] = CALIBRATION_LEVELS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Evaluate calibration effectiveness with proper few-shot adaptation.
    
    FIX: Feature pipeline built ONLY on training subjects (no leakage).
    FIX: global_predictions separated by calibration level.
    FIX: CMs built from PredictionArchive_Calibration.csv after completion.
    NEW: Confusion matrix visualizations generated for every calibration level and model.
    """
    loso = LeaveOneGroupOut()
    unique_subs = np.unique(subjects)
    
    cal_subject_results = []
    cal_gesture_results = []
    
    # FIX: Separate global predictions by calibration level
    # Structure: global_predictions[cal_level][model] = {"true": [], "pred": [], "prob": []}
    global_predictions = {}
    for level in calibration_levels:
        global_predictions[level] = {}
        for model in MODEL_NAMES:
            global_predictions[level][model] = {"true": [], "pred": [], "prob": []}
    
    # Prediction archive for Calibration
    cal_prediction_archive: List[Dict] = []
    
    logger.info(f"  Calibration Analysis: {len(calibration_levels)} levels, {len(unique_subs)} subjects")
    logger.info(f"  Dataset shape: X={X.shape}, y={y.shape}, subjects={len(unique_subs)}")
    
    # RESUME: Check for existing checkpoint
    checkpoint_path = OUT_DIR / "calibration_checkpoint.json"
    completed_levels = _load_calibration_checkpoint(checkpoint_path)
    completed_levels_set = set(completed_levels)
    
    if completed_levels_set:
        logger.info(f"  RESUMING: {len(completed_levels_set)} calibration levels already completed")
        logger.debug(f"    Completed levels: {[int(l*100) for l in completed_levels]}%")
    
    total_start_time = time.perf_counter()
    level_counter = 0

    for cal_level in calibration_levels:
        level_start_time = time.perf_counter()
        
        # RESUME: Skip already completed levels
        if cal_level in completed_levels_set:
            logger.debug(f"  Calibration {int(cal_level*100)}% already completed, skipping")
            continue
        
        level_counter += 1
        level_dir = CALIBRATION_CM_DIR / f"Calibration_{int(cal_level*100)}"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"    Processing calibration {int(cal_level*100)}%")
        
        for fold_i, (tr_idx, te_idx) in enumerate(loso.split(X, y, subjects), 1):
            fold_start_time = time.perf_counter()
            test_sub = subjects[te_idx[0]]
            
            X_tr_raw, y_tr = X[tr_idx], y[tr_idx]
            X_te_raw, y_te = X[te_idx], y[te_idx]
            
            logger.debug(f"    Calibration {int(cal_level*100):2d}% | Fold {fold_i}/{len(unique_subs)} | Subject {test_sub}")
            logger.debug(f"      Training set: {X_tr_raw.shape[0]} samples, {X_tr_raw.shape[1]} features")
            logger.debug(f"      Test set: {X_te_raw.shape[0]} samples, {X_te_raw.shape[1]} features")

            n_cal = int(len(X_te_raw) * cal_level)
            logger.debug(f"      Calibration samples: {n_cal}, Evaluation samples: {len(X_te_raw) - n_cal}")
            
            if n_cal < 1 and cal_level > 0:
                logger.debug(f"      Skipping fold {fold_i} - insufficient calibration samples")
                continue
            
            if n_cal > 0 and cal_level < 1.0:
                try:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, 
                        test_size=1.0 - cal_level, 
                        random_state=SEED + fold_i
                    )
                    for cal_idx_split, eval_idx_split in sss.split(X_te_raw, y_te):
                        cal_idx = cal_idx_split
                        eval_idx = eval_idx_split
                        break
                    
                    X_cal_raw, y_cal = X_te_raw[cal_idx], y_te[cal_idx]
                    X_eval_raw, y_eval = X_te_raw[eval_idx], y_te[eval_idx]
                    
                    logger.debug(f"      Calibration split: {len(X_cal_raw)} samples, Evaluation split: {len(X_eval_raw)} samples")
                    
                except Exception as e:
                    logger.exception(f"      Stratified split failed for fold {fold_i}: {e}")
                    continue
            else:
                X_cal_raw, y_cal = np.array([]), np.array([])
                X_eval_raw, y_eval = X_te_raw, y_te

            # Build feature pipeline ONLY on training subjects (no leakage)
            try:
                X_tr_sel, sc, sv, sk = _build_feature_pipeline(X_tr_raw, y_tr)
            except Exception as e:
                logger.exception(f"      Feature pipeline failed for fold {fold_i}: {e}")
                continue
            
            # Apply pipeline to training, calibration, and evaluation
            X_tr_sel_final = X_tr_sel
            X_cal_sel = _apply_feature_pipeline(X_cal_raw, sc, sv, sk) if len(X_cal_raw) > 0 else np.array([])
            X_eval_sel = _apply_feature_pipeline(X_eval_raw, sc, sv, sk)
            
            # SMOTE on training only
            X_tr_final, y_tr_final = _apply_smote(X_tr_sel_final, y_tr, tag=f"cal_fold{fold_i}")
            logger.debug(f"      After SMOTE: {X_tr_final.shape[0]} training samples")

            if len(X_eval_sel) > 0:
                # Train models with calibration
                logger.info("      Training models")
                
                process = psutil.Process(os.getpid())
                logger.info(f"      RAM BEFORE TRAIN: {process.memory_info().rss / 1024**2:.1f} MB")
                
                try:
                    trained_models = _train_models_with_calibration(
                        X_tr_final, y_tr_final,
                        X_cal_sel, y_cal
                    )
                except Exception as e:
                    logger.exception(f"      Training failed for calibration {int(cal_level*100)}%, fold {fold_i}: {e}")
                    del X_tr_raw, y_tr, X_te_raw, y_te
                    gc.collect()
                    continue
                
                process = psutil.Process(os.getpid())
                logger.info(f"      RAM AFTER TRAIN : {process.memory_info().rss / 1024**2:.1f} MB")
                logger.info("      Models trained")
                
                for name, clf in trained_models.items():
                    # Prediction
                    logger.info(f"        Predicting with {name}")

                    try:
                        preds = clf.predict(X_eval_sel)
                        logger.info(f"        Prediction finished: {name}")
                    except Exception as e:
                        logger.exception(f"        Prediction FAILED for {name}: {e}")
                        continue

                    # Probability estimation
                    y_prob = None
                    if hasattr(clf, "predict_proba"):
                        try:
                            logger.info(f"        Computing probabilities: {name}")
                            y_prob = clf.predict_proba(X_eval_sel)
                            logger.info(f"        Probability computation finished: {name}")
                        except Exception as e:
                            logger.warning(f"        predict_proba failed for {name}: {e}")

                    # Metrics
                    try:
                        metrics = _compute_metrics(y_eval, preds, y_prob)
                    except Exception as e:
                        logger.exception(f"        Metrics computation failed for {name}: {e}")
                        continue

                    # Store predictions in calibration archive
                    cal_prediction_archive.append({
                        "Calibration_Level": cal_level,
                        "Subject": test_sub,
                        "Fold": fold_i,
                        "Model": name,
                        "True_Label": y_eval.tolist(),
                        "Pred_Label": preds.tolist(),
                    })

                    # FIX: Store global predictions by calibration level
                    try:
                        global_predictions[cal_level][name]["true"].extend(y_eval.tolist())
                        global_predictions[cal_level][name]["pred"].extend(preds.tolist())
                        if y_prob is not None:
                            global_predictions[cal_level][name]["prob"].extend(y_prob.tolist())
                    except Exception as e:
                        logger.exception(f"        Failed to store global predictions for {name}: {e}")

                    # Subject-level metrics
                    try:
                        cal_subject_results.append({
                            "Calibration_Level": cal_level,
                            "Subject": test_sub,
                            "Model": name,
                            "Accuracy": metrics["Accuracy"],
                            "Precision": metrics["Precision"],
                            "Recall": metrics["Recall"],
                            "Macro_F1": metrics["Macro_F1"],
                            "Weighted_F1": metrics["Weighted_F1"],
                            "MCC": metrics["MCC"],
                            "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                            "ROC_AUC": metrics.get("ROC_AUC", np.nan),
                            "PR_AUC": metrics.get("PR_AUC", np.nan),
                        })
                    except Exception as e:
                        logger.exception(f"        Subject metrics failed for {name}: {e}")

                    # Gesture-wise metrics
                    try:
                        for gest_idx, gest_name in enumerate(GESTURE_SHORT):
                            gest_mask = np.array(y_eval) == gest_idx
                            if np.sum(gest_mask) > 0:
                                gest_true = np.array(y_eval)[gest_mask]
                                gest_pred = preds[gest_mask]
                                gest_acc = accuracy_score(gest_true, gest_pred)
                                gest_f1 = f1_score(gest_true, gest_pred, average="macro", zero_division=0)
                                cal_gesture_results.append({
                                    "Calibration_Level": cal_level,
                                    "Gesture": gest_name,
                                    "Model": name,
                                    "Accuracy": gest_acc,
                                    "F1": gest_f1,
                                })
                    except Exception as e:
                        logger.exception(f"        Gesture metrics failed for {name}: {e}")

                    # Cleanup per-model
                    try:
                        del preds
                        if y_prob is not None:
                            del y_prob
                    except Exception:
                        pass
                    gc.collect()
            
            # Explicit cleanup
            try:
                del X_tr_raw, y_tr, X_te_raw, y_te
                del X_tr_sel, X_cal_sel, X_eval_sel
                del X_tr_final, y_tr_final
                del sc, sv, sk
                
                if "X_cal_raw" in locals():
                    del X_cal_raw
                if "y_cal" in locals():
                    del y_cal
                if "X_eval_raw" in locals():
                    del X_eval_raw
                if "y_eval" in locals():
                    del y_eval
                    
                if "trained_models" in locals():
                    for model in trained_models.values():
                        if hasattr(model, "_Booster"):
                            try:
                                del model._Booster
                            except Exception:
                                pass
                        del model
                    del trained_models
            except Exception as e:
                logger.debug(f"      Cleanup warning: {e}")
            
            gc.collect()
            time.sleep(0.02)
            
            fold_elapsed = time.perf_counter() - fold_start_time
            if fold_i % 5 == 0:
                logger.debug(f"      Fold {fold_i} completed in {fold_elapsed:.1f}s")
        
        # RESUME: Save checkpoint after each calibration level
        completed_levels.append(cal_level)
        _save_calibration_checkpoint(completed_levels, checkpoint_path)
        
        level_elapsed = time.perf_counter() - level_start_time
        total_elapsed = time.perf_counter() - total_start_time
        logger.info(f"    Calibration {int(cal_level*100)}% completed in {level_elapsed/60:.1f} min")
        logger.info(f"    Total elapsed: {total_elapsed/60:.1f} min")
        
        remaining_levels = len(calibration_levels) - level_counter
        if remaining_levels > 0:
            avg_time_per_level = total_elapsed / level_counter
            estimated_remaining = avg_time_per_level * remaining_levels
            logger.info(f"    Estimated time remaining: {estimated_remaining/60:.1f} min")

    # Remove checkpoint after completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info("  Calibration checkpoint removed (successful completion)")
        except Exception as e:
            logger.exception(f"  Failed to remove calibration checkpoint: {e}")

    # Save Calibration Prediction Archive
    if cal_prediction_archive:
        expanded_rows = []
        for entry in cal_prediction_archive:
            for true_val, pred_val in zip(entry["True_Label"], entry["Pred_Label"]):
                expanded_rows.append({
                    "Calibration_Level": entry["Calibration_Level"],
                    "Subject": entry["Subject"],
                    "Fold": entry["Fold"],
                    "Model": entry["Model"],
                    "True_Label": true_val,
                    "Pred_Label": pred_val,
                })
        df_cal_pred_archive = pd.DataFrame(expanded_rows)
        try:
            save_csv(df_cal_pred_archive, OUT_DIR / "PredictionArchive_Calibration.csv")
            logger.info(f"  PredictionArchive_Calibration.csv saved: {len(df_cal_pred_archive)} predictions")
        except Exception as e:
            logger.exception(f"  Failed to save PredictionArchive_Calibration.csv: {e}")

    # ================================================================
    # FIX: Build confusion matrices from PredictionArchive_Calibration.csv
    # AND generate visualizations for every calibration level and model
    # ================================================================
    logger.info("  Building confusion matrices from PredictionArchive_Calibration.csv...")
    
    cal_cm_accum = {}
    cal_cm_0_accum = {}
    
    archive_path = OUT_DIR / "PredictionArchive_Calibration.csv"
    
    if archive_path.exists():
        for level in calibration_levels:
            cal_cm_accum[level] = {}
            level_dir = CALIBRATION_CM_DIR / f"Calibration_{int(level*100)}"
            level_dir.mkdir(parents=True, exist_ok=True)
            
            for model in MODEL_NAMES:
                cm = _build_calibration_confusion_matrices_from_archive(archive_path, level, model)
                cal_cm_accum[level][model] = cm
                
                if level == 0.0:
                    cal_cm_0_accum[model] = cm
                
                # Save CSV files
                if np.sum(cm) > 0:
                    try:
                        cm_path = level_dir / f"cm_{model}.csv"
                        pd.DataFrame(cm, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(cm_path)
                        
                        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                        cm_norm = np.nan_to_num(cm_norm)
                        cm_norm_path = level_dir / f"cm_{model}_normalized.csv"
                        pd.DataFrame(cm_norm, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(cm_norm_path)
                        
                        logger.debug(f"    Saved CM for Calibration {int(level*100)}%, {model}")
                        
                        # ================================================================
                        # NEW: Generate confusion matrix visualization
                        # ================================================================
                        _plot_normalized_confusion_matrix(
                            cm=cm,
                            model_name=model,
                            calibration_level=level,
                            output_dir=level_dir,
                            title_prefix="",  # No prefix for per-level figures
                        )
                        
                    except Exception as e:
                        logger.exception(f"    Failed to save CM for {model} at {int(level*100)}%: {e}")
    else:
        logger.warning("  PredictionArchive_Calibration.csv not found - cannot build confusion matrices from archive")
        # Fallback: use empty CMs
        for level in calibration_levels:
            cal_cm_accum[level] = {m: np.zeros((N_CLASSES, N_CLASSES), dtype=int) for m in MODEL_NAMES}
        cal_cm_0_accum = {m: np.zeros((N_CLASSES, N_CLASSES), dtype=int) for m in MODEL_NAMES}
    
    # ================================================================
    # FIX: Compute Calibration Metrics (ECE, Brier) per calibration level
    # ================================================================
    calibration_metrics = []
    
    for level in calibration_levels:
        for model in MODEL_NAMES:
            if model in global_predictions[level]:
                try:
                    true_vals = np.array(global_predictions[level][model]["true"])
                    probs = np.array(global_predictions[level][model]["prob"])
                    if len(true_vals) > 0 and len(probs) > 0:
                        ece = _compute_global_ece(true_vals, probs)
                        brier = _compute_brier_score_multiclass(true_vals, probs)
                        calibration_metrics.append({
                            "Calibration_Level": level,
                            "Model": model,
                            "ECE": ece,
                            "Brier_Score": brier,
                            "N_Samples": len(true_vals)
                        })
                except Exception as e:
                    logger.exception(f"  Calibration metrics failed for {model} @ {level*100}%: {e}")
    
    df_cal_metrics = pd.DataFrame(calibration_metrics)
    if not df_cal_metrics.empty:
        try:
            save_csv(df_cal_metrics, OUT_DIR / "Table_S31_Calibration_Metrics.csv")
            logger.info(f"  Table S31 saved: Calibration metrics for {len(df_cal_metrics)} rows")
        except Exception as e:
            logger.exception(f"  Failed to save Table S31: {e}")

    # Convert to DataFrames
    df_cal_subject = pd.DataFrame(cal_subject_results)
    if not df_cal_subject.empty:
        try:
            save_csv(df_cal_subject, OUT_DIR / "Table_S15_Calibration_Results_By_Subject.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S15: {e}")
    
    # Model-wise aggregation with CI
    if not df_cal_subject.empty:
        try:
            df_cal_model = df_cal_subject.groupby(["Calibration_Level", "Model"]).agg({
                "Accuracy": ["mean", "std"],
                "Precision": ["mean", "std"],
                "Recall": ["mean", "std"],
                "Macro_F1": ["mean", "std"],
                "Weighted_F1": ["mean", "std"],
                "MCC": ["mean", "std"],
                "Balanced_Accuracy": ["mean", "std"],
                "ROC_AUC": ["mean", "std"],
                "PR_AUC": ["mean", "std"],
            }).reset_index()
            
            df_cal_model.columns = [
                "Calibration_Level", "Model",
                "Accuracy_Mean", "Accuracy_Std",
                "Precision_Mean", "Precision_Std",
                "Recall_Mean", "Recall_Std",
                "F1_Mean", "F1_Std",
                "Weighted_F1_Mean", "Weighted_F1_Std",
                "MCC_Mean", "MCC_Std",
                "Balanced_Accuracy_Mean", "Balanced_Accuracy_Std",
                "ROC_AUC_Mean", "ROC_AUC_Std",
                "PR_AUC_Mean", "PR_AUC_Std"
            ]
            
            # Compute confidence intervals using SEM
            for metric in ["Accuracy", "Precision", "Recall", "F1", "Weighted_F1", "MCC", 
                           "Balanced_Accuracy", "ROC_AUC", "PR_AUC"]:
                try:
                    mean_col = f"{metric}_Mean"
                    std_col = f"{metric}_Std"
                    n = len(df_cal_subject["Subject"].unique())
                    sem = df_cal_model[std_col] / np.sqrt(n)
                    df_cal_model[f"{metric}_CI_Lower"] = df_cal_model[mean_col] - 1.96 * sem
                    df_cal_model[f"{metric}_CI_Upper"] = df_cal_model[mean_col] + 1.96 * sem
                except Exception as e:
                    logger.debug(f"      CI computation failed for {metric}: {e}")
            
            save_csv(df_cal_model, OUT_DIR / "Table_S16_Calibration_Results_By_Model.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S16: {e}")
            df_cal_model = pd.DataFrame()
    else:
        df_cal_model = pd.DataFrame()
    
    # Gesture-wise
    df_cal_gesture = pd.DataFrame(cal_gesture_results)
    if not df_cal_gesture.empty:
        try:
            df_cal_gesture_summary = df_cal_gesture.groupby(["Calibration_Level", "Gesture", "Model"]).agg({
                "Accuracy": "mean",
                "F1": "mean"
            }).reset_index()
            save_csv(df_cal_gesture_summary, OUT_DIR / "Table_S24_Gesture_Calibration_Improvement.csv")
        except Exception as e:
            logger.exception(f"  Failed to save gesture summary: {e}")
    
    # Compute calibration gain
    df_gain = _compute_calibration_gain(df_cal_model)
    if not df_gain.empty:
        try:
            save_csv(df_gain, OUT_DIR / "Table_S25_Calibration_Gain_Summary.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S25: {e}")
    
    # Optimal calibration level
    df_optimal = _find_optimal_calibration_level(df_gain)
    if not df_optimal.empty:
        try:
            save_csv(df_optimal, OUT_DIR / "Table_S29_Optimal_Calibration_Level.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S29: {e}")
    
    # Model calibration ranking
    if not df_optimal.empty:
        df_model_ranking = df_optimal.sort_values("Optimal_F1", ascending=False)
        df_model_ranking["Rank"] = range(1, len(df_model_ranking) + 1)
        df_model_ranking["Score"] = df_model_ranking["Optimal_F1"] / df_model_ranking["Optimal_F1"].max()
        try:
            save_csv(df_model_ranking, OUT_DIR / "Table_S26_Model_Calibration_Ranking.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S26: {e}")
    
    # Subject calibration ranking (same model comparison)
    if not df_cal_subject.empty:
        best_subject_perf = []
        for subject in df_cal_subject["Subject"].unique():
            sub_data = df_cal_subject[df_cal_subject["Subject"] == subject]
            best_row = sub_data.loc[sub_data["Macro_F1"].idxmax()]
            best_model = best_row["Model"]
            best_level = best_row["Calibration_Level"]
            best_f1 = best_row["Macro_F1"]
            best_acc = best_row["Accuracy"]
            best_mcc = best_row["MCC"]
            best_bal_acc = best_row["Balanced_Accuracy"]
            
            baseline_data = sub_data[(sub_data["Model"] == best_model) & (sub_data["Calibration_Level"] == 0.0)]
            baseline_f1 = baseline_data["Macro_F1"].mean() if not baseline_data.empty else 0.0
            
            best_subject_perf.append({
                "Subject": subject,
                "Best_Model": best_model,
                "Best_Calibration_Level": best_level,
                "Accuracy": best_acc,
                "Macro_F1": best_f1,
                "MCC": best_mcc,
                "Balanced_Accuracy": best_bal_acc,
                "Baseline_F1": baseline_f1,
                "F1_Gain": best_f1 - baseline_f1,
            })
        
        df_subject_ranking = pd.DataFrame(best_subject_perf).sort_values("F1_Gain", ascending=False)
        df_subject_ranking["Rank"] = range(1, len(df_subject_ranking) + 1)
        try:
            save_csv(df_subject_ranking, OUT_DIR / "Table_S27_Subject_Calibration_Ranking.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S27: {e}")
    
    # Gesture calibration ranking (same model comparison)
    if not df_cal_gesture.empty:
        best_gesture_perf = []
        for gesture in df_cal_gesture["Gesture"].unique():
            gest_data = df_cal_gesture[df_cal_gesture["Gesture"] == gesture]
            best_row = gest_data.loc[gest_data["F1"].idxmax()]
            best_model = best_row["Model"]
            best_level = best_row["Calibration_Level"]
            best_f1 = best_row["F1"]
            best_acc = best_row["Accuracy"]
            
            baseline_data = gest_data[(gest_data["Model"] == best_model) & (gest_data["Calibration_Level"] == 0.0)]
            baseline_f1 = baseline_data["F1"].mean() if not baseline_data.empty else 0.0
            
            best_gesture_perf.append({
                "Gesture": gesture,
                "Best_Calibration_Level": best_level,
                "Best_F1": best_f1,
                "Best_Accuracy": best_acc,
                "Best_Model": best_model,
                "Baseline_F1": baseline_f1,
                "F1_Gain": best_f1 - baseline_f1,
                "Percent_Improvement": (best_f1 - baseline_f1) / (baseline_f1 + 0.001) * 100 if baseline_f1 > 0 else 0,
                "Adaptation_Score": best_f1 / (baseline_f1 + 0.001) if baseline_f1 > 0 else best_f1,
            })
        
        df_gesture_ranking = pd.DataFrame(best_gesture_perf).sort_values("F1_Gain", ascending=False)
        df_gesture_ranking["Rank"] = range(1, len(df_gesture_ranking) + 1)
        try:
            save_csv(df_gesture_ranking, OUT_DIR / "Table_S28_Gesture_Calibration_Ranking.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S28: {e}")
    
    # Statistical tests across calibration levels
    df_stats = _compute_statistical_tests_calibration(df_cal_subject)
    if not df_stats.empty:
        try:
            save_csv(df_stats, OUT_DIR / "Table_S30_Calibration_Statistical_Comparison.csv")
        except Exception as e:
            logger.exception(f"  Failed to save Table S30: {e}")
    
    # FIX: Fusion readiness with ECE from optimal calibration level
    fusion_readiness = _compute_fusion_readiness(df_optimal, df_gain, df_cal_model, df_cal_metrics)
    if not fusion_readiness.empty:
        try:
            save_csv(fusion_readiness, OUT_DIR / "fusion_readiness_scores.csv")
        except Exception as e:
            logger.exception(f"  Failed to save fusion readiness: {e}")
    
    # Generate difference confusion matrices
    _generate_difference_confusion_matrices(cal_cm_accum, cal_cm_0_accum, df_optimal)
    
    # Generate calibration report
    _generate_calibration_report(df_optimal, fusion_readiness, df_cal_model, 
                                 df_subject_ranking if 'df_subject_ranking' in locals() else pd.DataFrame(),
                                 df_gesture_ranking if 'df_gesture_ranking' in locals() else pd.DataFrame(),
                                 df_stats, df_cal_metrics)
    
    return df_cal_subject, df_cal_model, df_cal_gesture, {
        "gain": df_gain,
        "optimal": df_optimal,
        "model_ranking": df_model_ranking if 'df_model_ranking' in locals() else pd.DataFrame(),
        "subject_ranking": df_subject_ranking if 'df_subject_ranking' in locals() else pd.DataFrame(),
        "gesture_ranking": df_gesture_ranking if 'df_gesture_ranking' in locals() else pd.DataFrame(),
        "fusion": fusion_readiness,
        "calibration_metrics": df_cal_metrics if not df_cal_metrics.empty else pd.DataFrame(),
        "global_predictions": global_predictions,
        "cm_accum": cal_cm_accum,
        "cm_0_accum": cal_cm_0_accum,
    }


def _compute_fusion_readiness(df_optimal, df_gain, df_cal_model, df_cal_metrics) -> pd.DataFrame:
    """Compute fusion readiness scores with improved reliability score."""
    fusion_rows = []
    
    # Try to load BLOCK_4 results for generalization score
    sd_summary_path = BLOCK_DIRS["BLOCK_4"] / "Table5_SubjectDependent_Summary.csv"
    sd_df = None
    if sd_summary_path.exists():
        try:
            sd_df = load_csv(sd_summary_path)
            logger.info(f"    Loaded BLOCK_4 results from {sd_summary_path}")
        except Exception as e:
            logger.warning(f"    Could not load BLOCK_4 results: {e}")
    
    for model in MODEL_NAMES:
        model_optimal = df_optimal[df_optimal["Model"] == model] if not df_optimal.empty else pd.DataFrame()
        if model_optimal.empty:
            continue
        
        row = model_optimal.iloc[0]
        max_gain = df_optimal["F1_Gain"].max() if not df_optimal.empty else 0.001
        calibration_score = row["F1_Gain"] / max_gain if max_gain > 0 else 0.0
        
        # Improved Generalization Score
        baseline_f1 = row["Optimal_F1"] - row["F1_Gain"]
        
        if sd_df is not None and "Model" in sd_df.columns:
            sd_model_data = sd_df[sd_df["Model"] == model]
            if not sd_model_data.empty:
                sd_f1 = sd_model_data["Macro_F1_Mean"].values[0] if "Macro_F1_Mean" in sd_model_data.columns else baseline_f1
                if sd_f1 > 0:
                    generalization_score = min(baseline_f1 / sd_f1, 1.0)
                else:
                    generalization_score = min(baseline_f1 / 0.95, 1.0)
            else:
                generalization_score = min(baseline_f1 / 0.95, 1.0)
        else:
            generalization_score = min(baseline_f1 / 0.95, 1.0)
        
        # Reliability Score using Coefficient of Variation
        model_gain_data = df_gain[df_gain["Model"] == model] if not df_gain.empty else pd.DataFrame()
        if not model_gain_data.empty:
            f1_values = model_gain_data["F1_Value"].values
            mean_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0.0
            cv = std_f1 / (mean_f1 + 1e-10)
            reliability_score = max(0, 1 - min(cv, 1.0))
        else:
            reliability_score = 0.0
        
        # FIX: ECE score from optimal calibration level
        ece_val = 0.5
        if not df_cal_metrics.empty:
            model_metrics = df_cal_metrics[
                (df_cal_metrics["Model"] == model) & 
                (df_cal_metrics["Calibration_Level"] == row["Optimal_Calibration_Level"])
            ]
            if not model_metrics.empty and "ECE" in model_metrics.columns:
                ece_val = model_metrics["ECE"].values[0]
                if np.isnan(ece_val):
                    ece_val = 0.5
        
        ece_score = max(0, 1 - min(ece_val, 1.0))
        
        fusion_rows.append({
            "Model": model,
            "Optimal_Calibration": row["Optimal_Calibration_Level"],
            "Calibration_Score": round(calibration_score, 4),
            "Generalization_Score": round(generalization_score, 4),
            "Reliability_Score": round(reliability_score, 4),
            "ECE_Score": round(ece_score, 4),
            "ECE_Value": round(ece_val, 4),
            "Fusion_Readiness_Score": round((calibration_score + generalization_score + reliability_score + ece_score) / 4, 4),
        })
    
    df_fusion = pd.DataFrame(fusion_rows)
    if not df_fusion.empty:
        df_fusion = df_fusion.sort_values("Fusion_Readiness_Score", ascending=False)
    
    return df_fusion


def _generate_difference_confusion_matrices(
    cal_cm_accum,
    cal_cm_0_accum,
    df_optimal,
):
    """Generate publication-quality normalized difference confusion matrices."""
    logger.info("Generating normalized difference confusion matrices...")

    # Determine best model and optimal calibration level
    best_model = MODEL_NAMES[0]
    optimal_level = max(cal_cm_accum.keys()) if cal_cm_accum else 0.0

    if not df_optimal.empty:
        best_row = df_optimal.loc[df_optimal["Optimal_F1"].idxmax()]
        best_model = best_row["Model"]
        optimal_level = min(
            cal_cm_accum.keys(),
            key=lambda x: abs(x - float(best_row["Optimal_Calibration_Level"])),
        )
        logger.info(f"  Using optimal calibration {int(optimal_level*100)}% for {best_model}")
    else:
        logger.warning("  Table_S29 not found. Using maximum calibration level.")

    def _normalize_cm(cm):
        cm = cm.astype(np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return cm / row_sum

    # Figure 29
    if (optimal_level in cal_cm_accum and best_model in cal_cm_accum.get(optimal_level, {}) 
        and best_model in cal_cm_0_accum):
        try:
            cm_level = cal_cm_accum[optimal_level][best_model]
            cm_0 = cal_cm_0_accum[best_model]
            cm_level_norm = _normalize_cm(cm_level)
            cm_0_norm = _normalize_cm(cm_0)
            cm_diff = cm_level_norm - cm_0_norm

            level_dir = CALIBRATION_CM_DIR / f"Calibration_{int(optimal_level*100)}"
            level_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(cm_diff, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(
                level_dir / f"cm_{best_model}_difference_normalized.csv"
            )

            vmax = max(np.max(np.abs(cm_diff)), 1e-6)
            fig, ax = plt.subplots(figsize=(8.5, 7))
            sns.heatmap(
                cm_diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
                square=True, xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                cbar_kws={"label": "Change in Classification Rate\n(Row-normalized)"},
                ax=ax,
            )
            ax.set_xlabel("Predicted Gesture")
            ax.set_ylabel("True Gesture")
            ax.set_title(f"Figure 29 – Normalized Difference Confusion Matrix\n{best_model} (Optimal {int(optimal_level*100)}% Calibration vs. 0% Baseline)")
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig29_Normalized_Difference_Confusion_Matrix_Heatmap.png")
            plt.close(fig)
            gc.collect()
            logger.info("  Figure 29 generated successfully.")
        except Exception as e:
            logger.exception(f"  Figure 29 generation failed: {e}")
    else:
        logger.warning(f"  Figure 29 skipped. Optimal Level: {optimal_level}, Best Model: {best_model}")

    # Difference matrices for every calibration level
    for level in CALIBRATION_LEVELS:
        if level == 0:
            continue

        level_dir = CALIBRATION_CM_DIR / f"Calibration_{int(level*100)}"
        level_dir.mkdir(parents=True, exist_ok=True)

        for model in MODEL_NAMES:
            if (level not in cal_cm_accum or model not in cal_cm_accum[level] 
                or model not in cal_cm_0_accum):
                continue

            try:
                cm_level = cal_cm_accum[level][model]
                cm_0 = cal_cm_0_accum[model]
                cm_diff = _normalize_cm(cm_level) - _normalize_cm(cm_0)

                pd.DataFrame(cm_diff, index=GESTURE_SHORT, columns=GESTURE_SHORT).to_csv(
                    level_dir / f"cm_{model}_difference_normalized.csv"
                )

                vmax = max(np.max(np.abs(cm_diff)), 1e-6)
                fig, ax = plt.subplots(figsize=(8.5, 7))
                sns.heatmap(
                    cm_diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
                    square=True, xticklabels=GESTURE_SHORT, yticklabels=GESTURE_SHORT,
                    cbar_kws={"label": "Change in Classification Rate\n(Row-normalized)"},
                    ax=ax,
                )
                ax.set_xlabel("Predicted Gesture")
                ax.set_ylabel("True Gesture")
                ax.set_title(f"Normalized Difference Confusion Matrix\n{model} : {int(level*100)}% vs. 0%")
                plt.tight_layout()
                save_fig(fig, level_dir / f"cm_{model}_difference_heatmap_normalized.png")
                plt.close(fig)
                gc.collect()
            except Exception as e:
                logger.exception(f"  Difference matrix failed for {model} at {int(level*100)}%: {e}")

    logger.info("  Normalized difference confusion matrices generated.")


def _generate_calibration_curves(calibration_results: Dict) -> None:
    """Generate Reliability Diagram (Figure 27) and Calibration Curve (Figure 28)."""
    global_predictions = calibration_results.get("global_predictions", {})
    df_optimal = calibration_results.get("optimal", pd.DataFrame())
    
    # Determine which calibration level to use for reliability diagrams
    # Use the optimal calibration level for the best model
    target_level = 0.0
    
    if not df_optimal.empty:
        best_row = df_optimal.loc[df_optimal["Optimal_F1"].idxmax()]
        target_level = float(best_row["Optimal_Calibration_Level"])
        logger.info(f"  Generating reliability curves for optimal calibration level {int(target_level*100)}%")
    
    # Figure 27: Reliability Diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in MODEL_NAMES:
        if target_level not in global_predictions or model not in global_predictions[target_level]:
            continue
            
        level_data = global_predictions[target_level][model]
        true_vals = np.array(level_data["true"])
        probs = np.array(level_data["prob"])
        
        if len(true_vals) == 0 or len(probs) == 0:
            # Try to use aggregated predictions across all levels
            all_true = []
            all_probs = []
            for level in global_predictions:
                if model in global_predictions[level]:
                    all_true.extend(global_predictions[level][model]["true"])
                    all_probs.extend(global_predictions[level][model]["prob"])
            
            if len(all_true) > 0 and len(all_probs) > 0:
                true_vals = np.array(all_true)
                probs = np.array(all_probs)
            else:
                continue
        
        try:
            mean_predicted, fraction_positives = _generate_multiclass_reliability_curve(
                true_vals, probs, n_bins=10
            )
            if len(mean_predicted) > 0 and len(fraction_positives) > 0:
                ax.plot(mean_predicted, fraction_positives, marker='o', lw=2, label=model,
                       color=MODEL_COLORS.get(model, '#888'), markersize=8,
                       markeredgecolor='black', markeredgewidth=0.5)
        except Exception as e:
            logger.warning(f"  Could not generate reliability curve for {model}: {e}")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Averaged across classes)")
    ax.set_ylabel("Observed Fraction of Positives (Averaged across classes)")
    ax.set_title(f"Figure 27 – Reliability Diagram (Optimal Calibration: {int(target_level*100)}%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig27_Reliability_Diagram.png")
    logger.info("  Figure 27 saved.")
    plt.close(fig)
    
    # Figure 28: Calibration Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in MODEL_NAMES:
        if target_level not in global_predictions or model not in global_predictions[target_level]:
            continue
            
        level_data = global_predictions[target_level][model]
        true_vals = np.array(level_data["true"])
        probs = np.array(level_data["prob"])
        
        if len(true_vals) == 0 or len(probs) == 0:
            continue
        
        try:
            mean_predicted, fraction_positives = _generate_multiclass_reliability_curve(
                true_vals, probs, n_bins=10
            )
            if len(mean_predicted) > 0 and len(fraction_positives) > 0:
                ax.plot(mean_predicted, fraction_positives, marker='s', lw=2, label=model,
                       color=MODEL_COLORS.get(model, '#888'), markersize=8,
                       markeredgecolor='black', markeredgewidth=0.5)
        except Exception as e:
            logger.warning(f"  Could not generate calibration curve for {model}: {e}")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Averaged across classes)")
    ax.set_ylabel("Observed Fraction of Positives (Averaged across classes)")
    ax.set_title(f"Figure 28 – Calibration Curve (Optimal Calibration: {int(target_level*100)}%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    save_fig(fig, OUT_DIR / "Fig28_Calibration_Curve.png")
    logger.info("  Figure 28 saved.")
    plt.close(fig)


def _generate_gain_heatmaps(df_cal_subject, df_cal_model, df_cal_gesture):
    """Generate Macro-F1 gain heatmaps relative to the 0% calibration baseline."""
    
    # Figure 30 – Subject-wise Macro-F1 Gain Heatmap
    if not df_cal_subject.empty:
        try:
            pivot = df_cal_subject.pivot_table(
                index="Subject", columns="Calibration_Level", values="Macro_F1", aggfunc="mean"
            )
            if not pivot.empty:
                baseline = pivot[0.0] if 0.0 in pivot.columns else 0
                gain_pivot = pivot.subtract(baseline, axis=0)
                fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.4)))
                sns.heatmap(
                    gain_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    linewidths=0.5, linecolor="white",
                    xticklabels=[f"{int(c*100)}%" for c in gain_pivot.columns],
                    cbar_kws={"label": "Macro-F1 Gain"}, ax=ax,
                )
                ax.set_title("Figure 30 – Subject-wise Macro-F1 Gain Heatmap\n(Relative to 0% Calibration)", fontsize=13, fontweight="bold")
                ax.set_xlabel("Calibration Level (%)")
                ax.set_ylabel("Subject")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig30_Subject_F1_Gain_Heatmap.png")
                logger.info("  Figure 30 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 30 generation failed: {e}")
    
    # Figure 31 – Model-wise Macro-F1 Gain Heatmap
    if not df_cal_model.empty and "F1_Mean" in df_cal_model.columns:
        try:
            pivot = df_cal_model.pivot_table(index="Model", columns="Calibration_Level", values="F1_Mean")
            if not pivot.empty:
                baseline = pivot[0.0] if 0.0 in pivot.columns else 0
                gain_pivot = pivot.subtract(baseline, axis=0)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    gain_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    linewidths=0.5, linecolor="white",
                    xticklabels=[f"{int(c*100)}%" for c in gain_pivot.columns],
                    cbar_kws={"label": "Macro-F1 Gain"}, ax=ax,
                )
                ax.set_title("Figure 31 – Model-wise Macro-F1 Gain Heatmap\n(Relative to 0% Calibration)", fontsize=13, fontweight="bold")
                ax.set_xlabel("Calibration Level (%)")
                ax.set_ylabel("Model")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig31_Model_F1_Gain_Heatmap.png")
                logger.info("  Figure 31 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 31 generation failed: {e}")
    
    # Figure 32 – Gesture-wise Macro-F1 Gain Heatmap
    if not df_cal_gesture.empty and "F1" in df_cal_gesture.columns:
        try:
            pivot = df_cal_gesture.pivot_table(
                index="Gesture", columns="Calibration_Level", values="F1", aggfunc="mean"
            )
            if not pivot.empty:
                baseline = pivot[0.0] if 0.0 in pivot.columns else 0
                gain_pivot = pivot.subtract(baseline, axis=0)
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
                sns.heatmap(
                    gain_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    linewidths=0.5, linecolor="white",
                    xticklabels=[f"{int(c*100)}%" for c in gain_pivot.columns],
                    cbar_kws={"label": "Macro-F1 Gain"}, ax=ax,
                )
                ax.set_title("Figure 32 – Gesture-wise Macro-F1 Gain Heatmap\n(Relative to 0% Calibration)", fontsize=13, fontweight="bold")
                ax.set_xlabel("Calibration Level (%)")
                ax.set_ylabel("Gesture")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig32_Gesture_F1_Gain_Heatmap.png")
                logger.info("  Figure 32 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 32 generation failed: {e}")


def _generate_calibration_report(df_optimal, df_fusion, df_cal_model, 
                                 df_subject_ranking, df_gesture_ranking, 
                                 df_stats, df_cal_metrics):
    """Generate comprehensive calibration_report.txt."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CALIBRATION ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    
    # Best model and calibration
    if not df_optimal.empty:
        best_row = df_optimal.loc[df_optimal["Optimal_F1"].idxmax()]
        report_lines.append("BEST PERFORMING MODEL (with Calibration)")
        report_lines.append("-" * 50)
        report_lines.append(f"  Model: {best_row['Model']}")
        report_lines.append(f"  Optimal Calibration Level: {int(best_row['Optimal_Calibration_Level']*100)}%")
        report_lines.append(f"  Best Macro-F1: {best_row['Optimal_F1']:.4f}")
        report_lines.append(f"  Best Accuracy: {best_row['Optimal_Accuracy']:.4f}")
        report_lines.append(f"  Best MCC: {best_row['Optimal_MCC']:.4f}")
        report_lines.append(f"  Best ROC-AUC: {best_row.get('Optimal_ROC_AUC', np.nan):.4f}")
        report_lines.append(f"  Best PR-AUC: {best_row.get('Optimal_PR_AUC', np.nan):.4f}")
        report_lines.append(f"  F1 Gain from 0%: {best_row['F1_Gain']:.4f}")
        report_lines.append(f"  Saturation Level: {int(best_row['Saturation_Level']*100)}%")
        report_lines.append("")
        
        # Calibration metrics for best model at optimal level
        if not df_cal_metrics.empty:
            best_metrics = df_cal_metrics[
                (df_cal_metrics["Model"] == best_row["Model"]) & 
                (df_cal_metrics["Calibration_Level"] == best_row["Optimal_Calibration_Level"])
            ]
            if not best_metrics.empty:
                report_lines.append("  CALIBRATION METRICS (at optimal level)")
                report_lines.append("  " + "-" * 35)
                report_lines.append(f"    Global ECE: {best_metrics['ECE'].values[0]:.4f}")
                report_lines.append(f"    Brier Score: {best_metrics['Brier_Score'].values[0]:.4f}")
                report_lines.append(f"    N Samples: {best_metrics['N_Samples'].values[0]}")
                report_lines.append("")
        
        # Model ranking
        report_lines.append("MODEL CALIBRATION RANKING (by Macro-F1)")
        report_lines.append("-" * 50)
        for i, (_, row) in enumerate(df_optimal.sort_values("Optimal_F1", ascending=False).iterrows(), 1):
            report_lines.append(f"  {i}. {row['Model']}: {row['Optimal_F1']:.4f} @ {int(row['Optimal_Calibration_Level']*100)}% (Gain: {row['F1_Gain']:.4f})")
        report_lines.append("")
    
    # Fusion readiness
    if not df_fusion.empty:
        report_lines.append("FUSION READINESS SCORES")
        report_lines.append("-" * 50)
        for _, row in df_fusion.iterrows():
            report_lines.append(f"  {row['Model']}: {row['Fusion_Readiness_Score']:.3f} (Cal:{row['Calibration_Score']:.3f}, Gen:{row['Generalization_Score']:.3f}, Rel:{row['Reliability_Score']:.3f}, ECE:{row['ECE_Score']:.3f})")
        report_lines.append("")
        report_lines.append(f"  Best Fusion Candidate: {df_fusion.iloc[0]['Model']} (Score: {df_fusion.iloc[0]['Fusion_Readiness_Score']:.3f})")
        report_lines.append("")
    
    # Subject ranking
    if not df_subject_ranking.empty:
        report_lines.append("TOP 5 SUBJECTS (by F1 Gain after calibration)")
        report_lines.append("-" * 50)
        top_subjects = df_subject_ranking.head(5)
        for _, row in top_subjects.iterrows():
            report_lines.append(f"  {row['Subject']}: Gain {row['F1_Gain']:.4f} (Best F1: {row['Macro_F1']:.4f}, Model: {row['Best_Model']}, Cal: {int(row['Best_Calibration_Level']*100)}%)")
        report_lines.append("")
    
    # Gesture ranking
    if not df_gesture_ranking.empty:
        report_lines.append("GESTURE CALIBRATION GAIN RANKING")
        report_lines.append("-" * 50)
        top_gestures = df_gesture_ranking.head(5)
        for _, row in top_gestures.iterrows():
            report_lines.append(f"  {row['Gesture']}: Gain {row['F1_Gain']:.4f} ({row['Percent_Improvement']:.1f}% improvement, Best F1: {row['Best_F1']:.4f} @ {int(row['Best_Calibration_Level']*100)}%)")
        report_lines.append("")
    
    # Statistical test results
    if not df_stats.empty:
        report_lines.append("STATISTICAL TEST RESULTS")
        report_lines.append("-" * 50)
        for _, row in df_stats.iterrows():
            if row["Test"] == "Friedman_Calibration_Levels":
                report_lines.append(f"  Friedman Test: χ²={row['Statistic']:.4f}, p={row['P_Value']:.6f}, Significant: {row['Significant']}")
            elif row["Test"] == "Nemenyi_Ranking":
                report_lines.append(f"  Nemenyi Ranking - Level {row['Calibration_Level']*100:.0f}%: Rank {row['Mean_Rank']:.2f}")
        report_lines.append("")
    
    # Recommended calibration
    report_lines.append("RECOMMENDED CALIBRATION SETTINGS")
    report_lines.append("-" * 50)
    if not df_optimal.empty:
        best_row = df_optimal.loc[df_optimal["Optimal_F1"].idxmax()]
        report_lines.append(f"  Best Model: {best_row['Model']}")
        report_lines.append(f"  Recommended Calibration: {int(best_row['Optimal_Calibration_Level']*100)}%")
        report_lines.append(f"  Saturation Point: {int(best_row['Saturation_Level']*100)}%")
        report_lines.append("")
        report_lines.append(f"  Expected Macro-F1: {best_row['Optimal_F1']:.4f}")
        report_lines.append(f"  Expected Accuracy: {best_row['Optimal_Accuracy']:.4f}")
        report_lines.append(f"  Expected MCC: {best_row['Optimal_MCC']:.4f}")
        report_lines.append(f"  Expected ROC-AUC: {best_row.get('Optimal_ROC_AUC', np.nan):.4f}")
        report_lines.append(f"  Expected PR-AUC: {best_row.get('Optimal_PR_AUC', np.nan):.4f}")
        if not df_cal_metrics.empty:
            best_metrics = df_cal_metrics[
                (df_cal_metrics["Model"] == best_row["Model"]) & 
                (df_cal_metrics["Calibration_Level"] == best_row["Optimal_Calibration_Level"])
            ]
            if not best_metrics.empty:
                report_lines.append(f"  Expected ECE: {best_metrics['ECE'].values[0]:.4f}")
                report_lines.append(f"  Expected Brier Score: {best_metrics['Brier_Score'].values[0]:.4f}")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    report_path = OUT_DIR / "calibration_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        logger.info(f"  Calibration report saved → {report_path}")
    except Exception as e:
        logger.exception(f"  Failed to save calibration report: {e}")


def _compute_ws_ol_sensitivity_analysis(
    grid_results_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute real Window Size and Overlap sensitivity analysis."""
    logger.info("  Computing Window Size × Overlap Sensitivity Analysis...")
    
    df_ws = pd.DataFrame()
    df_ol = pd.DataFrame()
    df_3d = pd.DataFrame()
    
    if grid_results_path is None:
        grid_results_path = BLOCK_DIRS["BLOCK_0"] / "grid_search_results.csv"
    
    if grid_results_path.exists():
        try:
            df_grid = pd.read_csv(grid_results_path)
            logger.info(f"    Loaded grid results from {grid_results_path}")
            
            ws_analysis = []
            for ws in WINDOW_SIZES_GRID:
                ws_data = df_grid[df_grid["WS"] == ws]
                if not ws_data.empty:
                    mean_val, ci_lower, ci_upper = _compute_confidence_interval(ws_data["Accuracy"].values)
                    ws_analysis.append({
                        "Window_Size": ws,
                        "Mean_Accuracy": mean_val,
                        "Std_Accuracy": ws_data["Accuracy"].std(),
                        "CI_Lower": ci_lower,
                        "CI_Upper": ci_upper,
                        "Min_Accuracy": ws_data["Accuracy"].min(),
                        "Max_Accuracy": ws_data["Accuracy"].max(),
                        "Sensitivity_Range": ws_data["Accuracy"].max() - ws_data["Accuracy"].min(),
                    })
            df_ws = pd.DataFrame(ws_analysis)
            if not df_ws.empty:
                save_csv(df_ws, OUT_DIR / "Table_S21_WindowSize_Sensitivity.csv")
                logger.info(f"    Table S21 saved: {len(df_ws)} rows")
            
            ol_analysis = []
            for ol in OVERLAPS_GRID:
                ol_data = df_grid[df_grid["Overlap"] == ol]
                if not ol_data.empty:
                    mean_val, ci_lower, ci_upper = _compute_confidence_interval(ol_data["Accuracy"].values)
                    ol_analysis.append({
                        "Overlap": f"{int(ol*100)}%",
                        "Overlap_Value": ol,
                        "Mean_Accuracy": mean_val,
                        "Std_Accuracy": ol_data["Accuracy"].std(),
                        "CI_Lower": ci_lower,
                        "CI_Upper": ci_upper,
                        "Min_Accuracy": ol_data["Accuracy"].min(),
                        "Max_Accuracy": ol_data["Accuracy"].max(),
                        "Sensitivity_Range": ol_data["Accuracy"].max() - ol_data["Accuracy"].min(),
                    })
            df_ol = pd.DataFrame(ol_analysis)
            if not df_ol.empty:
                save_csv(df_ol, OUT_DIR / "Table_S22_Overlap_Sensitivity.csv")
                logger.info(f"    Table S22 saved: {len(df_ol)} rows")
            
            if len(WINDOW_SIZES_GRID) > 0 and len(OVERLAPS_GRID) > 0:
                pivot_data = df_grid.pivot_table(index="WS", columns="Overlap", values="Accuracy")
                if not pivot_data.empty:
                    df_3d = pivot_data.reset_index().melt(
                        id_vars=["WS"], var_name="Overlap", value_name="Accuracy"
                    )
                    df_3d["Overlap_Label"] = df_3d["Overlap"].apply(lambda x: f"{int(x*100)}%")
                    df_3d = df_3d.rename(columns={"WS": "Window_Size"})
                    if not df_3d.empty:
                        save_csv(df_3d, OUT_DIR / "Table_S23_WS_OL_Sensitivity.csv")
                        logger.info(f"    Table S23 saved: {len(df_3d)} rows")
        except Exception as e:
            logger.exception(f"    Could not compute WS/OL analysis: {e}")
    
    # Fallback empty DataFrames
    if df_ws.empty:
        df_ws = pd.DataFrame({
            "Window_Size": WINDOW_SIZES_GRID,
            "Mean_Accuracy": [np.nan] * len(WINDOW_SIZES_GRID),
            "Std_Accuracy": [np.nan] * len(WINDOW_SIZES_GRID),
            "Sensitivity_Range": [np.nan] * len(WINDOW_SIZES_GRID)
        })
        save_csv(df_ws, OUT_DIR / "Table_S21_WindowSize_Sensitivity.csv")
    
    if df_ol.empty:
        df_ol = pd.DataFrame({
            "Overlap": [f"{int(ol*100)}%" for ol in OVERLAPS_GRID],
            "Overlap_Value": OVERLAPS_GRID,
            "Mean_Accuracy": [np.nan] * len(OVERLAPS_GRID),
            "Std_Accuracy": [np.nan] * len(OVERLAPS_GRID),
            "Sensitivity_Range": [np.nan] * len(OVERLAPS_GRID)
        })
        save_csv(df_ol, OUT_DIR / "Table_S22_Overlap_Sensitivity.csv")
    
    if df_3d.empty:
        data = []
        for ws in WINDOW_SIZES_GRID:
            for ol in OVERLAPS_GRID:
                data.append({
                    "Window_Size": ws,
                    "Overlap": ol,
                    "Overlap_Label": f"{int(ol*100)}%",
                    "Accuracy": np.nan
                })
        df_3d = pd.DataFrame(data)
        save_csv(df_3d, OUT_DIR / "Table_S23_WS_OL_Sensitivity.csv")
    
    return df_ws, df_ol, df_3d


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def _generate_figures_loso(
    df_folds: pd.DataFrame,
    df_subjects: pd.DataFrame,
    df_summary: pd.DataFrame
) -> None:
    """Generate Figures 10-15."""
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_folds, x="Model", y="Accuracy", order=MODEL_NAMES,
                    errorbar="sd", palette=PALETTE, hue="Model", legend=False, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Figure 10 – LOSO Performance Comparison (Mean ± SD)")
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig10_LOSO_Performance_Comparison.png")
        logger.info("  Figure 10 saved.")
        plt.close(fig)
    except Exception as e:
        logger.exception(f"  Figure 10 generation failed: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_folds, x="Model", y="Accuracy", order=MODEL_NAMES,
                    palette=PALETTE, width=0.5, ax=ax)
        sns.stripplot(data=df_folds, x="Model", y="Accuracy", order=MODEL_NAMES,
                      color="black", size=3.5, alpha=0.55, jitter=True, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Per-Fold Accuracy")
        ax.set_title("Figure 11 – LOSO Fold Distribution")
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig11_LOSO_Fold_Distribution.png")
        logger.info("  Figure 11 saved.")
        plt.close(fig)
    except Exception as e:
        logger.exception(f"  Figure 11 generation failed: {e}")

    metric_fig_map = [
        ("Accuracy", 12, "Figure 12 – Subject-Wise Accuracy Distribution"),
        ("Macro_F1", 13, "Figure 13 – Subject-Wise F1 Distribution"),
        ("MCC", 14, "Figure 14 – Subject-Wise MCC Distribution"),
    ]
    for metric, fig_no, title in metric_fig_map:
        if metric not in df_subjects.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=(max(10, len(df_subjects["Subject"].unique()) * 0.8), 5))
            sns.barplot(data=df_subjects, x="Subject", y=metric, hue="Model",
                        palette=PALETTE, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            save_fig(fig, OUT_DIR / f"Fig{fig_no}_SubjectWise_{metric}.png")
            logger.info(f"  Figure {fig_no} saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure {fig_no} generation failed: {e}")

    try:
        fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(4 * len(MODEL_NAMES), 5), sharey=True)
        for ax, model in zip(axes, MODEL_NAMES):
            sub_df = df_subjects[df_subjects["Model"] == model]
            if sub_df.empty:
                continue
            mean_acc = sub_df["Accuracy"].mean()
            std_acc = sub_df["Accuracy"].std()
            ax.bar(range(len(sub_df)), sub_df["Accuracy"].values,
                   color=MODEL_COLORS[model], alpha=0.8, edgecolor="white")
            ax.axhline(mean_acc, color="red", lw=1.8, ls="--",
                       label=f"μ={mean_acc:.3f}\nσ={std_acc:.3f}")
            ax.set_xticks(range(len(sub_df)))
            ax.set_xticklabels(sub_df["Subject"].values, rotation=90, fontsize=7)
            ax.set_title(model, fontsize=10)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2, axis="y")
        fig.suptitle("Figure 15 – Inter-Subject Variability Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, OUT_DIR / "Fig15_InterSubject_Variability.png")
        logger.info("  Figure 15 saved.")
        plt.close(fig)
    except Exception as e:
        logger.exception(f"  Figure 15 generation failed: {e}")


def _generate_figures_calibration(
    df_cal_model: pd.DataFrame,
    df_cal_subject: pd.DataFrame,
    df_cal_gesture: pd.DataFrame,
    df_ws: pd.DataFrame,
    df_ol: pd.DataFrame,
    df_3d: pd.DataFrame,
    calibration_results: Dict
) -> None:
    """Generate Figures 16-32 and S14-S16."""
    
    # Figures 16-18: Accuracy/F1/MCC vs Calibration Level
    if not df_cal_model.empty:
        metric_map = [
            ("Accuracy_Mean", "Accuracy", 16, "Figure 16 – Accuracy vs Calibration Level"),
            ("F1_Mean", "Macro-F1", 17, "Figure 17 – Macro-F1 vs Calibration Level"),
            ("MCC_Mean", "MCC", 18, "Figure 18 – MCC vs Calibration Level"),
        ]
        
        for metric_col, metric_name, fig_no, title in metric_map:
            if metric_col not in df_cal_model.columns:
                continue
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                for model in MODEL_NAMES:
                    model_data = df_cal_model[df_cal_model["Model"] == model]
                    if model_data.empty:
                        continue
                    ax.plot(model_data["Calibration_Level"] * 100, model_data[metric_col],
                            marker='o', lw=2, label=model, color=MODEL_COLORS[model])
                ax.set_xlabel("Calibration Level (%)")
                ax.set_ylabel(metric_name)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_fig(fig, OUT_DIR / f"Fig{fig_no}_{metric_name}_vs_Calibration.png")
                logger.info(f"  Figure {fig_no} saved.")
                plt.close(fig)
            except Exception as e:
                logger.exception(f"  Figure {fig_no} generation failed: {e}")
    
    # Figure 19 – Subject Calibration Heatmap
    if not df_cal_subject.empty:
        try:
            pivot = df_cal_subject.pivot_table(
                index="Subject", columns="Calibration_Level", values="Accuracy", aggfunc="mean"
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                           xticklabels=[f"{int(c*100)}%" for c in pivot.columns],
                           ax=ax, cbar_kws={"label": "Accuracy"})
                ax.set_title("Figure 19 – Subject Calibration Heatmap", fontsize=13, fontweight="bold")
                ax.set_xlabel("Calibration Level")
                ax.set_ylabel("Subject")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig19_Subject_Calibration_Heatmap.png")
                logger.info("  Figure 19 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 19 generation failed: {e}")
    
    # Figure 20 – Model Calibration Heatmap
    if not df_cal_model.empty:
        try:
            pivot = df_cal_model.pivot_table(index="Model", columns="Calibration_Level", values="F1_Mean")
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                           xticklabels=[f"{int(c*100)}%" for c in pivot.columns],
                           ax=ax, cbar_kws={"label": "F1 Score"})
                ax.set_title("Figure 20 – Model Calibration Heatmap", fontsize=13, fontweight="bold")
                ax.set_xlabel("Calibration Level")
                ax.set_ylabel("Model")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig20_Model_Calibration_Heatmap.png")
                logger.info("  Figure 20 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 20 generation failed: {e}")
    
    # Figure 21 – Gesture Calibration Improvement Curves
    if not df_cal_gesture.empty:
        try:
            gest_curves = []
            for gest in df_cal_gesture["Gesture"].unique():
                gest_data = df_cal_gesture[df_cal_gesture["Gesture"] == gest]
                level_means = gest_data.groupby("Calibration_Level")["F1"].mean()
                if len(level_means) > 1:
                    baseline = level_means[0.0] if 0.0 in level_means else 0.0
                    improvements = []
                    levels = sorted(level_means.index)
                    for level in levels:
                        improvements.append(level_means[level] - baseline)
                    
                    max_level = max(level_means.index)
                    gest_curves.append({
                        "Gesture": gest,
                        "Levels": [l * 100 for l in levels],
                        "Improvements": improvements,
                        "Total_Improvement": level_means[max_level] - level_means[0.0] if 0.0 in level_means else 0.0
                    })
            
            if gest_curves:
                gest_curves = sorted(gest_curves, key=lambda x: x["Total_Improvement"], reverse=True)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                for gc in gest_curves:
                    ax.plot(gc["Levels"], gc["Improvements"], marker='o', lw=2, label=gc["Gesture"])
                
                ax.set_xlabel("Calibration Level (%)")
                ax.set_ylabel("F1 Improvement (Gain from 0%)")
                ax.set_title("Figure 21 – Gesture Calibration Improvement Curves")
                ax.legend(loc="best", fontsize=8)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "Fig21_Gesture_Calibration_Improvement.png")
                logger.info("  Figure 21 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 21 generation failed: {e}")
    
    # Figure 22 – Calibration Gain Curve
    df_gain = calibration_results.get("gain", pd.DataFrame())
    if not df_gain.empty:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for model in MODEL_NAMES:
                model_gain = df_gain[df_gain["Model"] == model]
                if not model_gain.empty:
                    ax.plot(model_gain["Calibration_Level"] * 100, model_gain["F1_Gain"],
                            marker='o', lw=2, label=model, color=MODEL_COLORS[model])
            ax.set_xlabel("Calibration Level (%)")
            ax.set_ylabel("F1 Gain (from 0% calibration)")
            ax.set_title("Figure 22 – Calibration Gain Curve (Macro-F1)")
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig22_Calibration_Gain_Curve.png")
            logger.info("  Figure 22 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 22 generation failed: {e}")
    
    # Figure 23 – Calibration Saturation Curve
    df_optimal = calibration_results.get("optimal", pd.DataFrame())
    if not df_optimal.empty:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for model in MODEL_NAMES:
                model_opt = df_optimal[df_optimal["Model"] == model]
                if not model_opt.empty:
                    row = model_opt.iloc[0]
                    ax.scatter(row["Saturation_Level"] * 100, row["F1_at_Saturation"],
                              s=100, label=model, color=MODEL_COLORS[model])
                    ax.scatter(row["Optimal_Calibration_Level"] * 100, row["Optimal_F1"],
                              s=200, marker='*', color=MODEL_COLORS[model],
                              edgecolor='black', linewidth=1)
            ax.set_xlabel("Calibration Level (%)")
            ax.set_ylabel("Macro-F1")
            ax.set_title("Figure 23 – Calibration Saturation Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig23_Calibration_Saturation_Curve.png")
            logger.info("  Figure 23 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 23 generation failed: {e}")
    
    # Figure 24 – Subject Gain Distribution
    df_subject_ranking = calibration_results.get("subject_ranking", pd.DataFrame())
    if not df_subject_ranking.empty:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            gains = df_subject_ranking["F1_Gain"].values
            ax.hist(gains, bins=20, alpha=0.7, color='#4CAF50', edgecolor='black')
            ax.axvline(np.mean(gains), color='red', linestyle='--', label=f"Mean Gain: {np.mean(gains):.3f}")
            ax.set_xlabel("F1 Gain")
            ax.set_ylabel("Number of Subjects")
            ax.set_title("Figure 24 – Subject Gain Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig24_Subject_Gain_Distribution.png")
            logger.info("  Figure 24 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 24 generation failed: {e}")
    
    # Figure 25 – Gesture Gain Distribution
    df_gesture_ranking = calibration_results.get("gesture_ranking", pd.DataFrame())
    if not df_gesture_ranking.empty and "F1_Gain" in df_gesture_ranking.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df_gesture_ranking["Gesture"], df_gesture_ranking["F1_Gain"],
                    color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax.set_xlabel("F1 Gain")
            ax.set_title("Figure 25 – Gesture Gain Distribution")
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig25_Gesture_Gain_Distribution.png")
            logger.info("  Figure 25 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 25 generation failed: {e}")
    
    # Figure 26 – Model Calibration Ranking
    df_model_ranking = calibration_results.get("model_ranking", pd.DataFrame())
    if not df_model_ranking.empty:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [MODEL_COLORS.get(m, '#888') for m in df_model_ranking["Model"]]
            ax.barh(df_model_ranking["Model"], df_model_ranking["Score"],
                    color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel("Normalized Calibration Score (Macro-F1 based)")
            ax.set_title("Figure 26 – Model Calibration Ranking")
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "Fig26_Model_Calibration_Ranking.png")
            logger.info("  Figure 26 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure 26 generation failed: {e}")
    
    # Figures S14-S16: WS/OL Sensitivity
    if not df_ws.empty and "Mean_Accuracy" in df_ws.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_ws["Window_Size"], df_ws["Mean_Accuracy"], 
                   color='#4CAF50', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Window Size (samples)")
            ax.set_ylabel("Mean Accuracy")
            ax.set_title("Figure S14 – Window Size Sensitivity Analysis")
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "FigS14_WindowSize_Sensitivity.png")
            logger.info("  Figure S14 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure S14 generation failed: {e}")
    
    if not df_ol.empty and "Mean_Accuracy" in df_ol.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_ol["Overlap"], df_ol["Mean_Accuracy"],
                   color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Overlap (%)")
            ax.set_ylabel("Mean Accuracy")
            ax.set_title("Figure S15 – Overlap Sensitivity Analysis")
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            save_fig(fig, OUT_DIR / "FigS15_Overlap_Sensitivity.png")
            logger.info("  Figure S15 saved.")
            plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure S15 generation failed: {e}")
    
    if not df_3d.empty:
        try:
            pivot = df_3d.pivot_table(index="Window_Size", columns="Overlap", values="Accuracy", aggfunc="mean")
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                           xticklabels=[f"{int(c*100)}%" for c in pivot.columns] if hasattr(pivot.columns, '__iter__') else None,
                           ax=ax, cbar_kws={"label": "Accuracy"})
                ax.set_title("Figure S16 – Window Size × Overlap Sensitivity Heatmap")
                ax.set_xlabel("Overlap")
                ax.set_ylabel("Window Size")
                plt.tight_layout()
                save_fig(fig, OUT_DIR / "FigS16_WS_OL_Sensitivity_Heatmap.png")
                logger.info("  Figure S16 saved.")
                plt.close(fig)
        except Exception as e:
            logger.exception(f"  Figure S16 generation failed: {e}")
    
    # Generate calibration curves and gain heatmaps
    _generate_calibration_curves(calibration_results)
    _generate_gain_heatmaps(df_cal_subject, df_cal_model, df_cal_gesture)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_block3() -> Dict:
    """Main entry point for BLOCK 3."""
    logger.info("=" * 70)
    logger.info("BLOCK 3 | GEN1: Subject-Independent Generalization and Calibration Framework")
    logger.info("=" * 70)
    
    logger.info(f"Python : {sys.version}")
    logger.info(f"Numpy  : {np.__version__}")
    logger.info(f"Pandas : {pd.__version__}")
    logger.info(f"PID    : {os.getpid()}")
    logger.info(f"Calibration Levels: {[int(c*100) for c in CALIBRATION_LEVELS]}%")
    
    checkpoint_loso = OUT_DIR / "loso_checkpoint.json"
    checkpoint_cal = OUT_DIR / "calibration_checkpoint.json"
    
    if checkpoint_loso.exists():
        logger.debug(f"  LOSO checkpoint exists: {checkpoint_loso}")
    if checkpoint_cal.exists():
        logger.debug(f"  Calibration checkpoint exists: {checkpoint_cal}")

    # Try to load from BLOCK_2 artifacts
    try:
        X, y, subjects, feature_names = load_feature_matrix()
        logger.info(f"  Loaded {len(X)} windows from BLOCK_2 artifacts")
        logger.debug(f"    X shape: {X.shape}, y shape: {y.shape}, subjects: {len(np.unique(subjects))}")
    except (FileNotFoundError, KeyError, TypeError) as e:
        logger.warning(f"  BLOCK_2 artifacts not found: {e}")
        logger.info("  Falling back to raw data loading...")
        X, y, subjects, feature_names = _load_fallback_data()
        logger.debug(f"    Fallback X shape: {X.shape}, subjects: {len(np.unique(subjects))}")

    # GEN-01 to GEN-04: LOSO Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("GEN-01 to GEN-04: LOSO Performance Evaluation")
    logger.info("=" * 60)
    
    try:
        df_folds, df_fold_stats, df_subjects = run_loso(X, y, subjects)
    except Exception as e:
        logger.exception(f"  LOSO evaluation failed: {e}")
        raise
    
    try:
        save_csv(df_folds, OUT_DIR / "Table_S5_PerFold_LOSO.csv")
        logger.info(f"  Table S5 saved: {len(df_folds)} rows")
    except Exception as e:
        logger.exception(f"  Failed to save Table S5: {e}")
    
    df_summary = _summarise_results(df_folds.to_dict("records"))
    try:
        save_csv(df_summary, OUT_DIR / "Table3_LOSO_Summary.csv")
        logger.info(f"  Table 3 saved: {len(df_summary)} models")
    except Exception as e:
        logger.exception(f"  Failed to save Table 3: {e}")
    
    # Log the best LOSO model
    best_loso_model = df_summary.loc[df_summary["Accuracy_Mean"].idxmax(), "Model"]
    best_loso_acc = df_summary["Accuracy_Mean"].max()
    logger.info(f"  Best LOSO Model: {best_loso_model} @ {best_loso_acc:.4f}")
    logger.info("\n" + df_summary[["Model", "Accuracy_Mean", "Accuracy_SD",
                                   "Macro_F1_Mean", "MCC_Mean"]].to_string(index=False))

    # GEN-05 to GEN-08: Calibration Analysis
    logger.info("\n" + "=" * 60)
    logger.info("GEN-05 to GEN-08: Calibration Performance Analysis")
    logger.info("=" * 60)
    
    try:
        df_cal_subject, df_cal_model, df_cal_gesture, calibration_results = run_calibration_analysis(X, y, subjects)
    except Exception as e:
        logger.exception(f"  Calibration analysis failed: {e}")
        raise
    
    # WS/OL Sensitivity Analysis
    grid_results_path = BLOCK_DIRS["BLOCK_0"] / "grid_search_results.csv"
    df_ws, df_ol, df_3d = _compute_ws_ol_sensitivity_analysis(grid_results_path)
    
    # Figures
    logger.info("\n" + "=" * 60)
    logger.info("Generating Figures")
    logger.info("=" * 60)
    try:
        _generate_figures_loso(df_folds, df_subjects, df_summary)
        _generate_figures_calibration(df_cal_model, df_cal_subject, df_cal_gesture, df_ws, df_ol, df_3d, calibration_results)
    except Exception as e:
        logger.exception(f"  Figure generation failed: {e}")

    # Artifact Manifest
    artifact_map: Dict[str, Path] = {
        "table_s5": OUT_DIR / "Table_S5_PerFold_LOSO.csv",
        "table_s5b": OUT_DIR / "Table_S5b_Fold_Statistics.csv",
        "table3": OUT_DIR / "Table3_LOSO_Summary.csv",
        "table4": OUT_DIR / "Table4_SubjectWise_Metrics.csv",
        "prediction_archive_loso": OUT_DIR / "PredictionArchive_LOSO.csv",
        "prediction_archive_calibration": OUT_DIR / "PredictionArchive_Calibration.csv",
        "table_s15": OUT_DIR / "Table_S15_Calibration_Results_By_Subject.csv",
        "table_s16": OUT_DIR / "Table_S16_Calibration_Results_By_Model.csv",
        "table_s21": OUT_DIR / "Table_S21_WindowSize_Sensitivity.csv",
        "table_s22": OUT_DIR / "Table_S22_Overlap_Sensitivity.csv",
        "table_s23": OUT_DIR / "Table_S23_WS_OL_Sensitivity.csv",
        "table_s24": OUT_DIR / "Table_S24_Gesture_Calibration_Improvement.csv",
        "table_s25": OUT_DIR / "Table_S25_Calibration_Gain_Summary.csv",
        "table_s26": OUT_DIR / "Table_S26_Model_Calibration_Ranking.csv",
        "table_s27": OUT_DIR / "Table_S27_Subject_Calibration_Ranking.csv",
        "table_s28": OUT_DIR / "Table_S28_Gesture_Calibration_Ranking.csv",
        "table_s29": OUT_DIR / "Table_S29_Optimal_Calibration_Level.csv",
        "table_s30": OUT_DIR / "Table_S30_Calibration_Statistical_Comparison.csv",
        "table_s31": OUT_DIR / "Table_S31_Calibration_Metrics.csv",
        "fig10": OUT_DIR / "Fig10_LOSO_Performance_Comparison.png",
        "fig11": OUT_DIR / "Fig11_LOSO_Fold_Distribution.png",
        "fig12": OUT_DIR / "Fig12_SubjectWise_Accuracy.png",
        "fig13": OUT_DIR / "Fig13_SubjectWise_F1.png",
        "fig14": OUT_DIR / "Fig14_SubjectWise_MCC.png",
        "fig15": OUT_DIR / "Fig15_InterSubject_Variability.png",
        "fig16": OUT_DIR / "Fig16_Accuracy_vs_Calibration.png",
        "fig17": OUT_DIR / "Fig17_F1_vs_Calibration.png",
        "fig18": OUT_DIR / "Fig18_MCC_vs_Calibration.png",
        "fig19": OUT_DIR / "Fig19_Subject_Calibration_Heatmap.png",
        "fig20": OUT_DIR / "Fig20_Model_Calibration_Heatmap.png",
        "fig21": OUT_DIR / "Fig21_Gesture_Calibration_Improvement.png",
        "fig22": OUT_DIR / "Fig22_Calibration_Gain_Curve.png",
        "fig23": OUT_DIR / "Fig23_Calibration_Saturation_Curve.png",
        "fig24": OUT_DIR / "Fig24_Subject_Gain_Distribution.png",
        "fig25": OUT_DIR / "Fig25_Gesture_Gain_Distribution.png",
        "fig26": OUT_DIR / "Fig26_Model_Calibration_Ranking.png",
        "fig27": OUT_DIR / "Fig27_Reliability_Diagram.png",
        "fig28": OUT_DIR / "Fig28_Calibration_Curve.png",
        "fig29": OUT_DIR / "Fig29_Normalized_Difference_Confusion_Matrix_Heatmap.png",
        "fig30": OUT_DIR / "Fig30_Subject_F1_Gain_Heatmap.png",
        "fig31": OUT_DIR / "Fig31_Model_F1_Gain_Heatmap.png",
        "fig32": OUT_DIR / "Fig32_Gesture_F1_Gain_Heatmap.png",
        "fig_s14": OUT_DIR / "FigS14_WindowSize_Sensitivity.png",
        "fig_s15": OUT_DIR / "FigS15_Overlap_Sensitivity.png",
        "fig_s16": OUT_DIR / "FigS16_WS_OL_Sensitivity_Heatmap.png",
        "calibration_report": OUT_DIR / "calibration_report.txt",
        "fusion_readiness": OUT_DIR / "fusion_readiness_scores.csv",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    best_calibration_model = "N/A"
    best_calibration_level = 0
    best_calibration_f1 = 0.0
    
    if not calibration_results["optimal"].empty:
        try:
            best_calibration_model = str(calibration_results["optimal"].loc[calibration_results["optimal"]["Optimal_F1"].idxmax(), "Model"])
            best_calibration_level = int(calibration_results["optimal"].loc[calibration_results["optimal"]["Optimal_F1"].idxmax(), "Optimal_Calibration_Level"]*100)
            best_calibration_f1 = float(calibration_results["optimal"]["Optimal_F1"].max())
        except Exception as e:
            logger.exception(f"  Failed to extract best calibration results: {e}")

    summary = {
        "n_folds": int(df_folds["Fold"].max()),
        "n_subjects": int(len(np.unique(subjects))),
        "best_model": str(df_summary.loc[df_summary["Accuracy_Mean"].idxmax(), "Model"]),
        "best_acc": float(df_summary["Accuracy_Mean"].max()),
        "best_calibration_model": best_calibration_model,
        "best_calibration_level": best_calibration_level,
        "best_calibration_f1": best_calibration_f1,
        "n_windows": int(len(X)),
        "calibration_levels": [int(c*100) for c in CALIBRATION_LEVELS],
        "checkpoint_loso_used": checkpoint_loso.exists(),
        "checkpoint_cal_used": checkpoint_cal.exists(),
        "pipeline": PIPELINE_NAME,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("BLOCK 3 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best LOSO: {summary['best_model']} @ {summary['best_acc']:.4f}")
    logger.info(f"  Best Calibration: {summary['best_calibration_model']} @ {summary['best_calibration_level']}% (F1: {summary['best_calibration_f1']:.4f})")
    logger.info(f"  Total folds: {summary['n_folds']}")
    logger.info(f"  Total subjects: {summary['n_subjects']}")
    logger.info(f"  Calibration levels: {summary['calibration_levels']}")
    if checkpoint_loso.exists():
        logger.info(f"  ⚠ LOSO checkpoint still exists (should have been removed)")
    if checkpoint_cal.exists():
        logger.info(f"  ⚠ Calibration checkpoint still exists (should have been removed)")
    logger.info("=" * 60)
    
    return summary


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return

    try:
        result = run_block3()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.error(f"❌  {BLOCK_NAME} FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        gc.collect()


if __name__ == "__main__":
    main()