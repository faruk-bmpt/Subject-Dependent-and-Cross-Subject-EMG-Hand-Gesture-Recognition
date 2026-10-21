# BLOCK_4_PER1.py - COMPLETE UPDATED VERSION WITH PREDICTION ARCHIVE
# ================
# FILE PURPOSE:
#     BLOCK 4 — PER1: Personalized Gesture Recognition and Subject-Dependent Performance Framework.
#     Estimates the upper-bound performance achievable when training and testing on the same subject.
#
#     EVALUATION PROTOCOL:
#     - Subject-Dependent: 5-fold stratified cross-validation per subject
#     - Strict separation: No cross-subject contamination
#     - Personalization Gain: LOSO (BLOCK_3) vs Subject-Dependent comparison
#
#     PER MODULES:
#     - PER-01: Subject-Dependent Performance Evaluation
#     - PER-02: Subject-Wise Performance Analysis
#     - PER-03: Model-Wise Personalized Comparison
#     - PER-04: Personalization Gain Analysis
#     - PER-05: Subject Variability Analysis
#     - PER-06: Gesture-Wise Personalization Analysis
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (raw CSV files) [fallback if BLOCK_2 not run]
#     - outputs/BLOCK_2/X_features.npy (optional)
#     - outputs/BLOCK_2/y_labels.npy (optional)
#     - outputs/BLOCK_2/subjects.npy (optional)
#     - outputs/BLOCK_3/Table3_LOSO_Summary.csv (for LOSO comparison)
#     - outputs/BLOCK_3/loso_cm_*.csv (for gesture-wise LOSO accuracy)
#     - outputs/BLOCK_3/LOSO_best_cm.csv (best model LOSO confusion matrix)
#
# OUTPUT_ARTIFACTS :
#     - outputs/BLOCK_4/Table5_SubjectDependent_Summary.csv
#     - outputs/BLOCK_4/Table_S17_SubjectWise_Model_Comparison.csv
#     - outputs/BLOCK_4/Table_S18_SubjectDependent_Fold_Results.csv
#     - outputs/BLOCK_4/Table_S19_Personalization_Gain_Analysis.csv
#     - outputs/BLOCK_4/Table_S20_Gesture_Personalization_Improvement.csv
#     - outputs/BLOCK_4/Table_S21_Subject_Ranking.csv
#     - outputs/BLOCK_4/Table_S22_Best_Model_Per_Subject.csv
#     - outputs/BLOCK_4/Table_S23_Subject_Personalization_Gain.csv
#     - outputs/BLOCK_4/Table_S24_Subject_Performance_Summary.csv
#     - outputs/BLOCK_4/PredictionArchive_SubjectDependent.csv
#     - outputs/BLOCK_4/Fig21_SubjectDependent_Comparison.png
#     - outputs/BLOCK_4/Fig22_SubjectDependent_Accuracy_Distribution.png
#     - outputs/BLOCK_4/Fig23_SubjectWise_Performance_Heatmap.png
#     - outputs/BLOCK_4/Fig24_Personalization_Gain_Analysis.png
#     - outputs/BLOCK_4/Fig25_Subject_Ranking_by_Accuracy.png
#     - outputs/BLOCK_4/Fig26_Gesture_Personalization_Gain.png
#     - outputs/BLOCK_4/FigS17_SubjectWise_Model_Comparison_Heatmap.png
#     - outputs/BLOCK_4/FigS18_Subject_Variability_Distribution.png
#     - outputs/BLOCK_4/FigS19_FoldLevel_SubjectDependent_Performance.png
#     - outputs/BLOCK_4/subject_cm/ (Subject-wise confusion matrices per model)
#     - outputs/BLOCK_4/model_cm/ (Model-wise confusion matrices)
#     - outputs/BLOCK_4/fold_cm/ (Fold-level confusion matrices)
#     - outputs/BLOCK_4/subject_predictions/ (Prediction archives)
#     - outputs/BLOCK_4/environment_versions.json
#     - outputs/BLOCK_4/artifacts.json
#     - state/BLOCK_4_PER1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py, common.py
# STANDALONE RUN   : python BLOCK_4_PER1.py

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import os
# FIX: Use single thread for all math libraries to prevent segfault
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
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             matthews_corrcoef, precision_score, recall_score,
                             precision_recall_fscore_support, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
import sklearn
import xgboost

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
    VARIANCE_THRESHOLD,
    MIN_RMS_THRESHOLD,
    SMOTE_K_NEIGHBORS,
    WINDOW_SIZE,
    STEP_SIZE,
    INPUT_PATH,
    K_FEATURES,
    WINDOW_SIZES_GRID,
    OVERLAPS_GRID,
    LDA_PARAMS,
    KNN_PARAMS,
    SVM_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
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
    print("WARNING: Using fallback filter parameters (not found in config)")

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
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_4_PER1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_4")

BLOCK_NAME = "BLOCK_4_PER1"
OUT_DIR    = BLOCK_DIRS["BLOCK_4"]

# ── Archive directories ──────────────────────────────────────────────────────
SUBJECT_CM_DIR = OUT_DIR / "subject_cm"
MODEL_CM_DIR = OUT_DIR / "model_cm"
FOLD_CM_DIR = OUT_DIR / "fold_cm"
PREDICTIONS_DIR = OUT_DIR / "subject_predictions"

for d in [SUBJECT_CM_DIR, MODEL_CM_DIR, FOLD_CM_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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
    if len(data) < 50:
        return data
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
                except Exception:
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


def _save_environment_versions():
    """Save package versions for reproducibility."""
    env_info = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "python": sys.version,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    # Try to get shap version if available
    try:
        import shap
        env_info["shap"] = shap.__version__
    except ImportError:
        env_info["shap"] = "not_installed"
    
    save_json(env_info, OUT_DIR / "environment_versions.json")
    logger.info("  Environment versions saved.")


def _build_model(model_name: str):
    """
    FIX 1: Centralized model builder using config.py parameters.
    """
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAClf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    models = {
        "LDA": LDAClf(**LDA_PARAMS),
        "KNN": KNeighborsClassifier(**KNN_PARAMS),
        "SVM": SVC(**SVM_PARAMS),
        "RF": RandomForestClassifier(**RF_PARAMS),
        "XGB": XGBClassifier(**XGB_PARAMS),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]


def _build_feature_pipeline(X_tr, y_tr):
    """Build feature pipeline: scaling -> variance threshold -> KBest."""
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

    scaler = RobustScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)
    sel_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD).fit(X_s)
    X_v = sel_var.transform(X_s)
    k_safe = min(K_FEATURES, X_v.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_kb = SelectKBest(f_classif, k=k_safe).fit(X_v, y_tr)
    X_sel = sel_kb.transform(X_v)
    return X_sel, scaler, sel_var, sel_kb


def _apply_feature_pipeline(X, scaler, sel_var, sel_kb):
    """Apply fitted feature pipeline to test data."""
    return sel_kb.transform(sel_var.transform(scaler.transform(X)))


def _apply_smote(X, y, tag=""):
    """Apply SMOTE if minority class has enough samples."""
    from imblearn.over_sampling import SMOTE

    min_samples = min(Counter(y).values())

    if min_samples > SMOTE_K_NEIGHBORS:
        try:
            sm = SMOTE(
                random_state=SEED,
                k_neighbors=SMOTE_K_NEIGHBORS,
            )
            X, y = sm.fit_resample(X, y)
        except Exception as exc:
            logger.warning(f"  SMOTE failed {tag}: {exc}")
    else:
        logger.warning(
            f"  SMOTE skipped {tag} "
            f"(min_samples={min_samples}, "
            f"requires > {SMOTE_K_NEIGHBORS})"
        )

    return X, y


def _compute_metrics(y_true, y_pred):
    """
    Compute comprehensive classification metrics.
    
    Improvement 1: Use official balanced_accuracy_score.
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean, lower CI, upper CI using SEM.
    CI = Mean ± 1.96 * SEM
    """
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


def _summarise_results(rows, metric_cols=None):
    """Aggregate fold-level rows into mean ± SD per model with CI."""
    if metric_cols is None:
        metric_cols = ["Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "MCC", "Balanced_Accuracy"]
    
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


# ============================================================================
# FIX 3: Reusable Confusion Matrix Helper
# ============================================================================

def save_confusion_matrix(
    cm: np.ndarray,
    output_dir: Path,
    base_name: str,
    labels: List[str] = None,
    plot_heatmap: bool = True,
    cmap: str = 'Blues',
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = None,
) -> Dict[str, Path]:
    """
    Improvement 2 & 3: Reusable function to save confusion matrix in multiple formats.
    Removed normalize parameter - always saves both raw and normalized.
    Saves all outputs in the same directory (no separate NORMALIZED_CM_DIR).
    
    Args:
        cm: Confusion matrix (N_CLASSES x N_CLASSES)
        output_dir: Directory to save outputs
        base_name: Base filename (e.g., "Subject_S01_RF")
        labels: List of class labels (default: GESTURE_SHORT)
        plot_heatmap: Whether to generate heatmap PNG
        cmap: Colormap for heatmap
        vmin: Minimum value for heatmap (default: 0.0)
        vmax: Maximum value for heatmap (default: 1.0)
        title: Optional title for heatmap
    
    Returns:
        Dict mapping artifact names to Paths
    """
    if labels is None:
        labels = GESTURE_SHORT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # Save raw CM
    raw_path = output_dir / f"{base_name}_CM.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(raw_path)
    saved_files[f"{base_name}_CM"] = raw_path
    
    # Save normalized CM (row-wise)
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm_norm / row_sum
    
    norm_path = output_dir / f"{base_name}_CM_Normalized.csv"
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(norm_path)
    saved_files[f"{base_name}_CM_Normalized"] = norm_path
    
    # Save heatmap PNG (normalized)
    if plot_heatmap:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Proportion (Row-wise)"},
            ax=ax
        )
        ax.set_xlabel("Predicted Gesture")
        ax.set_ylabel("True Gesture")
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        heatmap_path = output_dir / f"{base_name}_CM_Heatmap.png"
        save_fig(fig, heatmap_path)
        plt.close(fig)
        saved_files[f"{base_name}_CM_Heatmap"] = heatmap_path
    
    return saved_files


# ============================================================================
# PER-01 to PER-03: Subject-Dependent Performance
# ============================================================================

def run_subject_dependent(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    PER-01 to PER-03: Subject-dependent evaluation with stratified cross-validation per subject.
    
    Improvement 4: Removed global_prob (not used later).
    Improvement 5: Moved gc.collect() to after each fold and after each subject.
    NEW: Saves PredictionArchive_SubjectDependent.csv with all predictions.
    """
    logger.info("  PER-01 to PER-03: Subject-Dependent Performance Evaluation...")
    
    unique_subs = np.unique(subjects)
    all_rows = []
    subject_metrics = defaultdict(list)
    
    # Archives
    subject_cm_accum = {}
    model_cm_accum = defaultdict(lambda: np.zeros((N_CLASSES, N_CLASSES), dtype=int))
    fold_predictions = {}
    
    # Store ALL predictions globally for gesture-wise analysis
    global_true = []
    global_pred = []
    
    # NEW: Prediction archive for Subject-Dependent (matching BLOCK_3 design)
    prediction_archive: List[Dict] = []
    
    # Store model metadata
    metadata_entries = []
    
    for sub_idx, sub in enumerate(unique_subs):
        logger.info(f"    Processing subject {sub_idx+1}/{len(unique_subs)}: {sub}")
        
        sub_mask = (subjects == sub)
        X_sub = X[sub_mask]
        y_sub = y[sub_mask]
        
        if len(X_sub) < 20:
            logger.warning(f"    {sub}: too few samples ({len(X_sub)}), skipping")
            continue
        
        # Initialize subject CM accumulator per model
        subject_cm_accum[sub] = {m: np.zeros((N_CLASSES, N_CLASSES), dtype=int) for m in MODEL_NAMES}
        
        # Robust StratifiedKFold with min_class_count
        class_counts = np.bincount(y_sub)
        class_counts_nonzero = class_counts[class_counts > 0]
        min_class_count = np.min(class_counts_nonzero)
        n_splits = min(n_folds, min_class_count)
        
        if n_splits < 2:
            logger.warning(f"    {sub}: min_class_count={min_class_count}, using train_test_split instead")
            train_idx, test_idx = train_test_split(
                np.arange(len(X_sub)), test_size=0.2, stratify=y_sub, random_state=SEED
            )
            folds = [(train_idx, test_idx)]
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            folds = list(skf.split(X_sub, y_sub))
        
        for fold_i, (train_idx, test_idx) in enumerate(folds, 1):
            logger.info(f"      Subject {sub} Fold {fold_i}/{len(folds)}")
            
            X_tr_raw, y_tr = X_sub[train_idx], y_sub[train_idx]
            X_te_raw, y_te = X_sub[test_idx], y_sub[test_idx]
            
            # Feature pipeline
            X_tr_sel, sc, sv, sk = _build_feature_pipeline(X_tr_raw, y_tr)
            X_te_sel = _apply_feature_pipeline(X_te_raw, sc, sv, sk)
            
            # SMOTE
            X_tr_final, y_tr_final = _apply_smote(X_tr_sel, y_tr, tag=f"{sub}_fold{fold_i}")
            
            # Train models
            fold_meta = {"Subject": sub, "Fold": fold_i}
            
            for model_name in MODEL_NAMES:
                logger.info(f"        Training {model_name}...")
                
                try:
                    clf = _build_model(model_name)
                    clf.fit(X_tr_final, y_tr_final)
                    logger.info(f"        {model_name} finished training")
                except Exception as e:
                    logger.error(f"        {model_name} crashed: {e}")
                    continue
                
                # Prediction
                try:
                    preds = clf.predict(X_te_sel)
                except Exception as e:
                    logger.error(f"        {model_name} prediction failed: {e}")
                    continue
                
                # Probability estimation
                y_prob = None
                if hasattr(clf, "predict_proba"):
                    try:
                        y_prob = clf.predict_proba(X_te_sel)
                    except Exception:
                        pass
                
                # Metrics
                metrics = _compute_metrics(y_te, preds)
                
                # NEW: Store predictions in archive
                prediction_archive.append({
                    "Subject": sub,
                    "Fold": fold_i,
                    "Model": model_name,
                    "True_Label": y_te.tolist(),
                    "Pred_Label": preds.tolist(),
                    "Probabilities": y_prob.tolist() if y_prob is not None else None,
                })
                
                # Store metadata with each confusion matrix
                metadata_entries.append({
                    "Subject": sub,
                    "Fold": fold_i,
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "Macro_F1": metrics["Macro_F1"],
                    "Weighted_F1": metrics["Weighted_F1"],
                    "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                    "MCC": metrics["MCC"],
                    "N_Samples": len(y_te),
                    "N_Train": len(y_tr),
                })
                
                # Accumulate confusion matrices
                cm = confusion_matrix(y_te, preds, labels=list(range(N_CLASSES)))
                subject_cm_accum[sub][model_name] += cm
                model_cm_accum[model_name] += cm
                
                # Save per-fold CM with normalized version
                fold_base = f"Fold_{sub}_F{fold_i:02d}_{model_name}"
                save_confusion_matrix(
                    cm,
                    FOLD_CM_DIR,
                    fold_base,
                    labels=GESTURE_SHORT,
                    plot_heatmap=True,
                    cmap='Blues',
                    vmin=0.0,
                    vmax=1.0,
                    title=f"Fold {fold_i} - {sub} - {model_name}\nSubject-Dependent"
                )
                
                # Store fold-level predictions
                fold_key = f"{sub}_Fold_{fold_i}_{model_name}"
                fold_predictions[fold_key] = {
                    "true": y_te.tolist(),
                    "pred": preds.tolist(),
                    "prob": y_prob.tolist() if y_prob is not None else None,
                    "subject": sub,
                    "fold": fold_i,
                    "model": model_name
                }
                
                # Save predictions with probabilities
                pred_path = PREDICTIONS_DIR / f"{fold_key}_predictions.csv"
                df_pred = pd.DataFrame({
                    "True": y_te,
                    "Predicted": preds,
                    "Subject": sub,
                    "Fold": fold_i,
                    "Model": model_name
                })
                # Add probabilities if available
                if y_prob is not None:
                    for i in range(y_prob.shape[1]):
                        df_pred[f"Prob_Class_{i}"] = y_prob[:, i]
                df_pred.to_csv(pred_path, index=False)
                
                # Store ALL predictions globally
                global_true.extend(y_te.tolist())
                global_pred.extend(preds.tolist())
                
                # Store metrics
                subject_metrics[sub].append({
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "Macro_F1": metrics["Macro_F1"],
                    "MCC": metrics["MCC"],
                    "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                })
                
                all_rows.append({
                    "Model": model_name,
                    "Fold": fold_i,
                    "Test_Subject": sub,
                    **metrics
                })
                
                # Cleanup model
                del clf
            
            # Improvement 5: gc.collect() after each fold
            gc.collect()
            time.sleep(0.05)
        
        # Save subject-wise confusion matrices per model with normalized version
        for model_name, cm in subject_cm_accum[sub].items():
            if np.sum(cm) > 0:
                subject_base = f"Subject_{sub}_{model_name}"
                save_confusion_matrix(
                    cm,
                    SUBJECT_CM_DIR,
                    subject_base,
                    labels=GESTURE_SHORT,
                    plot_heatmap=True,
                    cmap='Blues',
                    vmin=0.0,
                    vmax=1.0,
                    title=f"Subject {sub} - {model_name}\nSubject-Dependent"
                )
        
        # Improvement 5: gc.collect() after each subject
        gc.collect()
        time.sleep(0.3)
        logger.info(f"      Subject {sub} completed")
    
    # Save model-wise confusion matrices with normalized version
    for model_name, cm in model_cm_accum.items():
        if np.sum(cm) > 0:
            model_base = f"Model_{model_name}"
            save_confusion_matrix(
                cm,
                MODEL_CM_DIR,
                model_base,
                labels=GESTURE_SHORT,
                plot_heatmap=True,
                cmap='Blues',
                vmin=0.0,
                vmax=1.0,
                title=f"{model_name} - Subject-Dependent\nAggregated Across All Subjects"
            )
    
    # ================================================================
    # NEW: Save Prediction Archive (matching BLOCK_3 design)
    # ================================================================
    if prediction_archive:
        expanded_rows = []
        for entry in prediction_archive:
            subject = entry["Subject"]
            fold = entry["Fold"]
            model = entry["Model"]
            true_labels = entry["True_Label"]
            pred_labels = entry["Pred_Label"]
            probs = entry["Probabilities"]
            
            for i, (true_val, pred_val) in enumerate(zip(true_labels, pred_labels)):
                row = {
                    "Subject": subject,
                    "Fold": fold,
                    "Model": model,
                    "True_Label": true_val,
                    "Pred_Label": pred_val,
                }
                # Add probabilities if available
                if probs is not None and i < len(probs):
                    for j, prob in enumerate(probs[i]):
                        row[f"Prob_Class_{j}"] = prob
                expanded_rows.append(row)
        
        df_pred_archive = pd.DataFrame(expanded_rows)
        
        # Fill NaN for probability columns if any missing
        if df_pred_archive.empty:
            logger.warning("  Prediction archive is empty")
        else:
            # Ensure all probability columns exist
            prob_cols = [f"Prob_Class_{i}" for i in range(N_CLASSES)]
            for col in prob_cols:
                if col not in df_pred_archive.columns:
                    df_pred_archive[col] = np.nan
            
            # Reorder columns: Subject, Fold, Model, True_Label, Pred_Label, Prob_Class_*
            base_cols = ["Subject", "Fold", "Model", "True_Label", "Pred_Label"]
            prob_cols = [f"Prob_Class_{i}" for i in range(N_CLASSES) if f"Prob_Class_{i}" in df_pred_archive.columns]
            col_order = base_cols + prob_cols
            df_pred_archive = df_pred_archive[col_order]
            
            try:
                save_csv(df_pred_archive, OUT_DIR / "PredictionArchive_SubjectDependent.csv")
                logger.info(f"  PredictionArchive_SubjectDependent.csv saved: {len(df_pred_archive)} predictions")
            except Exception as e:
                logger.exception(f"  Failed to save PredictionArchive_SubjectDependent.csv: {e}")
    
    # Save metadata
    df_metadata = pd.DataFrame(metadata_entries)
    if not df_metadata.empty:
        df_metadata = df_metadata.sort_values(["Subject", "Model", "Fold"])
        save_csv(df_metadata, OUT_DIR / "model_metadata.csv")
        logger.info("  Model metadata saved.")
    
    # DataFrames
    df_folds = pd.DataFrame(all_rows)
    save_csv(df_folds, OUT_DIR / "Table_S18_SubjectDependent_Fold_Results.csv")
    
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
                })
    df_subjects = pd.DataFrame(sub_rows)
    save_csv(df_subjects, OUT_DIR / "Table_S17_SubjectWise_Model_Comparison.csv")
    
    # Deterministic sorting
    df_subjects = df_subjects.sort_values(
        ["Subject", "Model"],
        ascending=[True, True]
    )
    
    # Model summary
    df_summary = _summarise_results(all_rows)
    save_csv(df_summary, OUT_DIR / "Table5_SubjectDependent_Summary.csv")
    
    archives = {
        "subject_cm": subject_cm_accum,
        "model_cm": model_cm_accum,
        "fold_predictions": fold_predictions,
        "global_true": global_true,
        "global_pred": global_pred,
        "prediction_archive_df": df_pred_archive if 'df_pred_archive' in locals() else pd.DataFrame(),
    }
    
    return df_folds, df_subjects, df_summary, archives


# ============================================================================
# PER-04: Personalization Gain Analysis
# ============================================================================

def run_personalization_gain(
    df_subjects: pd.DataFrame,
    df_subject_loso: Optional[pd.DataFrame] = None,
    loso_summary_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PER-04: Compute personalization gain (Subject-Dependent vs LOSO).
    """
    logger.info("  PER-04: Personalization Gain Analysis...")
    
    # Load LOSO results if available
    loso_df = None
    if loso_summary_path is None:
        loso_summary_path = BLOCK_DIRS["BLOCK_3"] / "Table3_LOSO_Summary.csv"
    
    if loso_summary_path.exists():
        try:
            loso_df = pd.read_csv(loso_summary_path)
            logger.info(f"    Loaded LOSO results from {loso_summary_path}")
        except Exception as e:
            logger.warning(f"    Could not load LOSO results: {e}")
    
    # Load subject-wise LOSO results if available
    loso_subject_df = None
    loso_subject_path = BLOCK_DIRS["BLOCK_3"] / "Table4_SubjectWise_Metrics.csv"
    if loso_subject_path.exists():
        try:
            loso_subject_df = pd.read_csv(loso_subject_path)
            logger.info(f"    Loaded subject-wise LOSO results from {loso_subject_path}")
        except Exception as e:
            logger.warning(f"    Could not load subject-wise LOSO results: {e}")
    
    # Aggregate subject-dependent per model (across subjects)
    sd_by_model = df_subjects.groupby("Model").agg({
        "Accuracy": "mean",
        "Macro_F1": "mean",
        "MCC": "mean",
        "Balanced_Accuracy": "mean"
    }).reset_index()
    sd_by_model.columns = ["Model", "SD_Accuracy", "SD_F1", "SD_MCC", "SD_Balanced_Accuracy"]
    
    # Model-wise gain
    model_gain_rows = []
    if loso_df is not None:
        merged = sd_by_model.merge(loso_df, on="Model", how="inner")
        for _, row in merged.iterrows():
            acc_gain = row["SD_Accuracy"] - row["Accuracy_Mean"]
            model_gain_rows.append({
                "Model": row["Model"],
                "LOSO_Accuracy": row["Accuracy_Mean"],
                "SD_Accuracy": row["SD_Accuracy"],
                "Accuracy_Gain": acc_gain,
                "Accuracy_Gain_Pct": (acc_gain / (row["Accuracy_Mean"] + 1e-10)) * 100,
                "LOSO_F1": row.get("Macro_F1_Mean", 0),
                "SD_F1": row["SD_F1"],
                "F1_Gain": row["SD_F1"] - row.get("Macro_F1_Mean", 0),
                "LOSO_MCC": row.get("MCC_Mean", 0),
                "SD_MCC": row["SD_MCC"],
                "MCC_Gain": row["SD_MCC"] - row.get("MCC_Mean", 0),
            })
    else:
        for _, row in sd_by_model.iterrows():
            model_gain_rows.append({
                "Model": row["Model"],
                "LOSO_Accuracy": np.nan,
                "SD_Accuracy": row["SD_Accuracy"],
                "Accuracy_Gain": np.nan,
                "Accuracy_Gain_Pct": np.nan,
                "LOSO_F1": np.nan,
                "SD_F1": row["SD_F1"],
                "F1_Gain": np.nan,
                "LOSO_MCC": np.nan,
                "SD_MCC": row["SD_MCC"],
                "MCC_Gain": np.nan,
            })
    
    df_model_gain = pd.DataFrame(model_gain_rows)
    df_model_gain = df_model_gain.sort_values(["Accuracy_Gain", "Model"], ascending=[False, True])
    save_csv(df_model_gain, OUT_DIR / "Table_S19_Personalization_Gain_Analysis.csv")
    
    # Subject-wise gain (actual gain = SD - LOSO)
    subject_gain_rows = []
    for sub in df_subjects["Subject"].unique():
        sub_data = df_subjects[df_subjects["Subject"] == sub]
        # Get best model for this subject (by accuracy)
        best_row = sub_data.loc[sub_data["Accuracy"].idxmax()]
        best_model = best_row["Model"]
        sd_acc = best_row["Accuracy"]
        sd_f1 = best_row["Macro_F1"]
        sd_mcc = best_row["MCC"]
        
        # Get LOSO value for this subject
        loso_acc = np.nan
        loso_f1 = np.nan
        loso_mcc = np.nan
        
        if loso_subject_df is not None:
            loso_sub = loso_subject_df[(loso_subject_df["Subject"] == sub) & 
                                        (loso_subject_df["Model"] == best_model)]
            if not loso_sub.empty:
                loso_acc = loso_sub["Accuracy"].values[0]
                loso_f1 = loso_sub["Macro_F1"].values[0]
                loso_mcc = loso_sub["MCC"].values[0]
        
        subject_gain_rows.append({
            "Subject": sub,
            "Best_Model": best_model,
            "LOSO_Accuracy": loso_acc,
            "SD_Accuracy": sd_acc,
            "Accuracy_Gain": sd_acc - loso_acc if not np.isnan(loso_acc) else np.nan,
            "LOSO_F1": loso_f1,
            "SD_F1": sd_f1,
            "F1_Gain": sd_f1 - loso_f1 if not np.isnan(loso_f1) else np.nan,
            "LOSO_MCC": loso_mcc,
            "SD_MCC": sd_mcc,
            "MCC_Gain": sd_mcc - loso_mcc if not np.isnan(loso_mcc) else np.nan,
        })
    
    df_subject_gain = pd.DataFrame(subject_gain_rows)
    df_subject_gain = df_subject_gain.sort_values(["Accuracy_Gain", "Subject"], ascending=[False, True])
    save_csv(df_subject_gain, OUT_DIR / "Table_S23_Subject_Personalization_Gain.csv")
    
    # Table S24: Subject Performance Summary
    subject_perf_rows = []
    for sub in df_subjects["Subject"].unique():
        sub_data = df_subjects[df_subjects["Subject"] == sub]
        best_row = sub_data.loc[sub_data["Accuracy"].idxmax()]
        subject_perf_rows.append({
            "Subject": sub,
            "Best_Model": best_row["Model"],
            "Accuracy": best_row["Accuracy"],
            "Macro_F1": best_row["Macro_F1"],
            "MCC": best_row["MCC"],
            "Balanced_Accuracy": best_row["Balanced_Accuracy"],
        })
    
    df_subject_perf = pd.DataFrame(subject_perf_rows)
    df_subject_perf = df_subject_perf.sort_values(["Accuracy", "Subject"], ascending=[False, True])
    save_csv(df_subject_perf, OUT_DIR / "Table_S24_Subject_Performance_Summary.csv")
    
    return df_model_gain, df_subject_gain


# ============================================================================
# PER-05: Subject Variability Analysis
# ============================================================================

def analyze_subject_variability(df_subjects: pd.DataFrame) -> pd.DataFrame:
    """
    PER-05: Analyze variability of personalized performance across subjects.
    """
    logger.info("  PER-05: Subject Variability Analysis...")
    
    variability_rows = []
    
    for model in MODEL_NAMES:
        model_data = df_subjects[df_subjects["Model"] == model]
        if model_data.empty:
            continue
        
        accuracies = model_data["Accuracy"].values
        f1s = model_data["Macro_F1"].values
        mccs = model_data["MCC"].values
        
        variability_rows.append({
            "Model": model,
            "Mean_Accuracy": np.mean(accuracies),
            "Std_Accuracy": np.std(accuracies, ddof=1),
            "Min_Accuracy": np.min(accuracies),
            "Max_Accuracy": np.max(accuracies),
            "IQR_Accuracy": np.percentile(accuracies, 75) - np.percentile(accuracies, 25),
            "Mean_F1": np.mean(f1s),
            "Std_F1": np.std(f1s, ddof=1),
            "Mean_MCC": np.mean(mccs),
            "Std_MCC": np.std(mccs, ddof=1),
            "N_Subjects": len(accuracies),
        })
    
    df_variability = pd.DataFrame(variability_rows)
    df_variability = df_variability.sort_values(["Model"], ascending=[True])
    return df_variability


# ============================================================================
# PER-06: Gesture-Wise Personalization Analysis
# ============================================================================

def run_gesture_personalization(
    global_true: List[int],
    global_pred: List[int],
    loso_cm_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    PER-06: Compute gesture-wise personalization improvement.
    Uses global predictions (all samples) for one-vs-rest formulation.
    """
    logger.info("  PER-06: Gesture-Wise Personalization Analysis...")
    
    # Convert to numpy arrays
    global_true = np.array(global_true)
    global_pred = np.array(global_pred)
    
    if len(global_true) == 0:
        logger.warning("  No predictions available for gesture analysis")
        return pd.DataFrame()
    
    # Compute subject-dependent per-gesture metrics using one-vs-rest on ALL predictions
    sd_gesture_metrics = {}
    
    for gest_idx, gest_name in enumerate(GESTURE_SHORT):
        y_true_binary = [1 if t == gest_idx else 0 for t in global_true]
        y_pred_binary = [1 if p == gest_idx else 0 for p in global_pred]
        
        acc = accuracy_score(y_true_binary, y_pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        
        sd_gesture_metrics[gest_name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": np.sum(y_true_binary)
        }
    
    # Load LOSO best model confusion matrix
    loso_gesture_metrics = {}
    
    if loso_cm_dir is None:
        loso_cm_dir = BLOCK_DIRS["BLOCK_3"]
    
    loso_cm_path = loso_cm_dir / "LOSO_best_cm.csv"
    if not loso_cm_path.exists():
        loso_cm_path = loso_cm_dir / "loso_cm_RF.csv"
        logger.warning(f"    LOSO_best_cm.csv not found, using RF as reference")
    
    if loso_cm_path.exists():
        try:
            cm = pd.read_csv(loso_cm_path, index_col=0)
            total_samples = cm.sum().sum()
            
            for gest_idx, gest_name in enumerate(GESTURE_SHORT):
                if gest_name in cm.index:
                    tp = cm.loc[gest_name, gest_name]
                    fn = cm.loc[gest_name].sum() - tp
                    fp = cm[gest_name].sum() - tp
                    tn = total_samples - tp - fp - fn
                    
                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    loso_gesture_metrics[gest_name] = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
                    }
        except Exception as e:
            logger.warning(f"    Could not load LOSO CM: {e}")
            for gest_name in GESTURE_SHORT:
                loso_gesture_metrics[gest_name] = {
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1": 0.0
                }
    else:
        for gest_name in GESTURE_SHORT:
            loso_gesture_metrics[gest_name] = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0
            }
    
    # Build gesture gain table
    gesture_rows = []
    for gest_name in GESTURE_SHORT:
        sd = sd_gesture_metrics.get(gest_name, {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0})
        loso = loso_gesture_metrics.get(gest_name, {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0})
        
        gesture_rows.append({
            "Gesture": gest_name,
            "LOSO_Accuracy": loso["Accuracy"],
            "SD_Accuracy": sd["Accuracy"],
            "Accuracy_Gain": sd["Accuracy"] - loso["Accuracy"],
            "LOSO_Precision": loso["Precision"],
            "SD_Precision": sd["Precision"],
            "Precision_Gain": sd["Precision"] - loso["Precision"],
            "LOSO_Recall": loso["Recall"],
            "SD_Recall": sd["Recall"],
            "Recall_Gain": sd["Recall"] - loso["Recall"],
            "LOSO_F1": loso["F1"],
            "SD_F1": sd["F1"],
            "F1_Gain": sd["F1"] - loso["F1"],
            "Support": sd.get("Support", 0),
        })
    
    df_gesture = pd.DataFrame(gesture_rows)
    df_gesture = df_gesture.sort_values(["Accuracy_Gain", "Gesture"], ascending=[False, True])
    save_csv(df_gesture, OUT_DIR / "Table_S20_Gesture_Personalization_Improvement.csv")
    
    return df_gesture


# ============================================================================
# ADDITIONAL TABLES: Subject Ranking and Best Model Per Subject
# ============================================================================

def generate_subject_tables(df_subjects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Table S21: Subject Ranking and Table S22: Best Model Per Subject.
    """
    logger.info("  Generating Subject Ranking and Best Model tables...")
    
    # Table S21: Subject Ranking by Accuracy
    best_per_subject = df_subjects.loc[df_subjects.groupby("Subject")["Accuracy"].idxmax()]
    best_per_subject = best_per_subject.sort_values(["Accuracy", "Subject"], ascending=[False, True])
    best_per_subject = best_per_subject[["Subject", "Model", "Accuracy", "Macro_F1", "MCC"]]
    best_per_subject.columns = ["Subject", "Best_Model", "Accuracy", "F1", "MCC"]
    best_per_subject["Rank"] = range(1, len(best_per_subject) + 1)
    
    save_csv(best_per_subject, OUT_DIR / "Table_S21_Subject_Ranking.csv")
    
    # Table S22: Best Model Per Subject
    best_model_summary = []
    for model in MODEL_NAMES:
        model_data = df_subjects[df_subjects["Model"] == model]
        if model_data.empty:
            continue
        best_model_count = len(best_per_subject[best_per_subject["Best_Model"] == model])
        best_model_summary.append({
            "Model": model,
            "N_Subjects_Best": best_model_count,
            "Mean_Accuracy": model_data["Accuracy"].mean(),
            "Std_Accuracy": model_data["Accuracy"].std(),
            "Mean_F1": model_data["Macro_F1"].mean(),
            "Mean_MCC": model_data["MCC"].mean(),
        })
    
    df_best_model = pd.DataFrame(best_model_summary)
    df_best_model = df_best_model.sort_values(["N_Subjects_Best", "Model"], ascending=[False, True])
    save_csv(df_best_model, OUT_DIR / "Table_S22_Best_Model_Per_Subject.csv")
    
    return best_per_subject, df_best_model


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def _generate_figures(
    df_folds: pd.DataFrame,
    df_subjects: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_model_gain: pd.DataFrame,
    df_gesture: pd.DataFrame,
    df_variability: pd.DataFrame,
    best_per_subject: pd.DataFrame
) -> None:
    """Generate Figures 21-26 and S17-S19."""
    

    # Figure 21 – Subject-Dependent Performance Comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_folds,
        x="Model",
        y="Accuracy",
        order=MODEL_NAMES,
        errorbar="sd",
        palette=PALETTE,
        hue="Model",
        legend=False,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_ylim(0, 1.02)

    ax.set_xlabel(
        "Model",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Accuracy",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        "Figure 21 – Subject-Dependent Performance Comparison (Mean ± SD)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30,
    )

    plt.tight_layout()

    save_fig(
        fig,
        OUT_DIR / "Fig21_SubjectDependent_Comparison.png",
    )

    plt.close(fig)

    logger.info("  Figure 21 saved.")


    # ============================================================================
    # Figure 22 – Subject-Dependent Accuracy Distribution
    # ============================================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df_subjects,
        x="Model",
        y="Accuracy",
        order=MODEL_NAMES,
        palette=PALETTE,
        width=0.55,
        linewidth=1.2,
        fliersize=4,
        ax=ax,
    )

    sns.stripplot(
        data=df_subjects,
        x="Model",
        y="Accuracy",
        order=MODEL_NAMES,
        color="black",
        size=3,
        alpha=0.55,
        jitter=0.20,
        ax=ax,
    )

    ax.set_ylim(0, 1.02)

    ax.set_xlabel(
        "Model",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Accuracy",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        "Figure 22 – Subject-Dependent Accuracy Distribution",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30,
    )

    plt.tight_layout()

    save_fig(
        fig,
        OUT_DIR / "Fig22_SubjectDependent_Accuracy_Distribution.png",
    )

    plt.close(fig)

    logger.info("  Figure 22 saved.")
    
   
    # ============================================================================
    # Figure 23 – Subject-Wise Performance Heatmap
    # ============================================================================

    if not df_subjects.empty:

        pivot = df_subjects.pivot_table(
            index="Subject",
            columns="Model",
            values="Accuracy",
        )

        if not pivot.empty:

            # ------------------------------------------------------------------
            # Publication-quality numeric subject ordering
            # Works with:
            #   Subject_01 ... Subject_40
            #   S01 ... S40
            #   S1 ... S40
            #   1 ... 40
            # ------------------------------------------------------------------
            def _subject_sort_key(s):
                import re
                nums = re.findall(r"\d+", str(s))
                return int(nums[-1]) if nums else str(s)

            pivot = pivot.reindex(
                sorted(pivot.index, key=_subject_sort_key)
            )

            # Keep model order consistent across the paper
            model_order = [m for m in MODEL_NAMES if m in pivot.columns]
            pivot = pivot[model_order]

            # Publication-quality figure size
            fig_width = max(10, len(model_order) * 2.2)
            fig_height = max(14, len(pivot.index) * 0.35)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                linewidths=0.5,
                linecolor="white",
                square=False,
                vmin=0.0,
                vmax=1.0,
                cbar_kws={
                    "label": "Accuracy"
                },
                annot_kws={
                    "fontsize": 8
                },
                ax=ax,
            )

            # Show every subject label
            ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
            ax.set_yticklabels(
                pivot.index,
                rotation=0,
                fontsize=8,
            )

            ax.set_xticks(np.arange(len(model_order)) + 0.5)
            ax.set_xticklabels(
                model_order,
                rotation=0,
                fontsize=10,
                fontweight="bold",
            )

            ax.set_xlabel(
                "Model",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_ylabel(
                "Subject",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_title(
                "Figure 23 – Subject-Wise Performance Heatmap",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            plt.tight_layout()

            save_fig(
                fig,
                OUT_DIR / "Fig23_SubjectWise_Performance_Heatmap.png",
            )

            plt.close(fig)

            logger.info("  Figure 23 saved.")
    
    # ============================================================================
    # Figure 24 – Personalization Gain Analysis (LOSO vs Subject-Dependent)
    # ============================================================================

    if not df_model_gain.empty and "Accuracy_Gain" in df_model_gain.columns:

        # Keep model order consistent throughout the paper
        df_plot = (
            df_model_gain
            .set_index("Model")
            .reindex(MODEL_NAMES)
            .dropna(subset=["Accuracy_Gain"])
            .reset_index()
        )

        if not df_plot.empty:

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(
                df_plot["Model"],
                df_plot["Accuracy_Gain"],
                color=[MODEL_COLORS.get(m, "#808080") for m in df_plot["Model"]],
                edgecolor="black",
                linewidth=0.8,
            )

            # Zero reference line
            ax.axhline(
                y=0,
                color="black",
                linewidth=1.0,
                alpha=0.6,
            )

            # ------------------------------------------------------------------
            # Robust axis limits (works for positive, negative, or mixed gains)
            # ------------------------------------------------------------------
            gain_max = df_plot["Accuracy_Gain"].max()
            gain_min = df_plot["Accuracy_Gain"].min()

            span = max(abs(gain_max), abs(gain_min))

            if span == 0:
                span = 0.05

            padding = span * 0.10

            ax.set_ylim(
                gain_min - padding,
                gain_max + padding,
            )

            ax.set_xlabel(
                "Model",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_ylabel(
                "Accuracy Gain (Subject-Dependent − LOSO)",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_title(
                "Figure 24 – Personalization Gain Analysis (LOSO vs Subject-Dependent)",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.30,
            )

            # ------------------------------------------------------------------
            # Value labels
            # ------------------------------------------------------------------
            label_offset = span * 0.04

            for bar, value in zip(bars, df_plot["Accuracy_Gain"]):

                if value >= 0:
                    y = value + label_offset
                    va = "bottom"
                else:
                    y = value - label_offset
                    va = "top"

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=10,
                    fontweight="bold",
                )

            plt.tight_layout()

            save_fig(
                fig,
                OUT_DIR / "Fig24_Personalization_Gain_Analysis.png",
            )

            plt.close(fig)

            logger.info("  Figure 24 saved.")
    
    # ============================================================================
    # Figure 25 – Subject Ranking by Accuracy (Best Model per Subject)
    # ============================================================================
    
    if not best_per_subject.empty:

        # Sort subjects by best accuracy (highest first)
        best_per_subject = (
            best_per_subject
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True)
        )

        # Publication-quality figure size
        fig_width = max(16, len(best_per_subject) * 0.45)
        fig_height = 8

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Consistent model colors
        colors = [
            MODEL_COLORS.get(model, "#808080")
            for model in best_per_subject["Best_Model"]
        ]

        bars = ax.bar(
            range(len(best_per_subject)),
            best_per_subject["Accuracy"],
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        # Subject labels
        ax.set_xticks(range(len(best_per_subject)))
        ax.set_xticklabels(
            best_per_subject["Subject"],
            rotation=45,
            ha="right",
            fontsize=9,
        )

        # Model label above each bar
        for bar, model in zip(bars, best_per_subject["Best_Model"]):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                model,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlabel(
            "Subject",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Best Model Accuracy",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_title(
            "Figure 25 – Subject Ranking by Accuracy (Best Model per Subject)",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.set_ylim(0, 1.05)

        # Publication-style grid
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.30,
        )

        # Legend using the pipeline's MODEL_COLORS dictionary
        legend_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=MODEL_COLORS[m],
                edgecolor="black",
                label=m,
            )
            for m in MODEL_NAMES
            if m in MODEL_COLORS
        ]

        ax.legend(
            handles=legend_handles,
            title="Best Model",
            loc="upper right",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
        )

        plt.tight_layout()

        save_fig(
            fig,
            OUT_DIR / "Fig25_Subject_Ranking_by_Accuracy.png",
        )

        plt.close(fig)

        logger.info("  Figure 25 saved.")
    
   
    # ============================================================================
    # Figure 26 – Gesture-Wise Personalization Gain
    # ============================================================================

    if not df_gesture.empty:

        # ----------------------------------------------------------------------
        # Sort gestures by personalization gain
        # ----------------------------------------------------------------------
        sorted_df = (
            df_gesture
            .sort_values("Accuracy_Gain", ascending=False)
            .reset_index(drop=True)
        )

        # Publication color palette
        colors = [
            "#4CAF50" if gain >= 0 else "#E15759"
            for gain in sorted_df["Accuracy_Gain"]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            sorted_df["Gesture"],
            sorted_df["Accuracy_Gain"],
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        # ----------------------------------------------------------------------
        # Zero reference line
        # ----------------------------------------------------------------------
        ax.axvline(
            x=0,
            color="black",
            linewidth=1.0,
            alpha=0.6,
        )

        # ----------------------------------------------------------------------
        # Robust axis limits
        # Works for:
        #   • all positive
        #   • all negative
        #   • mixed values
        # ----------------------------------------------------------------------
        gain_max = sorted_df["Accuracy_Gain"].max()
        gain_min = sorted_df["Accuracy_Gain"].min()

        span = max(abs(gain_max), abs(gain_min))

        if span == 0:
            span = 0.05

        padding = span * 0.10

        ax.set_xlim(
            gain_min - padding,
            gain_max + padding,
        )

        ax.set_xlabel(
            "Accuracy Gain (Subject-Dependent − LOSO)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Gesture",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_title(
            "Figure 26 – Gesture-Wise Personalization Gain",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.30,
        )

        # ----------------------------------------------------------------------
        # Value labels
        # ----------------------------------------------------------------------
        label_offset = span * 0.04

        for bar, value in zip(bars, sorted_df["Accuracy_Gain"]):

            if value >= 0:
                x = value + label_offset
                ha = "left"
            else:
                x = value - label_offset
                ha = "right"

            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha=ha,
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        save_fig(
            fig,
            OUT_DIR / "Fig26_Gesture_Personalization_Gain.png",
        )

        plt.close(fig)

        logger.info("  Figure 26 saved.")
    
    # ============================================================================
    # Figure S17 – Subject-Wise Model Comparison Heatmap (Macro-F1)
    # ============================================================================

    if not df_subjects.empty:

        pivot_f1 = df_subjects.pivot_table(
            index="Subject",
            columns="Model",
            values="Macro_F1",
        )

        if not pivot_f1.empty:

            # ------------------------------------------------------------------
            # Publication-quality numeric subject ordering
            # Works with:
            #   Subject_01 ... Subject_40
            #   S01 ... S40
            #   S1 ... S40
            #   1 ... 40
            # ------------------------------------------------------------------
            def _subject_sort_key(s):
                import re
                nums = re.findall(r"\d+", str(s))
                return int(nums[-1]) if nums else str(s)

            pivot_f1 = pivot_f1.reindex(
                sorted(pivot_f1.index, key=_subject_sort_key)
            )

            # Keep model order consistent throughout the paper
            model_order = [m for m in MODEL_NAMES if m in pivot_f1.columns]
            pivot_f1 = pivot_f1[model_order]

            # Publication-quality figure size
            fig_width = max(10, len(model_order) * 2.2)
            fig_height = max(14, len(pivot_f1.index) * 0.35)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                pivot_f1,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                linewidths=0.5,
                linecolor="white",
                square=False,
                vmin=0.0,
                vmax=1.0,
                cbar_kws={
                    "label": "Macro-F1 Score"
                },
                annot_kws={
                    "fontsize": 8
                },
                ax=ax,
            )

            # Force all subject labels to be displayed
            ax.set_yticks(np.arange(len(pivot_f1.index)) + 0.5)
            ax.set_yticklabels(
                pivot_f1.index,
                rotation=0,
                fontsize=8,
            )

            ax.set_xticks(np.arange(len(model_order)) + 0.5)
            ax.set_xticklabels(
                model_order,
                rotation=0,
                fontsize=10,
                fontweight="bold",
            )

            ax.set_xlabel(
                "Model",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_ylabel(
                "Subject",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_title(
                "Figure S17 – Subject-Wise Model Comparison Heatmap (Macro-F1)",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

            plt.tight_layout()

            save_fig(
                fig,
                OUT_DIR / "FigS17_SubjectWise_Model_Comparison_Heatmap.png",
            )

            plt.close(fig)

            logger.info("  Figure S17 saved.")


    # ============================================================================
    # Figure S18 – Subject Variability Distribution
    # ============================================================================

    if not df_variability.empty:

        df_plot = (
            df_variability
            .set_index("Model")
            .reindex(MODEL_NAMES)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            df_plot["Model"],
            df_plot["Std_Accuracy"],
            color=[MODEL_COLORS[m] for m in df_plot["Model"]],
            edgecolor="black",
            linewidth=0.8,
        )

        ymax = df_plot["Std_Accuracy"].max() * 1.15

        ax.set_ylim(0, ymax)

        ax.set_xlabel(
            "Model",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Standard Deviation of Subject Accuracy",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_title(
            "Figure S18 – Subject Variability Distribution",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.30,
        )

        offset = ymax * 0.015

        for bar, value in zip(bars, df_plot["Std_Accuracy"]):

            ax.text(
                bar.get_x() + bar.get_width()/2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        save_fig(
            fig,
            OUT_DIR / "FigS18_Subject_Variability_Distribution.png",
        )

        plt.close(fig)

        logger.info("  Figure S18 saved.")


    # ============================================================================
    # Figure S19 – Fold-Level Subject-Dependent Performance
    # ============================================================================

    if not df_folds.empty:

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=df_folds,
            x="Model",
            y="Accuracy",
            order=MODEL_NAMES,
            palette=PALETTE,
            linewidth=1.2,
            width=0.55,
            fliersize=4,
            ax=ax,
        )

        sns.stripplot(
            data=df_folds,
            x="Model",
            y="Accuracy",
            order=MODEL_NAMES,
            color="black",
            size=3,
            alpha=0.50,
            jitter=0.20,
            ax=ax,
        )

        ax.set_ylim(0, 1.02)

        ax.set_xlabel(
            "Model",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Fold Accuracy",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_title(
            "Figure S19 – Fold-Level Subject-Dependent Performance",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.30,
        )

        plt.tight_layout()

        save_fig(
            fig,
            OUT_DIR / "FigS19_FoldLevel_SubjectDependent_Performance.png",
        )

        plt.close(fig)

        logger.info("  Figure S19 saved.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_block4() -> Dict:
    """Main entry point for BLOCK 4."""
    
    # Runtime logging
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK 4 | PER1: Personalized Gesture Recognition Framework")
    logger.info("=" * 70)
    
    # Log software versions
    logger.info(f"Python : {sys.version}")
    logger.info(f"Numpy  : {np.__version__}")
    logger.info(f"Pandas : {pd.__version__}")
    logger.info(f"Sklearn: {sklearn.__version__}")
    logger.info(f"XGBoost: {xgboost.__version__}")
    logger.info(f"PID    : {os.getpid()}")
    
    # Log thread settings
    logger.info(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    logger.info(f"  XGBOOST_NUM_THREADS: {os.environ.get('XGBOOST_NUM_THREADS', 'not set')}")

    # ── Try to load from BLOCK_2 artifacts ─────────────────────────────
    try:
        X, y, subjects, feature_names = load_feature_matrix()
        logger.info(f"  Loaded {len(X)} windows from BLOCK_2 artifacts")
    except (FileNotFoundError, KeyError, TypeError) as e:
        logger.warning(f"  BLOCK_2 artifacts not found: {e}")
        logger.info("  Falling back to raw data loading...")
        X, y, subjects, feature_names = _load_fallback_data()

    # ── PER-01 to PER-03: Subject-Dependent Evaluation ──────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PER-01 to PER-03: Subject-Dependent Performance Evaluation")
    logger.info("=" * 60)
    
    df_folds, df_subjects, df_summary, archives = run_subject_dependent(X, y, subjects)
    
    logger.info(f"  Table 5 saved: {len(df_summary)} models")
    logger.info("\n" + df_summary[["Model", "Accuracy_Mean", "Accuracy_SD",
                                   "Macro_F1_Mean", "MCC_Mean"]].to_string(index=False))

    # ── PER-04: Personalization Gain Analysis ──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PER-04: Personalization Gain Analysis")
    logger.info("=" * 60)
    
    df_model_gain, df_subject_gain = run_personalization_gain(df_subjects)
    logger.info(f"  Table S19 saved: {len(df_model_gain)} rows")
    logger.info(f"  Table S23 saved: {len(df_subject_gain)} rows")
    logger.info(f"  Table S24 saved: subject performance summary")

    # ── PER-05: Subject Variability Analysis ────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PER-05: Subject Variability Analysis")
    logger.info("=" * 60)
    
    df_variability = analyze_subject_variability(df_subjects)
    logger.info(f"  Variability analysis complete: {len(df_variability)} models")

    # ── PER-06: Gesture-Wise Personalization ───────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PER-06: Gesture-Wise Personalization Analysis")
    logger.info("=" * 60)
    
    df_gesture = run_gesture_personalization(
        archives["global_true"],
        archives["global_pred"],
        BLOCK_DIRS["BLOCK_3"]
    )
    logger.info(f"  Table S20 saved: {len(df_gesture)} rows")

    # ── Additional Tables ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Generating Additional Tables")
    logger.info("=" * 60)
    
    best_per_subject, df_best_model = generate_subject_tables(df_subjects)
    logger.info(f"  Table S21 saved: {len(best_per_subject)} rows")
    logger.info(f"  Table S22 saved: {len(df_best_model)} rows")

    # ── Figures ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Generating Figures")
    logger.info("=" * 60)
    
    _generate_figures(
        df_folds, df_subjects, df_summary, df_model_gain, df_gesture,
        df_variability, best_per_subject
    )
    
    # Save environment versions
    _save_environment_versions()

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "table5": OUT_DIR / "Table5_SubjectDependent_Summary.csv",
        "table_s17": OUT_DIR / "Table_S17_SubjectWise_Model_Comparison.csv",
        "table_s18": OUT_DIR / "Table_S18_SubjectDependent_Fold_Results.csv",
        "table_s19": OUT_DIR / "Table_S19_Personalization_Gain_Analysis.csv",
        "table_s20": OUT_DIR / "Table_S20_Gesture_Personalization_Improvement.csv",
        "table_s21": OUT_DIR / "Table_S21_Subject_Ranking.csv",
        "table_s22": OUT_DIR / "Table_S22_Best_Model_Per_Subject.csv",
        "table_s23": OUT_DIR / "Table_S23_Subject_Personalization_Gain.csv",
        "table_s24": OUT_DIR / "Table_S24_Subject_Performance_Summary.csv",
        "prediction_archive": OUT_DIR / "PredictionArchive_SubjectDependent.csv",
        "fig21": OUT_DIR / "Fig21_SubjectDependent_Comparison.png",
        "fig22": OUT_DIR / "Fig22_SubjectDependent_Accuracy_Distribution.png",
        "fig23": OUT_DIR / "Fig23_SubjectWise_Performance_Heatmap.png",
        "fig24": OUT_DIR / "Fig24_Personalization_Gain_Analysis.png",
        "fig25": OUT_DIR / "Fig25_Subject_Ranking_by_Accuracy.png",
        "fig26": OUT_DIR / "Fig26_Gesture_Personalization_Gain.png",
        "fig_s17": OUT_DIR / "FigS17_SubjectWise_Model_Comparison_Heatmap.png",
        "fig_s18": OUT_DIR / "FigS18_Subject_Variability_Distribution.png",
        "fig_s19": OUT_DIR / "FigS19_FoldLevel_SubjectDependent_Performance.png",
        "metadata": OUT_DIR / "model_metadata.csv",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    elapsed = time.perf_counter() - t0
    logger.info(f"  BLOCK 4 runtime: {elapsed:.2f} sec")

    summary = {
        "n_subjects": int(len(np.unique(subjects))),
        "best_model": str(df_summary.loc[df_summary["Accuracy_Mean"].idxmax(), "Model"]),
        "best_acc": float(df_summary["Accuracy_Mean"].max()),
        "n_windows": int(len(X)),
        "runtime_sec": elapsed,
        "pipeline": PIPELINE_NAME,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("BLOCK 4 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best Subject-Dependent: {summary['best_model']} @ {summary['best_acc']:.4f}")
    logger.info(f"  Total subjects: {summary['n_subjects']}")
    logger.info(f"  Runtime: {elapsed:.2f} sec")
    logger.info("=" * 60)
    
    return summary


def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return

    try:
        result = run_block4()
        mark_completed(BLOCK_NAME, metadata=result)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.error(f"❌  {BLOCK_NAME} FAILED: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_4 WITH ALL FIXES
# ✓ Improvement 1: Official balanced_accuracy_score() used
# ✓ Improvement 2: Removed NORMALIZED_CM_DIR - saves all in same directory
# ✓ Improvement 3: Removed normalize parameter from save_confusion_matrix()
# ✓ Improvement 4: Removed global_prob (not used later)
# ✓ Improvement 5: Moved gc.collect() to after each fold and subject
# ✓ FIX 1: Centralized model builder (_build_model) using config.py parameters
# ✓ FIX 2: Normalized confusion matrices with heatmaps (Blues, vmin=0, vmax=1)
# ✓ FIX 3: Reusable save_confusion_matrix() helper
# ✓ FIX 4: plt.close(fig) after every figure (no gc.collect needed)
# ✓ FIX 5: Deterministic table sorting (Accuracy + Subject)
# ✓ FIX 6: Model metadata saved (Subject, Fold, Model, all metrics)
# ✓ FIX 7: Raw + Normalized CMs saved for Subject, Model, Fold
# ✓ FIX 8: Identical naming as BLOCK_3 (Subject_S01_RF_CM, etc.)
# ✓ FIX 9: Prediction probabilities saved in CSV (when available)
# ✓ FIX 10: Runtime logging (start/end elapsed time)
# ✓ FIX 11: Software versions saved (environment_versions.json)
# ✓ FIX 12: Artifact manifest includes all new CM and metadata files
# ✓ NEW: PredictionArchive_SubjectDependent.csv with Subject, Fold, Model, True_Label, Pred_Label, Probabilities
# ✓ PER-01 to PER-06: All modules complete
# ✓ Figures 21-26, S17-S19
# ✓ Tables 5, S17-S24
# ✓ No leakage (fit only on training data per subject)
# ✓ SMOTE guard (min_samples > 6)
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────
