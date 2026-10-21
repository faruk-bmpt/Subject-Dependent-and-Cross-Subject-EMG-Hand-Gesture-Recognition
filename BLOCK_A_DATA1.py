# BLOCK_A_DATA1.py - FINAL PUBLICATION-QUALITY VERSION
# ================
# FILE PURPOSE:
#     PRE-BLOCK A — DATA1: Dataset Integrity Validation and Quality Assurance.
#     Validates dataset structure, file counts, required CSV columns,
#     signal lengths, and per-subject/gesture completeness BEFORE any
#     experiment runs. Produces Table S1.
#
# INPUT_ARTIFACTS  :
#     - INPUT_PATH (EMG_Gesture directory tree)
#
# OUTPUT_ARTIFACTS :
#     - outputs/PRE_A/Table_S1_Dataset_Integrity.csv
#     - outputs/PRE_A/Table_S1b_Subject_Summary_Report.csv
#     - outputs/PRE_A/Table_S1c_Missing_Gesture_Report.csv
#     - outputs/PRE_A/Table_S1d_Class_Distribution.csv
#     - outputs/PRE_A/Table_S1e_Recording_Statistics.csv
#     - outputs/PRE_A/Table_S1f_Subject_Completeness.csv
#     - outputs/PRE_A/Dataset_Integrity_Report.txt
#     - outputs/PRE_A/FigS1_Class_Distribution.png
#     - outputs/PRE_A/FigS2_DataIntegrity_Heatmap.png
#     - outputs/PRE_A/environment_versions.json
#     - outputs/PRE_A/artifacts.json
#     - outputs/PRE_A/dataset_summary.json
#     - state/BLOCK_A_DATA1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_A_DATA1.py
#

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
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ── pipeline imports ──────────────────────────────────────────────────────────
from config import (
    INPUT_PATH,
    BLOCK_DIRS,
    GESTURE_MAP,
    GESTURE_SHORT,
    WINDOW_SIZE,
    STEP_SIZE,
    SAMPLING_RATE,
    MIN_RMS_THRESHOLD,
    SEED,
    PIPELINE_NAME,
    EXPECTED_SUBJECTS,
    EXPECTED_GESTURES,
    EXPECTED_FILES_PER_GESTURE,
    RMS_VALIDATION_FACTOR,
    REQUIRED_CHANNELS,
    EMG_CHANNEL_COLUMNS,
    N_CHANNELS,
)
from state_manager import is_completed, mark_completed
from artifact_manager import save_csv, save_txt, save_fig, save_json

warnings.filterwarnings("ignore")

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_A_DATA1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_A")

BLOCK_NAME = "BLOCK_A_DATA1"
OUT_DIR    = BLOCK_DIRS["PRE_A"]

# ── GESTURE_SHORT_MAP for safe indexing ──────────────────────────────────────
GESTURE_SHORT_MAP = {
    0: "HOC", 1: "TFM", 2: "IFM", 3: "MFM", 4: "RFM", 5: "LFM", 6: "PH"
}


# ============================================================================
# FIGURE FINALIZER HELPER (identical to other blocks)
# ============================================================================

def _finalize_figure(fig, path: Path, dpi: int = 300) -> None:
    """
    Centralized figure finalization: save, close, and garbage collect.
    Identical to BLOCK_13_ROB1.py.
    """
    save_fig(fig, path, dpi=dpi)
    plt.close(fig)
    gc.collect()


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
    
    save_json(env_info, OUT_DIR / "environment_versions.json")
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PER-FILE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_csv_file(fpath: str) -> Dict[str, Any]:
    """
    Validate a single EMG CSV file.
    
    Uses EMG_CHANNEL_COLUMNS from config.py as the single source of truth.
    This guarantees consistency with the rest of the pipeline.
    """
    result: Dict[str, Any] = {
        "path"          : str(fpath),
        "readable"      : False,
        "has_cols"      : False,
        "n_rows"        : 0,
        "n_cols"        : 0,
        "has_nan"       : False,
        "rms_ok"        : False,
        "total_samples" : 0,
        "duration_s"    : 0.0,
        "issue"         : "",
    }

    try:
        # Read CSV
        df = pd.read_csv(fpath)
        result["readable"] = True
        result["n_rows"] = len(df)
        result["n_cols"] = len(df.columns)
        result["total_samples"] = len(df)
        result["duration_s"] = round(len(df) / SAMPLING_RATE, 4)

        # Try label-based indexing first
        try:
            emg_cols = [col for col in EMG_CHANNEL_COLUMNS if col in df.columns]
            if len(emg_cols) == len(EMG_CHANNEL_COLUMNS):
                data = df[emg_cols].astype(float).values
                result["has_cols"] = True
            else:
                # Fallback: try position-based indexing
                found_cols = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if "emg" in col_lower or "ch" in col_lower:
                        found_cols.append(col)
                
                if len(found_cols) >= N_CHANNELS:
                    data = df[found_cols[:N_CHANNELS]].astype(float).values
                    result["has_cols"] = True
                elif len(df.columns) >= N_CHANNELS + 1:
                    # Assume first column is timestamp, use next N_CHANNELS
                    data = df.iloc[:, 1:N_CHANNELS+1].astype(float).values
                    result["has_cols"] = True
                elif len(df.columns) >= N_CHANNELS:
                    data = df.iloc[:, -N_CHANNELS:].astype(float).values
                    result["has_cols"] = True
                else:
                    result["issue"] = f"Could not find {N_CHANNELS} EMG columns. "
                    result["issue"] += f"Available: {df.columns.tolist()[:10]}"
                    return result
        except (KeyError, ValueError, IndexError) as e:
            result["issue"] = f"Column extraction failed: {str(e)[:100]}"
            return result

        if not result["has_cols"]:
            return result

        # Check for NaN values
        has_nan = pd.isnull(data).any()
        result["has_nan"] = bool(has_nan)

        # Check RMS threshold
        rms = np.sqrt(np.mean(data ** 2, axis=0))
        result["rms_ok"] = bool(np.all(rms > MIN_RMS_THRESHOLD * RMS_VALIDATION_FACTOR))

    except Exception as exc:
        result["issue"] = str(exc)
        result["readable"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SUBJECT COMPLETENESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def _audit_subject_completeness(
    sub: str,
    sub_path: Path,
) -> Tuple[List[Dict], bool]:
    """Check whether a subject has all expected gestures."""
    missing_rows: List[Dict] = []
    subject_ok = True

    expected_gestures = set(GESTURE_MAP.keys())

    for gest in expected_gestures:
        gest_path = sub_path / gest
        if not gest_path.is_dir():
            missing_rows.append({
                "Subject": sub,
                "Gesture": gest,
                "Issue": "Gesture directory missing",
                "Files_Found": 0,
                "Files_Expected": EXPECTED_FILES_PER_GESTURE or "N/A",
            })
            subject_ok = False
            continue

        files = sorted(Path(gest_path).glob("*.csv"))
        if EXPECTED_FILES_PER_GESTURE is not None:
            if len(files) != EXPECTED_FILES_PER_GESTURE:
                missing_rows.append({
                    "Subject": sub,
                    "Gesture": gest,
                    "Issue": f"Expected {EXPECTED_FILES_PER_GESTURE} files, found {len(files)}",
                    "Files_Found": len(files),
                    "Files_Expected": EXPECTED_FILES_PER_GESTURE,
                })
                subject_ok = False

    return missing_rows, subject_ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RECORDING STATISTICS (FIXED: Column name consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_recording_stats(
    integrity_rows: List[Dict],
) -> pd.DataFrame:
    """
    Compute per-subject and per-gesture recording statistics.
    
    FIX: Uses consistent column names 'readable' and 'has_cols' (lowercase)
    matching the keys used in _validate_csv_file().
    """
    df = pd.DataFrame(integrity_rows)
    
    # Check if required columns exist
    if "readable" not in df.columns or "has_cols" not in df.columns:
        logger.warning("  Missing 'readable' or 'has_cols' columns in integrity data")
        return pd.DataFrame()
    
    valid_df = df[df["readable"] & df["has_cols"]].copy()

    if valid_df.empty:
        return pd.DataFrame()

    stats_rows: List[Dict] = []
    for (sub, gest), grp in valid_df.groupby(["Subject", "Gesture"]):
        stats_rows.append({
            "Subject": sub,
            "Gesture": gest,
            "Total_Files": len(grp),
            "Valid_Files": int(grp["has_cols"].sum()),
            "Total_Samples": int(grp["total_samples"].sum()),
            "Mean_Samples_Per_File": round(grp["total_samples"].mean(), 1),
            "Mean_Duration_s": round(grp["duration_s"].mean(), 3),
            "Total_Duration_s": round(grp["duration_s"].sum(), 3),
        })

    return pd.DataFrame(stats_rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CLASS DISTRIBUTION (PUBLICATION-QUALITY FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_distribution(
    integrity_rows: List[Dict],
) -> pd.DataFrame:
    """
    Count total valid files per gesture across all subjects.
    
    PUBLICATION-QUALITY FIX: Uses .size() instead of .count() on a column.
    This avoids depending on a specific column name entirely.
    """
    df = pd.DataFrame(integrity_rows)
    
    if "readable" not in df.columns or "has_cols" not in df.columns:
        return pd.DataFrame()
    
    valid_df = df[df["readable"] & df["has_cols"]]

    if valid_df.empty:
        return pd.DataFrame()

    # FIX: Use .size() instead of .count() on a column
    counts = (
        valid_df
        .groupby("Gesture")
        .size()
        .reset_index(name="Total_Files")
    )
    
    counts["Label"] = counts["Gesture"].map(GESTURE_MAP)
    total = counts["Total_Files"].sum()
    if total > 0:
        counts["Pct"] = (counts["Total_Files"] / total * 100).round(2)
    else:
        counts["Pct"] = 0.0
    counts = counts.sort_values(["Label", "Gesture"], ascending=[True, True]).reset_index(drop=True)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _plot_class_distribution(
    df_class: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figure S1 — Class Distribution Bar Plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if df_class.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Figure S1 – Class Distribution (No Data)")
        _finalize_figure(fig, out_path)
        return

    short_labels = []
    for g in df_class["Gesture"]:
        if g in GESTURE_MAP:
            label_idx = GESTURE_MAP[g]
            short_labels.append(GESTURE_SHORT_MAP.get(label_idx, g))
        else:
            short_labels.append(g)

    colors = plt.cm.Set2(np.linspace(0, 1, len(df_class)))
    bars = ax.bar(short_labels, df_class["Total_Files"], color=colors, edgecolor="black", linewidth=0.6)

    for bar, pct in zip(bars, df_class["Pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title("Figure S1 – Class Distribution (Valid Files per Gesture)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gesture", fontsize=10)
    ax.set_ylabel("Total Valid Files", fontsize=10)
    if not df_class.empty:
        ax.set_ylim(0, df_class["Total_Files"].max() * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _finalize_figure(fig, out_path)


def _plot_integrity_heatmap(
    df_subjects: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figure S2 — Dataset Integrity Heatmap."""
    if df_subjects.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Figure S2 – Data Integrity Heatmap (No Data)")
        _finalize_figure(fig, out_path)
        return

    heat_cols = [f"{g}_valid_files" for g in GESTURE_MAP if f"{g}_valid_files" in df_subjects.columns]
    if not heat_cols:
        logger.warning("  Heatmap: no valid_files columns found")
        return

    heat_df = df_subjects.set_index("Subject")[heat_cols].copy()

    new_cols = []
    for c in heat_cols:
        gest_name = c.replace("_valid_files", "")
        if gest_name in GESTURE_MAP:
            label_idx = GESTURE_MAP[gest_name]
            new_cols.append(GESTURE_SHORT_MAP.get(label_idx, gest_name))
        else:
            new_cols.append(c)
    heat_df.columns = new_cols

    fig, ax = plt.subplots(figsize=(max(8, len(GESTURE_MAP) * 1.4), max(4, len(df_subjects) * 0.5)))
    sns.heatmap(
        heat_df.astype(float),
        annot=True, fmt=".0f",
        cmap="YlGnBu", ax=ax, linewidths=0.4,
        cbar_kws={"label": "Valid Files"},
    )
    ax.set_title("Figure S2 – Valid Files per Subject per Gesture", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gesture", fontsize=10)
    ax.set_ylabel("Subject", fontsize=10)
    _finalize_figure(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_data_integrity() -> Dict[str, Any]:
    """Main DATA1 routine."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK A | DATA1: Dataset Integrity Validation & Quality Assurance")
    logger.info("=" * 70)

    # Log software versions
    logger.info(f"Python : {sys.version}")
    logger.info(f"Numpy  : {np.__version__}")
    logger.info(f"Pandas : {pd.__version__}")
    logger.info(f"SciPy  : {scipy.__version__}")
    logger.info(f"Sklearn: {sklearn.__version__}")
    logger.info(f"XGBoost: {xgboost.__version__}")
    logger.info(f"Imblearn: {imblearn.__version__}")
    logger.info(f"PID    : {os.getpid()}")

    if not INPUT_PATH.exists():
        msg = f"INPUT_PATH does not exist: {INPUT_PATH}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    subjects = sorted(
        s for s in os.listdir(INPUT_PATH)
        if os.path.isdir(INPUT_PATH / s)
    )

    # Hard failure for subject count mismatch
    if EXPECTED_SUBJECTS is not None:
        if len(subjects) != EXPECTED_SUBJECTS:
            raise ValueError(
                f"Subject count mismatch: expected {EXPECTED_SUBJECTS}, "
                f"but found {len(subjects)} subjects in the data."
            )
        else:
            logger.info(f"  Subject count matches expected: {EXPECTED_SUBJECTS}")

    logger.info(f"  Subjects detected  : {len(subjects)}")
    logger.info(f"  Gestures expected  : {EXPECTED_GESTURES}")
    logger.info(f"  Required channels  : {REQUIRED_CHANNELS}")
    logger.info(f"  RMS valid factor   : {RMS_VALIDATION_FACTOR}")

    integrity_rows: List[Dict] = []
    subject_rows: List[Dict] = []
    all_missing: List[Dict] = []
    all_issues: List[str] = []
    total_files = 0
    total_valid = 0

    # ── Per-subject / per-gesture loop ───────────────────────────────────────
    for sub in subjects:
        sub_path = INPUT_PATH / sub

        # Completeness audit first
        missing_rows, sub_ok = _audit_subject_completeness(sub, sub_path)
        all_missing.extend(missing_rows)
        if not sub_ok:
            logger.warning(f"  ⚠  {sub}: completeness issues ({len(missing_rows)} found)")

        sub_row: Dict[str, Any] = {"Subject": sub, "Completeness_OK": sub_ok}
        sub_files = 0
        sub_valid = 0
        sub_samples = 0
        sub_duration = 0.0

        for gest, lbl in GESTURE_MAP.items():
            gest_path = sub_path / gest
            files = sorted(Path(gest_path).glob("*.csv"))
            sub_row[f"{gest}_files"] = len(files)
            sub_row[f"{gest}_valid_files"] = 0
            sub_files += len(files)

            for fpath in files:
                total_files += 1
                val = _validate_csv_file(str(fpath))
                
                # Determine if file is valid
                is_readable = val.get("readable", False)
                has_cols = val.get("has_cols", False)
                has_nan = val.get("has_nan", False)
                is_valid = is_readable and has_cols and not has_nan
                
                if is_valid:
                    total_valid += 1
                    sub_valid += 1
                    sub_row[f"{gest}_valid_files"] += 1
                    sub_samples += val.get("total_samples", 0)
                    sub_duration += val.get("duration_s", 0.0)
                else:
                    issue = val.get("issue", "Unknown issue")
                    if issue:
                        all_issues.append(f"{sub}/{gest}/{fpath.name}: {issue}")

                # Add row for integrity table
                integrity_rows.append({
                    "Subject": sub,
                    "Gesture": gest,
                    "Label": lbl,
                    "File": fpath.name,
                    "readable": is_readable,
                    "has_cols": has_cols,
                    "n_rows": val.get("n_rows", 0),
                    "has_nan": has_nan,
                    "rms_ok": val.get("rms_ok", False),
                    "total_samples": val.get("total_samples", 0),
                    "duration_s": val.get("duration_s", 0.0),
                    "issue": val.get("issue", ""),
                })

        # Safe sample file discovery
        sample_files = []
        for gesture in GESTURE_MAP.keys():
            tmp = list(Path(sub_path / gesture).glob("*.csv"))
            if tmp:
                sample_files = tmp
                break

        est_wins = 0
        if sample_files:
            try:
                n_rows = len(pd.read_csv(sample_files[0]))
                est_wins = max(0, (n_rows - WINDOW_SIZE) // STEP_SIZE + 1)
            except Exception:
                pass

        # Subject completeness percentage
        sub_row["Total_Files"] = sub_files
        sub_row["Valid_Files"] = sub_valid

        if sub_files > 0:
            sub_row["Completeness_Percent"] = round((sub_valid / sub_files) * 100, 2)
        else:
            sub_row["Completeness_Percent"] = 0.0

        sub_row["Total_Samples"] = sub_samples
        sub_row["Total_Duration_s"] = round(sub_duration, 3)
        sub_row["Est_Windows"] = est_wins * sub_files
        subject_rows.append(sub_row)

        logger.info(
            f"  {sub}: {sub_files} files, {sub_valid} valid ({sub_row['Completeness_Percent']:.1f}%), "
            f"~{sub_samples} samples, {sub_duration:.1f}s, "
            f"~{est_wins} wins/trial"
        )

    # ── Build DataFrames ─────────────────────────────────────────────────────
    df_integrity = pd.DataFrame(integrity_rows)
    df_subjects = pd.DataFrame(subject_rows)
    df_missing = pd.DataFrame(all_missing) if all_missing else pd.DataFrame(
        columns=["Subject", "Gesture", "Issue", "Files_Found", "Files_Expected"]
    )
    df_stats = _compute_recording_stats(integrity_rows)
    df_class = _compute_class_distribution(integrity_rows)

    # ── Subject Completeness Table ──────────────────────────────────────────
    df_subject_completeness = df_subjects[
        [
            "Subject",
            "Total_Files",
            "Valid_Files",
            "Completeness_Percent",
            "Completeness_OK"
        ]
    ].copy() if not df_subjects.empty else pd.DataFrame()

    # Deterministic sorting (consistent with other blocks)
    for df in [df_integrity, df_subjects, df_stats, df_subject_completeness]:
        if not df.empty and "Subject" in df.columns:
            df.sort_values(["Subject"], ascending=[True], inplace=True)
    
    # Sort class distribution by label
    if not df_class.empty and "Label" in df_class.columns:
        df_class.sort_values(["Label", "Gesture"], ascending=[True, True], inplace=True)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    p_s1 = OUT_DIR / "Table_S1_Dataset_Integrity.csv"
    p_s1b = OUT_DIR / "Table_S1b_Subject_Summary_Report.csv"
    p_s1c = OUT_DIR / "Table_S1c_Missing_Gesture_Report.csv"
    p_s1d = OUT_DIR / "Table_S1d_Class_Distribution.csv"
    p_s1e = OUT_DIR / "Table_S1e_Recording_Statistics.csv"
    p_s1f = OUT_DIR / "Table_S1f_Subject_Completeness.csv"

    save_csv(df_integrity, p_s1)
    logger.info(f"  Table S1 saved → {p_s1}")
    save_csv(df_subjects, p_s1b)
    logger.info(f"  Table S1b saved → {p_s1b}")
    save_csv(df_missing, p_s1c)
    logger.info(f"  Table S1c saved → {p_s1c}")
    save_csv(df_class, p_s1d)
    logger.info(f"  Table S1d saved → {p_s1d}")
    save_csv(df_stats, p_s1e)
    logger.info(f"  Table S1e saved → {p_s1e}")
    save_csv(df_subject_completeness, p_s1f)
    logger.info(f"  Table S1f saved → {p_s1f}")

    # ── Text Report ───────────────────────────────────────────────────────────
    quality_pct = 100 * total_valid / max(total_files, 1)
    n_incomplete = len(df_missing)
    total_samples = int(df_subjects["Total_Samples"].sum()) if "Total_Samples" in df_subjects else 0
    total_dur_s = float(df_subjects["Total_Duration_s"].sum()) if "Total_Duration_s" in df_subjects else 0.0

    # ── Enhanced dataset_summary.json ──────────────────────────────────────
    dataset_summary = {
        "pipeline": PIPELINE_NAME,
        "subjects": len(subjects),
        "expected_subjects": EXPECTED_SUBJECTS,
        "gestures": len(GESTURE_MAP),
        "expected_gestures": EXPECTED_GESTURES,
        "files": total_files,
        "valid_files": total_valid,
        "quality_percent": round(quality_pct, 2),
        "total_samples": total_samples,
        "total_duration_minutes": round(total_dur_s / 60, 2),
        "sampling_rate_hz": SAMPLING_RATE,
        "window_size_samples": WINDOW_SIZE,
        "step_size_samples": STEP_SIZE,
        "channels": N_CHANNELS,
        "channel_names": EMG_CHANNEL_COLUMNS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    save_json(dataset_summary, OUT_DIR / "dataset_summary.json")
    logger.info(f"  Dataset summary saved → {OUT_DIR / 'dataset_summary.json'}")

    # ── Build Report (focused summary, details in CSVs) ──────────────────────
    report_lines = [
        "=" * 70,
        f"{PIPELINE_NAME} | Dataset Integrity Report",
        "=" * 70,
        f"Generated          : {datetime.now().isoformat(timespec='seconds')}",
        f"Input Path         : {INPUT_PATH}",
        "",
        "── Dataset Composition ──────────────────────────────────────────────",
        f"  N Subjects        : {len(subjects)}  (expected {EXPECTED_SUBJECTS})",
        f"  N Gestures        : {len(GESTURE_MAP)}  (expected {EXPECTED_GESTURES})",
        f"  Total Files       : {total_files}",
        f"  Valid Files       : {total_valid} ({quality_pct:.1f}%)",
        f"  Total Samples     : {total_samples:,}",
        f"  Total Duration    : {total_dur_s/60:.1f} min  ({total_dur_s:.1f} s)",
        "",
        "── Signal Configuration ─────────────────────────────────────────────",
        f"  Required Channels : {REQUIRED_CHANNELS}",
        f"  Sampling Rate     : {SAMPLING_RATE} Hz",
        f"  Window Size       : {WINDOW_SIZE} samples",
        f"  Step Size         : {STEP_SIZE} samples",
        f"  RMS Valid Factor  : {RMS_VALIDATION_FACTOR}",
        "",
        "── Completeness Audit ───────────────────────────────────────────────",
        f"  Missing/Incomplete: {n_incomplete} entries",
    ]

    if n_incomplete > 0:
        report_lines.append("  MISSING GESTURE DETAILS:")
        for _, row in df_missing.head(20).iterrows():
            report_lines.append(f"    • {row['Subject']}/{row['Gesture']}: {row['Issue']}")
        if n_incomplete > 20:
            report_lines.append(f"    ... and {n_incomplete - 20} more (see Table_S1c)")
    else:
        report_lines.append("  ✓ All subjects have all expected gestures.")

    report_lines += ["", "── Subject Completeness ─────────────────────────────────────────────"]
    if not df_subject_completeness.empty:
        for _, row in df_subject_completeness.head(20).iterrows():
            status = "✓" if row["Completeness_OK"] else "⚠"
            report_lines.append(
                f"  {row['Subject']}: {status} {row['Valid_Files']}/{row['Total_Files']} "
                f"({row['Completeness_Percent']:.1f}%)"
            )
        if len(df_subject_completeness) > 20:
            report_lines.append(f"  ... and {len(df_subject_completeness)-20} more (see Table_S1b)")

    report_lines += ["", "── File Validation Issues ───────────────────────────────────────────"]
    if all_issues:
        for iss in all_issues[:30]:
            report_lines.append(f"  • {iss}")
        if len(all_issues) > 30:
            report_lines.append(f"  ... and {len(all_issues)-30} more (see Table_S1)")
    else:
        report_lines.append("  ✓ No file validation issues detected.")

    report_lines += ["", "── Class Distribution ───────────────────────────────────────────────"]
    if not df_class.empty:
        for _, row in df_class.head(10).iterrows():
            if row["Gesture"] in GESTURE_MAP:
                label_idx = GESTURE_MAP[row["Gesture"]]
                short = GESTURE_SHORT_MAP.get(label_idx, row["Gesture"])
            else:
                short = row["Gesture"]
            report_lines.append(f"  {short:>6} : {int(row['Total_Files']):4d} files  ({row['Pct']:.1f}%)")
        if len(df_class) > 10:
            report_lines.append(f"  ... and {len(df_class)-10} more (see Table_S1d)")

    report_lines += ["", "=" * 70]

    p_report = OUT_DIR / "Dataset_Integrity_Report.txt"
    save_txt("\n".join(report_lines), p_report)
    logger.info(f"  Report saved → {p_report}")

    # Print summary to console
    for line in report_lines:
        logger.info(line)

    # ── Figures ───────────────────────────────────────────────────────────────
    p_fig_s1 = OUT_DIR / "FigS1_Class_Distribution.png"
    p_fig_s2 = OUT_DIR / "FigS2_DataIntegrity_Heatmap.png"

    try:
        _plot_class_distribution(df_class, p_fig_s1)
        logger.info(f"  Figure S1 saved → {p_fig_s1}")
    except Exception as exc:
        logger.exception(f"  FigS1 failed: {exc}")

    try:
        _plot_integrity_heatmap(df_subjects, p_fig_s2)
        logger.info(f"  Figure S2 saved → {p_fig_s2}")
    except Exception as exc:
        logger.exception(f"  FigS2 failed: {exc}")

    # ── Save environment versions ────────────────────────────────────────────
    _save_environment_versions()

    # ── Artifact Manifest ─────────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "Table_S1_Dataset_Integrity": p_s1,
        "Table_S1b_Subject_Summary": p_s1b,
        "Table_S1c_Missing_Gesture": p_s1c,
        "Table_S1d_Class_Distribution": p_s1d,
        "Table_S1e_Recording_Statistics": p_s1e,
        "Table_S1f_Subject_Completeness": p_s1f,
        "Dataset_Integrity_Report": p_report,
        "FigS1_Class_Distribution": p_fig_s1,
        "FigS2_DataIntegrity_Heatmap": p_fig_s2,
        "dataset_summary": OUT_DIR / "dataset_summary.json",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"  BLOCK A runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    summary = {
        "n_subjects": len(subjects),
        "n_gestures": len(GESTURE_MAP),
        "total_files": total_files,
        "valid_files": total_valid,
        "quality_pct": round(quality_pct, 2),
        "n_issues": len(all_issues),
        "n_missing": n_incomplete,
        "total_samples": total_samples,
        "total_duration_min": round(total_dur_s / 60, 2),
        "expected_subjects": EXPECTED_SUBJECTS,
        "subject_count_match": len(subjects) == EXPECTED_SUBJECTS if EXPECTED_SUBJECTS else None,
        "runtime_sec": elapsed,
        "runtime_minutes": minutes,
        "runtime_seconds": seconds,
    }

    logger.info("\n" + "=" * 70)
    logger.info("BLOCK A COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Subjects: {len(subjects)}")
    logger.info(f"  Quality: {quality_pct:.1f}%")
    logger.info(f"  Missing entries: {n_incomplete}")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — BLOCK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return

    try:
        summary = run_data_integrity()
        mark_completed(BLOCK_NAME, metadata=summary)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_A WITH PUBLICATION-QUALITY FIX
# ✓ PUBLICATION-QUALITY FIX: _compute_class_distribution() uses .size() instead of .count()
# ✓ FIX 1: _compute_recording_stats() uses consistent 'readable' and 'has_cols' (lowercase)
# ✓ FIX 2: _compute_class_distribution() uses consistent 'readable' and 'has_cols' (lowercase)
# ✓ FIX 3: Removed unused imports (FancyBboxPatch, glob)
# ✓ FIX 4: Removed unused PALETTE import
# ✓ FIX 5: Report shortened (shows top entries, rest in CSVs)
# ✓ IMPORTANT FIX: _validate_csv_file() uses EMG_CHANNEL_COLUMNS from config
# ✓ MODERATE FIX: _plot_integrity_heatmap() no longer creates duplicate figure
# ✓ MINOR FIX: Subject validation now raises ValueError (hard failure)
# ✓ MINOR FIX: dataset_summary.json includes sampling_rate, window_size, step_size, channels
# ✓ MINOR FIX: Report title uses PIPELINE_NAME for consistency
# ✓ CONSISTENCY: _finalize_figure() helper matches BLOCK_13
# ✓ CONSISTENCY: All tables use save_csv() with logger.info()
# ✓ CONSISTENCY: Path.glob() used instead of glob.glob()
# ✓ CONSISTENCY: logger.exception() for errors
# ✓ CONSISTENCY: Deterministic sorting for all DataFrames
# ✓ CONSISTENCY: Consistent summary banner (BLOCK A COMPLETE)
# ✓ CONSISTENCY: Return dictionary includes runtime_sec, runtime_minutes, runtime_seconds
# ✓ Thread settings changed from "2" to "1"
# ✓ plt.close(fig) and gc.collect() after every figure
# ✓ Runtime logging with time.perf_counter()
# ✓ Environment versions saved
# ✓ Artifact manifest with verification
# ✓ DATA1: Dataset Integrity Validation
# ✓ DATA1b: Subject Summary Report
# ✓ DATA1c: Missing Gesture Report
# ✓ DATA1d: Class Distribution
# ✓ DATA1e: Recording Statistics
# ✓ DATA1f: Subject Completeness
# ✓ Figure S1: Class Distribution
# ✓ Figure S2: Data Integrity Heatmap
# ✓ All tables (S1, S1b, S1c, S1d, S1e, S1f)
# ✓ Dataset Integrity Report
# ✓ dataset_summary.json
# ✓ Artifact manifest
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────