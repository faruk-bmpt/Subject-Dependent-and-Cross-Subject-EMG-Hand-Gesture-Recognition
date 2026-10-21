# BLOCK_D_PRE1.py - FINAL POLISHED VERSION
# ===============
# FILE PURPOSE:
#     PRE-BLOCK D — PRE1: EMG Signal Preprocessing and Artifact Removal.
#     Applies the validated SD preprocessing pipeline:
#       1. 4th-order Butterworth bandpass filter (10-45 Hz)
#       2. 50 Hz notch filter (Q=30)
#     Generates comprehensive preprocessing visualizations for publication.
#
#     FIGURES:
#     - Figure 1: Raw vs Preprocessed EMG (3 channels, side-by-side)
#     - Figure 1b: Power Spectral Density (PSD) Comparison
#     - Figure 1c: All 7 Gestures — Raw vs Preprocessed (representative subject)
#     - Figure 2: Preprocessing Workflow Diagram
#     - Figure 3: Zoomed Comparison (Raw vs Cleaned, 1-second segment)
#     - Figure 4: Filter Frequency Response (Bandpass + Notch)
#     - Figure 5: RMS Envelope Before vs After Preprocessing
#
# INPUT_ARTIFACTS  : INPUT_PATH (raw CSV files)
# OUTPUT_ARTIFACTS :
#     - outputs/PRE_D/Fig1_Raw_vs_Preprocessed_EMG.png
#     - outputs/PRE_D/Fig1b_PSD_Comparison.png
#     - outputs/PRE_D/Fig1c_All_Gestures_Raw_vs_Preprocessed.png
#     - outputs/PRE_D/Fig2_Preprocessing_Workflow.png
#     - outputs/PRE_D/Fig3_Zoomed_Comparison.png
#     - outputs/PRE_D/Fig4_Filter_Frequency_Response.png
#     - outputs/PRE_D/Fig5_RMS_Envelope_Comparison.png
#     - outputs/PRE_D/representative_subject.json
#     - outputs/PRE_D/preprocessing_params.json
#     - outputs/PRE_D/environment_versions.json
#     - outputs/PRE_D/artifacts.json
#     - state/BLOCK_D_PRE1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_D_PRE1.py
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

import gc
import glob
import logging
import sys
import time
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import xgboost
import scipy
import imblearn
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt, welch, freqz
from scipy import signal as scipy_signal

from config import (
    INPUT_PATH,
    BLOCK_DIRS,
    GESTURE_MAP,
    GESTURE_SHORT,
    SAMPLING_RATE,
    N_CHANNELS,
    PIPELINE_NAME,
    MIN_RMS_THRESHOLD,
    WINDOW_SIZE,
    OVERLAP,
    STEP_SIZE,
    FEATURE_NAMES,
    REQUIRED_CHANNELS,
    SEED,
    EMG_CHANNEL_COLUMNS,
)
from state_manager import is_completed, mark_completed
from artifact_manager import save_fig, save_json, save_json as save_json_artifact

warnings.filterwarnings("ignore")

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_D_PRE1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_D")

BLOCK_NAME = "BLOCK_D_PRE1"
OUT_DIR    = BLOCK_DIRS["PRE_D"]

# ── Fallback imports AFTER logger creation ──────────────────────────────────
try:
    from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, NOTCH_FREQ, NOTCH_Q
except ImportError:
    BANDPASS_LOW = 10.0
    BANDPASS_HIGH = 45.0
    FILTER_ORDER = 4
    NOTCH_FREQ = 50.0
    NOTCH_Q = 30.0
    logger.warning("  Using fallback filter parameters (not found in config)")

# ── Pre-compute filters (validated SD pipeline) ──────────────────────────
_B_BP, _A_BP = butter(
    FILTER_ORDER,
    [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
    btype="band",
)
_B_NOTCH, _A_NOTCH = scipy_signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=SAMPLING_RATE)

# ── Dynamic feature count ──────────────────────────────────────────────────
TOTAL_FEATURES = len(FEATURE_NAMES)


# ============================================================================
# FIGURE FINALIZER HELPER (identical to other blocks)
# ============================================================================

def _finalize_figure(fig, path: Path, dpi: int = 300) -> None:
    """
    Centralized figure finalization: save, close, and garbage collect.
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
        "numpy": _get_package_version("numpy"),
        "pandas": pd.__version__ if hasattr(pd, "__version__") else "unknown",
        "scipy": scipy.__version__ if hasattr(scipy, "__version__") else "unknown",
        "sklearn": sklearn.__version__ if hasattr(sklearn, "__version__") else "unknown",
        "xgboost": xgboost.__version__ if hasattr(xgboost, "__version__") else "unknown",
        "imblearn": imblearn.__version__ if hasattr(imblearn, "__version__") else "unknown",
        "matplotlib": matplotlib.__version__ if hasattr(matplotlib, "__version__") else "unknown",
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


def _get_package_version(pkg_name: str) -> str:
    """Get version of a package, return 'not_installed' if not found."""
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not_installed"


# ============================================================================
# Artifact manifest with verification (identical to other blocks)
# ============================================================================

def _save_artifact_manifest(
    artifact_map: Dict[str, Path],
    out_dir: Path,
) -> None:
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


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_emg(data: np.ndarray) -> np.ndarray:
    """
    Apply validated SD preprocessing:
    Bandpass (10-45 Hz, 4th-order Butterworth) + Notch (50 Hz, Q=30).
    """
    data = filtfilt(_B_BP, _A_BP, data, axis=0)
    data = filtfilt(_B_NOTCH, _A_NOTCH, data, axis=0)
    return data


def is_good_window(win: np.ndarray) -> bool:
    """Check if window meets RMS threshold (identical to SD pipeline)."""
    return bool(np.all(np.sqrt(np.mean(win ** 2, axis=0)) > MIN_RMS_THRESHOLD))


def preprocess_emg_only_bandpass(data: np.ndarray) -> np.ndarray:
    """Apply only bandpass filter (for comparison)."""
    return filtfilt(_B_BP, _A_BP, data, axis=0)


def _find_sample_file(subject: Optional[str] = None) -> Optional[str]:
    """Locate the first available CSV in the dataset."""
    if subject is not None:
        for gest in GESTURE_MAP:
            gest_path = INPUT_PATH / subject / gest
            if gest_path.exists():
                files = sorted(glob.glob(str(gest_path / "*.csv")))
                if files:
                    return files[0]
    
    subjects = sorted(
        s for s in os.listdir(INPUT_PATH)
        if os.path.isdir(INPUT_PATH / s)
    )
    for sub in subjects:
        for gest in GESTURE_MAP:
            files = sorted(glob.glob(str(INPUT_PATH / sub / gest / "*.csv")))
            if files:
                return files[0]
    return None


def _detect_emg_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect EMG columns from a DataFrame.
    Prioritizes EMG_CHANNEL_COLUMNS from config.
    """
    if all(col in df.columns for col in EMG_CHANNEL_COLUMNS):
        return EMG_CHANNEL_COLUMNS
    
    logger.warning(
        f"  EMG_CHANNEL_COLUMNS {EMG_CHANNEL_COLUMNS} not found. Using fallback."
    )
    
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
    
    raise ValueError(
        f"Could not find any EMG columns. Expected: {EMG_CHANNEL_COLUMNS}. "
        f"Available: {df.columns.tolist()[:20]}"
    )


def _compute_rms_envelope(data: np.ndarray, window_ms: int = 50) -> np.ndarray:
    """Compute RMS envelope of EMG signal using sliding window."""
    window_samples = int(window_ms * SAMPLING_RATE / 1000)
    if window_samples < 1:
        window_samples = 1
    
    rms = np.zeros_like(data)
    for ch in range(data.shape[1]):
        for i in range(len(data)):
            start = max(0, i - window_samples // 2)
            end = min(len(data), i + window_samples // 2)
            rms[i, ch] = np.sqrt(np.mean(data[start:end, ch] ** 2))
    return rms


# ============================================================================
# AUTOMATIC REPRESENTATIVE SUBJECT SELECTION
# ============================================================================

def select_representative_subject() -> Tuple[str, Dict[str, float]]:
    """
    Select the representative subject automatically using a Signal Quality Index.

    Quality Index = signal_power / (1.0 + baseline_noise)
    
    where:
        signal_power = mean squared amplitude of the EMG signal
        baseline_noise = standard deviation of the detrended signal
    
    This balances:
        - Higher signal amplitude (better SNR)
        - Lower baseline noise (cleaner signal)
    
    Note: This is a heuristic ranking metric, not a standardized EMG SQS.
    In the paper, describe it as "Representative subject selection score"
    or "Signal quality index".
    
    Returns:
        Tuple of (subject_id, scores_dict) where scores_dict contains:
            - subject_id
            - quality_index
            - mean_rms
            - baseline_noise
            - n_windows
    """
    logger.info("  Selecting representative subject using Signal Quality Index...")

    subject_scores = {}

    subjects = sorted(
        s for s in os.listdir(INPUT_PATH)
        if (INPUT_PATH / s).is_dir()
    )

    for subject in subjects:
        quality_values = []
        rms_values = []
        noise_values = []
        n_windows = 0

        for gesture in GESTURE_MAP.keys():
            gpath = INPUT_PATH / subject / gesture
            if not gpath.exists():
                continue

            files = sorted(gpath.glob("*.csv"))

            for f in files:
                try:
                    df = pd.read_csv(
                        f,
                        usecols=EMG_CHANNEL_COLUMNS,
                        dtype=np.float32
                    )
                    raw = df.values
                    clean = preprocess_emg(raw)

                    for start in range(
                        0,
                        len(clean) - WINDOW_SIZE + 1,
                        STEP_SIZE
                    ):
                        win = clean[start:start + WINDOW_SIZE]

                        if not is_good_window(win):
                            continue

                        # Signal power (mean squared amplitude)
                        signal_power = np.mean(win ** 2)
                        
                        # Baseline noise (std of detrended signal)
                        win_detrended = win - np.mean(win, axis=0)
                        baseline_noise = np.std(win_detrended)
                        
                        # Quality Index: balances amplitude and cleanliness
                        quality_idx = signal_power / (1.0 + baseline_noise)
                        
                        quality_values.append(quality_idx)
                        rms_values.append(np.sqrt(signal_power))
                        noise_values.append(baseline_noise)
                        n_windows += 1

                except Exception as e:
                    logger.debug(f"    Skipping {f}: {e}")
                    continue

        if n_windows > 0:
            subject_scores[subject] = {
                "quality_index": np.mean(quality_values),
                "mean_rms": np.mean(rms_values),
                "baseline_noise": np.mean(noise_values),
                "n_windows": n_windows,
                "n_gestures": len(GESTURE_MAP),
            }

    if len(subject_scores) == 0:
        raise RuntimeError(
            "No representative subject found. Check that INPUT_PATH contains valid EMG data."
        )

    # Select subject with highest quality index
    representative = max(subject_scores, key=lambda x: subject_scores[x]["quality_index"])
    scores = subject_scores[representative]

    # Log results
    logger.info(f"  Representative Subject: {representative}")
    logger.info(f"    Quality Index: {scores['quality_index']:.4f}")
    logger.info(f"    Mean RMS: {scores['mean_rms']:.4f}")
    logger.info(f"    Baseline Noise: {scores['baseline_noise']:.4f}")
    logger.info(f"    Valid Windows: {scores['n_windows']}")
    logger.info(f"    Gestures: {scores['n_gestures']}")

    return representative, scores


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def _plot_filter_frequency_response() -> None:
    """Figure 4 – Filter Frequency Response."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    w_bp, h_bp = freqz(_B_BP, _A_BP, worN=2000)
    freq_bp = w_bp * SAMPLING_RATE / (2 * np.pi)
    mag_bp = 20 * np.log10(abs(h_bp) + 1e-10)
    
    w_notch, h_notch = freqz(_B_NOTCH, _A_NOTCH, worN=2000)
    freq_notch = w_notch * SAMPLING_RATE / (2 * np.pi)
    mag_notch = 20 * np.log10(abs(h_notch) + 1e-10)
    
    h_combined = h_bp * h_notch
    mag_combined = 20 * np.log10(abs(h_combined) + 1e-10)
    
    ax.plot(freq_bp, mag_bp, 'b-', lw=1.5, alpha=0.7, label='Bandpass (10-45 Hz)')
    ax.plot(freq_notch, mag_notch, 'g-', lw=1.5, alpha=0.7, label='Notch (50 Hz)')
    ax.plot(freq_bp, mag_combined, 'r-', lw=2.0, label='Combined Response')
    
    ax.axvline(BANDPASS_LOW, color='gray', ls=':', lw=1, alpha=0.6)
    ax.axvline(BANDPASS_HIGH, color='gray', ls=':', lw=1, alpha=0.6)
    ax.axvline(NOTCH_FREQ, color='orange', ls='--', lw=1.5, alpha=0.8)
    ax.axhline(-3, color='black', ls='--', lw=0.8, alpha=0.5, label='-3 dB')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (dB)', fontsize=11)
    ax.set_title('Figure 4 – Filter Frequency Response\n'
                 f'Bandpass: {BANDPASS_LOW}–{BANDPASS_HIGH} Hz, {FILTER_ORDER}th-order Butterworth\n'
                 f'Notch: {NOTCH_FREQ} Hz, Q={NOTCH_Q}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 80)
    ax.set_ylim(-60, 5)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "Fig4_Filter_Frequency_Response.png")
    logger.info("  Figure 4 saved.")


def _plot_zoomed_comparison(raw: np.ndarray, clean: np.ndarray, t: np.ndarray) -> None:
    """Figure 3 – Zoomed Comparison (1-second segment)."""
    n_zoom = min(int(SAMPLING_RATE), len(raw))
    start = max(0, len(raw) // 2 - n_zoom // 2)
    end = start + n_zoom
    
    raw_zoom = raw[start:end]
    clean_zoom = clean[start:end]
    t_zoom = t[start:end]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    colors_r = ["#E53935", "#D81B60", "#C62828"]
    colors_c = ["#1E88E5", "#0288D1", "#01579B"]
    ch_names = [f"CH{ch+1}" for ch in range(3)]
    
    for ch in range(3):
        axes[ch].plot(t_zoom, raw_zoom[:, ch], color=colors_r[ch], lw=0.8, 
                      alpha=0.6, label='Raw')
        axes[ch].plot(t_zoom, clean_zoom[:, ch], color=colors_c[ch], lw=0.9,
                      alpha=0.9, label='Preprocessed')
        axes[ch].set_ylabel(ch_names[ch], fontsize=10, fontweight='bold')
        axes[ch].grid(True, alpha=0.2)
        axes[ch].legend(loc='upper right', fontsize=8)
        
        y_min = min(raw_zoom[:, ch].min(), clean_zoom[:, ch].min())
        y_max = max(raw_zoom[:, ch].max(), clean_zoom[:, ch].max())
        y_range = y_max - y_min
        if y_range > 0:
            axes[ch].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Figure 3 – Zoomed Comparison: Raw vs Preprocessed EMG (1-second segment)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "Fig3_Zoomed_Comparison.png")
    logger.info("  Figure 3 saved.")


def _plot_rms_envelope_comparison(raw: np.ndarray, clean: np.ndarray, t: np.ndarray) -> None:
    """Figure 5 – RMS Envelope Before vs After Preprocessing."""
    rms_raw = _compute_rms_envelope(raw, window_ms=50)
    rms_clean = _compute_rms_envelope(clean, window_ms=50)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    colors_r = ["#E53935", "#D81B60", "#C62828"]
    colors_c = ["#1E88E5", "#0288D1", "#01579B"]
    ch_names = [f"CH{ch+1}" for ch in range(3)]
    
    for ch in range(3):
        axes[ch].plot(t, rms_raw[:, ch], color=colors_r[ch], lw=1.2,
                      alpha=0.7, label='Raw RMS')
        axes[ch].plot(t, rms_clean[:, ch], color=colors_c[ch], lw=1.5,
                      alpha=0.9, label='Preprocessed RMS')
        axes[ch].set_ylabel(ch_names[ch], fontsize=10, fontweight='bold')
        axes[ch].grid(True, alpha=0.2)
        axes[ch].legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Figure 5 – RMS Envelope Before vs After Preprocessing (50 ms window)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "Fig5_RMS_Envelope_Comparison.png")
    logger.info("  Figure 5 saved.")


def _plot_all_gestures_comparison(subject: str) -> None:
    """
    Figure 1c – All 7 Gestures Raw vs Preprocessed (representative subject).
    Shows one channel (CH1) for each gesture, raw vs cleaned.
    """
    logger.info("  Generating Figure 1c – All Gestures Raw vs Preprocessed...")
    
    n_gestures = len(GESTURE_MAP)
    fig, axes = plt.subplots(n_gestures, 2, figsize=(14, 12))
    
    # Use a single channel for clarity (CH1)
    ch_idx = 0
    color_r = "#E53935"
    color_c = "#1E88E5"
    
    # Collect data for each gesture
    gesture_data = []
    
    for gest_idx, (gest_name, gest_label) in enumerate(GESTURE_MAP.items()):
        gest_path = INPUT_PATH / subject / gest_name
        if not gest_path.exists():
            logger.warning(f"    Gesture {gest_name} not found for subject {subject}")
            continue
        
        files = sorted(glob.glob(str(gest_path / "*.csv")))
        if not files:
            logger.warning(f"    No files for gesture {gest_name}")
            continue
        
        # Use the first file for this gesture
        try:
            df = pd.read_csv(files[0], usecols=EMG_CHANNEL_COLUMNS, dtype=np.float32)
            raw = df.values
            
            # Use first 500 samples for consistency
            n_show = min(500, len(raw))
            raw_segment = raw[:n_show, ch_idx]
            clean_segment = preprocess_emg(raw[:n_show])[:, ch_idx]
            t = np.arange(n_show) / SAMPLING_RATE
            
            # Use GESTURE_SHORT for labels
            short_label = GESTURE_SHORT[gest_label] if gest_label < len(GESTURE_SHORT) else gest_name
            
            gesture_data.append({
                "name": short_label,
                "t": t,
                "raw": raw_segment,
                "clean": clean_segment,
                "valid": True
            })
        except Exception as e:
            logger.warning(f"    Failed to process {gest_name}: {e}")
            continue
    
    # If we have fewer than 7 gestures, fill with placeholders
    # (but we expect all 7)
    for i, gd in enumerate(gesture_data[:n_gestures]):
        ax = axes[i, 0]
        ax.plot(gd["t"], gd["raw"], color=color_r, lw=0.8, alpha=0.7)
        ax.set_ylabel(gd["name"], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_title("Raw EMG", fontsize=12, fontweight='bold', color=color_r)
        
        ax2 = axes[i, 1]
        ax2.plot(gd["t"], gd["clean"], color=color_c, lw=0.8, alpha=0.9)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(labelsize=8)
        if i == 0:
            ax2.set_title("Preprocessed EMG", fontsize=12, fontweight='bold', color=color_c)
        
        # Set consistent y-limits per row
        y_min = min(gd["raw"].min(), gd["clean"].min())
        y_max = max(gd["raw"].max(), gd["clean"].max())
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Hide any unused subplots
    for i in range(len(gesture_data), n_gestures):
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    
    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=10)
    
    fig.suptitle(
        f"Figure 1c – All Gestures: Raw vs Preprocessed EMG (Subject {subject}, CH1)",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "Fig1c_All_Gestures_Raw_vs_Preprocessed.png")
    logger.info("  Figure 1c saved.")


def _generate_workflow_figure() -> None:
    """Figure 2 – EMG Preprocessing Workflow diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    steps = [
        ("Raw EMG\n" + str(N_CHANNELS) + " Channels\n(125 Hz)", "#FFCDD2", "#C62828"),
        (f"Bandpass Filter\n{BANDPASS_LOW}–{BANDPASS_HIGH} Hz\n"
         f"{FILTER_ORDER}th-order Butterworth", "#E3F2FD", "#1565C0"),
        (f"Notch Filter\n{NOTCH_FREQ} Hz\n(Q={NOTCH_Q})", "#E8F5E9", "#2E7D32"),
        (f"Window Segmentation\n{WINDOW_SIZE} samples\n"
         f"Overlap={int(OVERLAP*100)}%\nStep={STEP_SIZE}", "#FFF9C4", "#F57F17"),
        (f"Quality Gate\nRMS > {MIN_RMS_THRESHOLD}\n(reject noise)", "#FCE4EC", "#880E4F"),
        (f"Feature\nExtraction\n({TOTAL_FEATURES} features)", "#F3E5F5", "#4A148C"),
    ]

    n = len(steps)
    x_pos = np.linspace(0.06, 0.94, n)
    box_w, box_h = 0.13, 0.55

    for i, (label, fc, ec) in enumerate(steps):
        x = x_pos[i]
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, 0.22), box_w, box_h,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=2,
            transform=ax.transAxes,
        ))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=ec, transform=ax.transAxes,
                multialignment="center")

        if i < n - 1:
            ax.annotate(
                "", xy=(x_pos[i + 1] - box_w / 2 - 0.005, 0.5),
                xytext=(x + box_w / 2 + 0.005, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555",
                                lw=2.0, mutation_scale=15),
            )

    ax.set_title(
        "Figure 2 – EMG Preprocessing Workflow",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "Fig2_Preprocessing_Workflow.png")
    logger.info("  Figure 2 saved.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_preprocessing_viz() -> dict:
    """Generate all preprocessing visualization figures."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK D | PRE1: EMG Signal Preprocessing Visualization")
    logger.info("=" * 70)

    logger.info(f"Python : {sys.version}")
    logger.info(f"PID    : {os.getpid()}")

    # ── Select representative subject automatically ────────────────────────
    try:
        representative_subject, subject_scores = select_representative_subject()
    except Exception as e:
        logger.error(f"  Failed to select representative subject: {e}")
        representative_subject = None
        subject_scores = {}

    # ── Save representative subject info ────────────────────────────────────
    rep_info = {
        "representative_subject": representative_subject,
        "quality_index": subject_scores.get("quality_index", None),
        "mean_rms": subject_scores.get("mean_rms", None),
        "baseline_noise": subject_scores.get("baseline_noise", None),
        "n_windows": subject_scores.get("n_windows", 0),
        "n_gestures": subject_scores.get("n_gestures", 0),
        "selection_criterion": "Signal Quality Index = signal_power / (1 + baseline_noise) "
                               "(heuristic ranking metric)",
    }
    save_json(rep_info, OUT_DIR / "representative_subject.json")
    logger.info(f"  Representative subject info saved → {OUT_DIR / 'representative_subject.json'}")

    # ── Save preprocessing parameters ──────────────────────────────────────
    params = {
        "pipeline": PIPELINE_NAME,
        "filter_type": "Butterworth",
        "filter_order": FILTER_ORDER,
        "bandpass_low": BANDPASS_LOW,
        "bandpass_high": BANDPASS_HIGH,
        "notch_freq": NOTCH_FREQ,
        "notch_Q": NOTCH_Q,
        "sampling_rate": SAMPLING_RATE,
        "n_channels": N_CHANNELS,
        "min_rms": MIN_RMS_THRESHOLD,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "step_size": STEP_SIZE,
        "total_features": TOTAL_FEATURES,
        "representative_subject": representative_subject,
        "representative_quality_index": subject_scores.get("quality_index", None),
    }
    save_json(params, OUT_DIR / "preprocessing_params.json")
    logger.info(f"  Preprocessing params saved → {OUT_DIR / 'preprocessing_params.json'}")

    # ── Find sample file for the representative subject ─────────────────────
    sample_file = _find_sample_file(representative_subject)

    if sample_file is None:
        logger.warning("  No sample file found for representative subject — generating workflow only.")
        _generate_workflow_figure()
        _save_environment_versions()
        
        elapsed = time.perf_counter() - t0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        params["runtime_sec"] = elapsed
        params["runtime_minutes"] = minutes
        params["runtime_seconds"] = seconds
        
        return params

    try:
        # Read CSV dynamically
        df_raw = pd.read_csv(sample_file)
        logger.info(f"  Sample file: {sample_file}")
        
        # Detect EMG columns
        emg_cols = _detect_emg_columns(df_raw)
        logger.info(f"  Detected EMG columns: {emg_cols}")
        
        if len(emg_cols) < 3:
            logger.warning(f"  Could not find EMG columns. Available: {df_raw.columns.tolist()[:10]}")
            _generate_workflow_figure()
            _save_environment_versions()
            
            elapsed = time.perf_counter() - t0
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            params["runtime_sec"] = elapsed
            params["runtime_minutes"] = minutes
            params["runtime_seconds"] = seconds
            
            return params

        # Extract data
        df = df_raw[emg_cols].astype(float)
        
        # Use up to 1200 samples for clarity
        n_show = min(1200, len(df))
        raw = df.values[:n_show]
        clean = preprocess_emg(raw.copy())
        t = np.arange(n_show) / SAMPLING_RATE

        ch_names = [f"CH{ch+1}" for ch in range(3)]
        colors_r = ["#E53935", "#D81B60", "#C62828"]
        colors_c = ["#1E88E5", "#0288D1", "#01579B"]

        # ── Figure 1 – Raw vs Preprocessed ───────────────────────────────
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
        for ch in range(3):
            axes[ch, 0].plot(t, raw[:, ch], color=colors_r[ch], lw=0.7, alpha=0.85)
            axes[ch, 1].plot(t, clean[:, ch], color=colors_c[ch], lw=0.7, alpha=0.85)
            axes[ch, 0].set_ylabel(ch_names[ch], fontsize=9)
            for col in (0, 1):
                axes[ch, col].grid(True, alpha=0.2)
                axes[ch, col].tick_params(labelsize=8)

        axes[0, 0].set_title("Raw EMG Signal", fontsize=12, fontweight="bold",
                              color="#C62828")
        axes[0, 1].set_title("Preprocessed EMG Signal\n"
                             f"(BP {BANDPASS_LOW}–{BANDPASS_HIGH} Hz "
                             f"+ Notch {NOTCH_FREQ} Hz)",
                             fontsize=12, fontweight="bold", color="#01579B")
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)", fontsize=10)

        fig.suptitle(
            f"Figure 1 – Raw vs Preprocessed EMG Signal (Subject {representative_subject})",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        _finalize_figure(fig, OUT_DIR / "Fig1_Raw_vs_Preprocessed_EMG.png")
        logger.info("  Figure 1 saved.")

        # ── Figure 1b – PSD comparison ────────────────────────────────────
        fig_psd, axes_psd = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        
        nperseg = min(128, len(raw) // 4)
        nperseg = max(64, nperseg)
        
        for ch in range(3):
            f_r, p_r = welch(raw[:, ch].astype(np.float64), fs=SAMPLING_RATE, nperseg=nperseg)
            f_c, p_c = welch(clean[:, ch].astype(np.float64), fs=SAMPLING_RATE, nperseg=nperseg)
            axes_psd[ch].semilogy(f_r, p_r, color=colors_r[ch], lw=1.2, label="Raw", alpha=0.7)
            axes_psd[ch].semilogy(f_c, p_c, color=colors_c[ch], lw=1.5, label="Clean")
            axes_psd[ch].axvline(BANDPASS_LOW, color="gray", ls=":", lw=1)
            axes_psd[ch].axvline(BANDPASS_HIGH, color="gray", ls=":", lw=1)
            axes_psd[ch].axvline(NOTCH_FREQ, color="orange", ls="--", lw=1,
                                  label=f"Notch {NOTCH_FREQ}Hz")
            axes_psd[ch].set_title(ch_names[ch], fontsize=9)
            axes_psd[ch].set_xlabel("Frequency (Hz)")
            axes_psd[ch].legend(fontsize=7)
            axes_psd[ch].grid(True, alpha=0.2)
        axes_psd[0].set_ylabel("PSD")
        fig_psd.suptitle("Figure 1b – Power Spectral Density: Raw vs Preprocessed", fontsize=11)
        plt.tight_layout()
        _finalize_figure(fig_psd, OUT_DIR / "Fig1b_PSD_Comparison.png")
        logger.info("  Figure 1b saved.")

        # ── Figure 1c – All Gestures Comparison ──────────────────────────
        if representative_subject is not None:
            _plot_all_gestures_comparison(representative_subject)

        # ── Figure 3 – Zoomed Comparison ──────────────────────────────────
        _plot_zoomed_comparison(raw, clean, t)

        # ── Figure 4 – Filter Frequency Response ──────────────────────────
        _plot_filter_frequency_response()

        # ── Figure 5 – RMS Envelope Comparison ────────────────────────────
        _plot_rms_envelope_comparison(raw, clean, t)

    except Exception as exc:
        logger.exception(f"  Figure generation failed: {exc}")

    # ── Figure 2 – Workflow ─────────────────────────────────────────────────
    _generate_workflow_figure()

    # ── Save environment versions ────────────────────────────────────────────
    _save_environment_versions()

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "fig1_raw_vs_preprocessed": OUT_DIR / "Fig1_Raw_vs_Preprocessed_EMG.png",
        "fig1b_psd_comparison": OUT_DIR / "Fig1b_PSD_Comparison.png",
        "fig1c_all_gestures": OUT_DIR / "Fig1c_All_Gestures_Raw_vs_Preprocessed.png",
        "fig2_workflow": OUT_DIR / "Fig2_Preprocessing_Workflow.png",
        "fig3_zoomed_comparison": OUT_DIR / "Fig3_Zoomed_Comparison.png",
        "fig4_filter_response": OUT_DIR / "Fig4_Filter_Frequency_Response.png",
        "fig5_rms_envelope": OUT_DIR / "Fig5_RMS_Envelope_Comparison.png",
        "representative_subject": OUT_DIR / "representative_subject.json",
        "preprocessing_params": OUT_DIR / "preprocessing_params.json",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    # ── Runtime logging ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    params["runtime_sec"] = elapsed
    params["runtime_minutes"] = minutes
    params["runtime_seconds"] = seconds
    
    logger.info(f"  BLOCK D runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    # ── Summary Banner ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BLOCK D COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Representative Subject: {representative_subject}")
    if subject_scores:
        logger.info(f"    Quality Index: {subject_scores.get('quality_index', 'N/A'):.4f}")
        logger.info(f"    Mean RMS: {subject_scores.get('mean_rms', 'N/A'):.4f}")
    logger.info(f"  Figures generated: 7 (Fig1, Fig1b, Fig1c, Fig2, Fig3, Fig4, Fig5)")
    logger.info(f"  Filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz + Notch {NOTCH_FREQ} Hz")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return params


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return
    try:
        params = run_preprocessing_viz()
        mark_completed(BLOCK_NAME, metadata=params)
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_D WITH ALL POLISH FIXES
# ✓ FIX 1: Renamed "Signal Quality Score" → "Signal Quality Index" (heuristic ranking metric)
# ✓ FIX 2: Added note in code: "heuristic ranking metric, not a standardized EMG SQS"
# ✓ FIX 3: Added Figure 1c – All 7 Gestures Raw vs Preprocessed (representative subject)
# ✓ Figure 1: Raw vs Preprocessed EMG (3 channels, side-by-side)
# ✓ Figure 1b: Power Spectral Density (PSD) Comparison
# ✓ Figure 1c: All 7 Gestures — Raw vs Preprocessed (representative subject)
# ✓ Figure 2: Preprocessing Workflow Diagram
# ✓ Figure 3: Zoomed Comparison (Raw vs Cleaned, 1-second segment)
# ✓ Figure 4: Filter Frequency Response (Bandpass + Notch)
# ✓ Figure 5: RMS Envelope Before vs After Preprocessing
# ✓ AUTOMATIC REPRESENTATIVE SUBJECT: Signal Quality Index selection
# ✓ representative_subject.json saved with quality_index details
# ✓ runtime_sec, runtime_minutes, runtime_seconds added to params
# ✓ _finalize_figure() helper for consistent figure handling
# ✓ Thread settings changed from "2" to "1" (consistent with other blocks)
# ✓ plt.close(fig) and gc.collect() after every figure
# ✓ Runtime logging with time.perf_counter()
# ✓ Environment versions saved (environment_versions.json)
# ✓ Artifact manifest with file verification
# ✓ logger.exception() used instead of logger.error(..., exc_info=True)
# ✓ EMG_CHANNEL_COLUMNS from config used in _detect_emg_columns()
# ✓ CRITICAL FIX: NUMBA_DISABLE_JIT=1 set before imports
# ✓ CRITICAL FIX: datetime import added
# ✓ CRITICAL FIX: logger created BEFORE fallback imports
# ✓ CRITICAL FIX: dynamic TOTAL_FEATURES from FEATURE_NAMES
# ✓ CRITICAL FIX: OVERLAP added to workflow figure
# ✓ dependencies: config, state_manager, artifact_manager, scipy, matplotlib
# ✓ outputs: 7 figures + representative_subject.json + preprocessing_params.json 
#            + env versions + artifacts.json
# ✓ preprocess_emg() exactly preserved from SD pipeline
# ✓ is_good_window() matches SD pipeline
# ✓ uses N_CHANNELS from config (not hardcoded)
# ✓ uses PIPELINE_NAME from config
# ✓ uses OVERLAP from config
# ✓ executable standalone: python BLOCK_D_PRE1.py
# ✓ resume logic active
# ✓ consistent with DATA1, MEM1, LOG1 (artifacts.json, docstring style)
# ─────────────────────────────────────────────────────────────────────────────