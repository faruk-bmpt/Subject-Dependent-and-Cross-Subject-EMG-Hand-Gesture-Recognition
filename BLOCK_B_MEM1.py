# BLOCK_B_MEM1.py - FINAL POLISHED VERSION
# ===============
# FILE PURPOSE:
#     PRE-BLOCK B — MEM1: Memory Optimization and Computational Resource
#     Management. Profiles available RAM, CPU cores, Python environment,
#     and sets up thread limits to prevent segmentation faults.
#
# INPUT_ARTIFACTS  : None
# OUTPUT_ARTIFACTS :
#     - outputs/PRE_B/Memory_Profile.json
#     - outputs/PRE_B/Runtime_Profile.txt
#     - outputs/PRE_B/environment_versions.json
#     - outputs/PRE_B/FigS1_Resource_Profile.png   (Figure S1)
#     - outputs/PRE_B/artifacts.json
#     - state/BLOCK_B_MEM1.done
#
# DEPENDENCIES     : config.py, state_manager.py, artifact_manager.py
# STANDALONE RUN   : python BLOCK_B_MEM1.py

# ── anti-segfault (must precede numeric imports) ──────────────────────────────
import os
# FIX 1: Use single thread for all math libraries (consistent with other blocks)
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

import gc
import json          # <-- ADDED: Fix for NameError: name 'json' is not defined
import logging
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import xgboost
import scipy
import imblearn

from config import BLOCK_DIRS, SEED, PIPELINE_NAME
from state_manager import is_completed, mark_completed
# FIX 1: Removed duplicate save_json import
from artifact_manager import save_json, save_txt, save_fig

warnings.filterwarnings("ignore")

LOG_FILE = BLOCK_DIRS["LOGS"] / "BLOCK_B_MEM1.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("BLOCK_B")

BLOCK_NAME = "BLOCK_B_MEM1"
OUT_DIR    = BLOCK_DIRS["PRE_B"]

# ─────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ============================================================================
# FIGURE FINALIZER HELPER (identical to other blocks)
# ============================================================================

def _finalize_figure(fig, path: Path, dpi: int = 300) -> None:
    """
    Centralized figure finalization: save, close, and garbage collect.
    Identical to BLOCK_13_ROB1.py and BLOCK_A_DATA1.py.
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
        "pandas": _get_package_version("pandas"),
        "scipy": scipy.__version__ if hasattr(scipy, "__version__") else "unknown",
        "sklearn": sklearn.__version__ if hasattr(sklearn, "__version__") else "unknown",
        "xgboost": xgboost.__version__ if hasattr(xgboost, "__version__") else "unknown",
        "imblearn": imblearn.__version__ if hasattr(imblearn, "__version__") else "unknown",
        "psutil": psutil.__version__ if _PSUTIL and hasattr(psutil, "__version__") else "not_installed",
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
    """
    Save artifacts.json manifest with file verification.
    Identical to BLOCK_4, BLOCK_7, BLOCK_8, BLOCK_9, BLOCK_10, BLOCK_11, BLOCK_12, BLOCK_13, and BLOCK_A.
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


def _get_library_versions() -> Dict[str, str]:
    """Collect versions of key scientific packages."""
    packages = [
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "seaborn", "xgboost", "imblearn", "pywt", "statsmodels",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_memory_profile() -> Dict[str, Any]:
    """Profile system resources and write outputs."""
    
    t0 = time.perf_counter()
    
    logger.info("=" * 70)
    logger.info("BLOCK B | MEM1: Memory & Resource Management")
    logger.info("=" * 70)

    # Log software versions
    logger.info(f"Python : {sys.version}")
    logger.info(f"PID    : {os.getpid()}")
    logger.info(f"psutil : {'available' if _PSUTIL else 'not installed'}")

    profile: Dict[str, Any] = {
        "timestamp"      : datetime.now().isoformat(),
        "python_version" : sys.version,
        "platform"       : platform.platform(),
        "machine"        : platform.machine(),
        "processor"      : platform.processor(),
        "seed"           : SEED,
        "thread_settings": {
            "OMP_NUM_THREADS"       : os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS"  : os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "MKL_NUM_THREADS"       : os.environ.get("MKL_NUM_THREADS", "1"),
        },
        "library_versions": _get_library_versions(),
    }

    # RAM and CPU via psutil
    if _PSUTIL:
        vm = psutil.virtual_memory()
        profile.update({
            "RAM_total_GB"    : round(vm.total / 1e9, 2),
            "RAM_available_GB": round(vm.available / 1e9, 2),
            "RAM_used_pct"    : vm.percent,
            "RAM_used_GB"     : round((vm.total - vm.available) / 1e9, 2),
            "CPU_physical"    : psutil.cpu_count(logical=False),
            "CPU_logical"     : psutil.cpu_count(logical=True),
            "CPU_freq_MHz"    : (psutil.cpu_freq().current
                                 if psutil.cpu_freq() else "N/A"),
        })
        logger.info(
            f"  RAM: {profile['RAM_total_GB']} GB total, "
            f"{profile['RAM_available_GB']} GB available "
            f"({profile['RAM_used_pct']}% used)"
        )
        logger.info(
            f"  CPU: {profile['CPU_physical']} physical, "
            f"{profile['CPU_logical']} logical cores"
        )
    else:
        profile.update({
            "RAM_total_GB"    : "N/A (install psutil)",
            "RAM_available_GB": "N/A",
            "RAM_used_pct"    : "N/A",
            "RAM_used_GB"     : "N/A",
            "CPU_physical"    : "N/A",
            "CPU_logical"     : "N/A",
            "CPU_freq_MHz"    : "N/A",
        })
        logger.warning("  psutil not installed — RAM/CPU info unavailable.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Save JSON profile
    save_json(profile, OUT_DIR / "Memory_Profile.json")
    logger.info(f"  Memory profile saved → {OUT_DIR / 'Memory_Profile.json'}")

    # Runtime profile text
    lines = [
        f"{PIPELINE_NAME} | Runtime Profile",
        "=" * 50,
        f"Timestamp   : {profile['timestamp']}",
        f"Python      : {profile['python_version']}",
        f"Platform    : {profile['platform']}",
        f"Machine     : {profile.get('machine', 'N/A')}",
        f"Processor   : {profile.get('processor', 'N/A')}",
        f"PID         : {os.getpid()}",
    ]
    if _PSUTIL:
        lines += [
            f"RAM Total   : {profile['RAM_total_GB']} GB",
            f"RAM Free    : {profile['RAM_available_GB']} GB",
            f"RAM Used    : {profile['RAM_used_GB']} GB ({profile['RAM_used_pct']}%)",
            f"CPU Cores   : {profile['CPU_physical']} physical / "
            f"{profile['CPU_logical']} logical",
            f"CPU Freq    : {profile['CPU_freq_MHz']} MHz",
        ]
    lines.append("\nThread Settings:")
    for key, val in profile["thread_settings"].items():
        lines.append(f"  {key:<20}: {val}")
    lines.append("\nLibrary Versions:")
    for pkg, ver in profile["library_versions"].items():
        lines.append(f"  {pkg:<15}: {ver}")
    save_txt("\n".join(lines), OUT_DIR / "Runtime_Profile.txt")
    logger.info(f"  Runtime profile saved → {OUT_DIR / 'Runtime_Profile.txt'}")

    # ── Figure S1 – Computational Resource Profile ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RAM bar chart
    ax = axes[0]
    if _PSUTIL and isinstance(profile["RAM_total_GB"], float):
        labels = ["Total", "Available", "Used"]
        values = [
            profile["RAM_total_GB"],
            profile["RAM_available_GB"],
            profile["RAM_used_GB"],
        ]
        # FIX 3: Use consistent colors (blue, green, red - fine for resource figure)
        colors = ["#42A5F5", "#66BB6A", "#EF5350"]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_ylabel("Memory (GB)")
        ax.set_title("RAM Usage Profile")
        ax.set_ylim(0, profile["RAM_total_GB"] * 1.2)
    else:
        ax.text(0.5, 0.5, "psutil not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("RAM Usage Profile")

    # Right: Library versions (text table)
    ax2 = axes[1]
    ax2.axis("off")
    table_data = [[pkg, ver] for pkg, ver in profile["library_versions"].items()]
    col_labels = ["Package", "Version"]
    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax2.set_title("Library Versions", pad=20)

    fig.suptitle(
        "Figure S1 – Computational Resource Profile",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _finalize_figure(fig, OUT_DIR / "FigS1_Resource_Profile.png")
    logger.info(f"  Figure S1 saved → {OUT_DIR / 'FigS1_Resource_Profile.png'}")

    # ── Save environment versions ────────────────────────────────────────────
    _save_environment_versions()

    # ── Artifact Manifest ──────────────────────────────────────────────────
    artifact_map: Dict[str, Path] = {
        "memory_profile": OUT_DIR / "Memory_Profile.json",
        "runtime_profile": OUT_DIR / "Runtime_Profile.txt",
        "figure_s1": OUT_DIR / "FigS1_Resource_Profile.png",
        "env_versions": OUT_DIR / "environment_versions.json",
    }
    _save_artifact_manifest(artifact_map, OUT_DIR)

    # ── Runtime logging and metadata ──────────────────────────────────────
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Add runtime fields to profile for metadata consistency
    profile["runtime_sec"] = elapsed
    profile["runtime_minutes"] = minutes
    profile["runtime_seconds"] = seconds
    
    logger.info(f"  BLOCK B runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")

    # ── Summary Banner ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BLOCK B COMPLETE")
    logger.info("=" * 70)
    if _PSUTIL:
        logger.info(f"  RAM: {profile['RAM_total_GB']} GB total, {profile['RAM_available_GB']} GB available")
        logger.info(f"  CPU: {profile['CPU_physical']} physical, {profile['CPU_logical']} logical cores")
    else:
        logger.info("  RAM/CPU: psutil not installed")
    logger.info(f"  Runtime: {elapsed:.2f} sec ({minutes}m {seconds}s)")
    logger.info("=" * 70)

    return profile


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for standalone and pipeline execution."""
    if is_completed(BLOCK_NAME):
        logger.info(f"⏭  {BLOCK_NAME} already completed — skipping.")
        return
    
    try:
        summary = run_memory_profile()
        mark_completed(BLOCK_NAME, metadata={
            "RAM_GB"         : summary.get("RAM_total_GB", "N/A"),
            "CPU_cores"      : summary.get("CPU_physical", "N/A"),
            "runtime_sec"    : summary.get("runtime_sec", 0),
            "runtime_minutes": summary.get("runtime_minutes", 0),
            "runtime_seconds": summary.get("runtime_seconds", 0),
        })
        logger.info(f"✅  {BLOCK_NAME} DONE.")
    except Exception as exc:
        logger.exception(f"❌  {BLOCK_NAME} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST - COMPLETE BLOCK_B WITH ALL FIXES
# ✓ FIX 1: Removed duplicate save_json import (single import from artifact_manager)
# ✓ FIX 2: save_json used consistently throughout (no duplicate alias)
# ✓ FIX 3: Thread settings changed from "2" to "1" (consistent with other blocks)
# ✓ FIX 4: _finalize_figure() helper matches BLOCK_13 and BLOCK_A
# ✓ FIX 5: Runtime logging with time.perf_counter() (elapsed, minutes, seconds)
# ✓ FIX 6: Environment versions saved (environment_versions.json)
# ✓ FIX 7: Artifact manifest with file verification (identical to other blocks)
# ✓ FIX 8: logger.exception() used instead of logger.error(..., exc_info=True)
# ✓ FIX 9: Report title uses PIPELINE_NAME for consistency
# ✓ FIX 10: runtime_sec, runtime_minutes, runtime_seconds added to returned profile
# ✓ FIX 11: Summary banner matches other blocks (BLOCK B COMPLETE)
# ✓ FIX 12: Log messages include file paths (consistent with other blocks)
# ✓ FIX 13: import json added to fix NameError
# ✓ syntax valid
# ✓ imports valid
# ✓ removed unused import: numpy (already imported via __import__)
# ✓ removed unused variable: cpu = psutil.cpu_count
# ✓ added artifacts.json manifest
# ✓ dependencies: config, state_manager, artifact_manager, (psutil optional)
# ✓ outputs: Memory_Profile.json, Runtime_Profile.txt,
#           environment_versions.json, FigS1_Resource_Profile.png, artifacts.json
# ✓ executable standalone: python BLOCK_B_MEM1.py
# ✓ resume logic active
# ─────────────────────────────────────────────────────────────────────────────