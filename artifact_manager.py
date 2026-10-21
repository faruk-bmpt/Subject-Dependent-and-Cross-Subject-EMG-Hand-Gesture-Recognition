# artifact_manager.py
# ===================
# Manages artifact loading and saving across all frozen blocks.
#
# VERSION: 1.0.0-90FEATURE
# FROZEN: Aligned with EMG_5ML_PIPELINE_40
#
# All save functions use UTF-8 encoding for text/JSON files.
# All load functions handle errors gracefully when possible.
# Atomic file writes prevent partial artifacts on crash.
#

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── FIX 1: Broader exception handling with FEATURE_COUNT ──────────────────
try:
    from config import (
        OUTPUT_ROOT, BLOCK_DIRS, ARTIFACTS, FEATURE_NAMES,
        FEATURE_COUNT, PIPELINE_VERSION, FEATURE_LIBRARY_VERSION,
        PIPELINE_NAME
    )
except Exception:
    # Fallback values if config not available or has errors
    OUTPUT_ROOT = Path("./outputs")
    BLOCK_DIRS = {
        "BLOCK_2": OUTPUT_ROOT / "BLOCK2_FEATURES",
        "LOGS": OUTPUT_ROOT / "logs",
    }
    ARTIFACTS = {
        "X_features_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "X_features.npy"),
        "y_labels_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "y_labels.npy"),
        "subjects_npy": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "subjects.npy"),
        "feature_names_json": str(OUTPUT_ROOT / "BLOCK2_FEATURES" / "feature_names.json"),
    }
    FEATURE_COUNT = 90
    FEATURE_NAMES = [f"F{i:03d}" for i in range(FEATURE_COUNT)]
    PIPELINE_VERSION = "1.0.0-90FEATURE"
    FEATURE_LIBRARY_VERSION = "90_FEATURE_FINAL"
    PIPELINE_NAME = "EMG_5ML_PIPELINE_40"


#
# SETUP DIRECTORIES
#

def setup_directories() -> None:
    """Create all necessary output directories from BLOCK_DIRS."""
    for d in BLOCK_DIRS.values():
        Path(d).mkdir(parents=True, exist_ok=True)


#
# ATOMIC FILE WRITE HELPERS
#

def _json_serializer(obj: Any) -> Any:
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _atomic_write_text(content: str, path: Path, encoding: str = "utf-8") -> None:
    """
    Write text content atomically using a temporary file.
    
    Parameters
    ----------
    content : str
        Text content to write
    path : Path
        Target file path
    encoding : str
        Encoding for text files
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
        # Atomic replace
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_json(data: Dict, path: Path) -> None:
    """
    Write JSON data atomically using a temporary file.
    
    Parameters
    ----------
    data : dict
        Data to write
    path : Path
        Target file path
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_serializer)
        # Atomic replace
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_binary(content: bytes, path: Path) -> None:
    """
    Write binary content atomically using a temporary file.
    
    Parameters
    ----------
    content : bytes
        Binary content to write
    path : Path
        Target file path
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        # Atomic replace
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


#
# SAVE FUNCTIONS (with atomic writes)
#

def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a DataFrame to CSV with UTF-8 encoding (atomic).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    # Atomic replace
    tmp_path.replace(path)


def save_json(data: Dict, path: Union[str, Path]) -> None:
    """
    Save data to JSON with UTF-8 encoding (atomic).
    
    Parameters
    ----------
    data : dict
        Data to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(data, path)


def save_txt(text: str, path: Union[str, Path]) -> None:
    """
    Save text to a file with UTF-8 encoding (atomic).
    
    Parameters
    ----------
    text : str
        Text content to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(text, path)


def save_fig(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 600,
) -> None:
    """
    Save a matplotlib figure using publication-quality settings.

    Features
    --------
    • 600 DPI (journal quality)
    • Atomic file write (.tmp extension)
    • Tight bounding box
    • White background
    • Small padding to avoid clipping
    • Original file extension preserved
    • Automatically closes figure after saving

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.

    path : str or Path
        Output file path.

    dpi : int, optional
        Figure resolution (default: 600 DPI).
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve extension (e.g. .png, .pdf, .svg)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")

    fig.savefig(
        tmp_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        metadata={
            "Creator": PIPELINE_NAME,
            "Software": PIPELINE_VERSION,
        },
    )

    # Atomic replace
    tmp_path.replace(path)

    # Free memory
    plt.close(fig)


def save_npy(arr: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a numpy array (atomic).
    
    FIX: Proper .tmp.npy extension to avoid .npy.tmp.npy issue.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # FIX: Use .tmp.npy extension to avoid double .npy
    tmp_path = path.with_suffix(".tmp.npy")
    np.save(tmp_path, arr)
    # Atomic replace
    tmp_path.replace(path)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save an object using pickle with HIGHEST_PROTOCOL (atomic).
    
    Parameters
    ----------
    obj : Any
        Object to save
    path : str or Path
        Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Atomic replace
    tmp_path.replace(path)


#
# LOAD FUNCTIONS
#

def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from CSV with UTF-8 encoding.
    
    Parameters
    ----------
    path : str or Path
        Input file path
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    return pd.read_csv(path, encoding="utf-8")


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load data from JSON with UTF-8 encoding.
    
    Parameters
    ----------
    path : str or Path
        Input file path
    
    Returns
    -------
    dict
        Loaded JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_txt(path: Union[str, Path]) -> str:
    """
    Load text from a file with UTF-8 encoding.
    
    Parameters
    ----------
    path : str or Path
        Input file path
    
    Returns
    -------
    str
        Loaded text content
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_npy(path: Union[str, Path], allow_pickle: bool = False) -> np.ndarray:
    """
    Load a numpy array.
    
    Parameters
    ----------
    path : str or Path
        Input file path
    allow_pickle : bool, optional
        Whether to allow pickle loading (default: False)
    
    Returns
    -------
    np.ndarray
        Loaded array
    """
    return np.load(path, allow_pickle=allow_pickle)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load an object using pickle.
    
    Parameters
    ----------
    path : str or Path
        Input file path
    
    Returns
    -------
    Any
        Loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def artifact_exists(path: Union[str, Path]) -> bool:
    """
    Check if an artifact file exists.
    
    Parameters
    ----------
    path : str or Path
        File path to check
    
    Returns
    -------
    bool
        True if file exists
    """
    return Path(path).exists()


def load_feature_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load feature matrix from BLOCK_2 artifacts.
    
    Uses FEATURE_COUNT from config for validation.
    Safer JSON loading with fallback.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    subjects : np.ndarray
        Subject IDs
    feature_names : List[str]
        Feature names

    Raises
    ------
    FileNotFoundError
        If required artifacts are missing
    AssertionError
        If feature count mismatch is detected
    """
    X_path = Path(ARTIFACTS.get("X_features_npy", "outputs/BLOCK2_FEATURES/X_features.npy"))
    y_path = Path(ARTIFACTS.get("y_labels_npy", "outputs/BLOCK2_FEATURES/y_labels.npy"))
    subjects_path = Path(ARTIFACTS.get("subjects_npy", "outputs/BLOCK2_FEATURES/subjects.npy"))
    feature_names_path = Path(ARTIFACTS.get("feature_names_json", "outputs/BLOCK2_FEATURES/feature_names.json"))

    # Check required files exist
    if not X_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels not found: {y_path}")
    if not subjects_path.exists():
        raise FileNotFoundError(f"Subjects not found: {subjects_path}")

    X = load_npy(X_path)
    y = load_npy(y_path)
    subjects = load_npy(subjects_path, allow_pickle=True).astype(str)

    # Safer JSON loading with fallback
    try:
        feature_names = load_json(feature_names_path)
        # Handle both list and dict formats
        if isinstance(feature_names, dict):
            feature_names = feature_names.get("features", FEATURE_NAMES)
        elif not isinstance(feature_names, list):
            feature_names = FEATURE_NAMES
    except Exception:
        # Fallback to config feature names if JSON is corrupted
        feature_names = FEATURE_NAMES

    # Validate feature count against FEATURE_COUNT
    expected_features = FEATURE_COUNT
    actual_features = X.shape[1]
    
    assert actual_features == expected_features, (
        f"Feature mismatch: X has {actual_features} features, "
        f"but FEATURE_COUNT is {expected_features}"
    )
    
    # Also verify against feature_names length
    assert len(feature_names) == expected_features, (
        f"Feature name mismatch: feature_names has {len(feature_names)}, "
        f"but FEATURE_COUNT is {expected_features}"
    )

    return X, y, subjects, feature_names


def save_artifact_manifest(
    artifact_map: Dict[str, Path],
    out_dir: Path,
    block_name: str,
    pipeline_name: Optional[str] = None,
) -> None:
    """
    Save artifacts.json manifest listing every output produced by a block.
    
    Includes pipeline_version and feature_library_version for traceability.
    
    Parameters
    ----------
    artifact_map : dict
        Mapping of artifact names to their paths
    out_dir : Path
        Output directory
    block_name : str
        Name of the block (e.g., "BLOCK_0_OPT1")
    pipeline_name : str, optional
        Pipeline name (defaults to config value)
    """
    pipeline = pipeline_name or PIPELINE_NAME
    
    manifest = {
        "block": block_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pipeline": pipeline,
        "pipeline_version": PIPELINE_VERSION,
        "feature_library_version": FEATURE_LIBRARY_VERSION,
        "feature_count": FEATURE_COUNT,
        "artifacts": {
            name: str(path) for name, path in artifact_map.items()
        },
    }
    
    manifest_path = out_dir / "artifacts.json"
    save_json(manifest, manifest_path)


def get_block_output_dir(block_name: str) -> Path:
    """
    Get the output directory for a specific block.
    
    Parameters
    ----------
    block_name : str
        Name of the block (e.g., "BLOCK_0_OPT1")
    
    Returns
    -------
    Path
        Output directory path
    """
    # Map block name to BLOCK_DIRS key
    block_map = {
        "BLOCK_0": "BLOCK_0",
        "BLOCK_1": "BLOCK_1",
        "BLOCK_2": "BLOCK_2",
        "BLOCK_3": "BLOCK_3",
        "BLOCK_4": "BLOCK_4",
        "BLOCK_5": "BLOCK_5",
        "BLOCK_6": "BLOCK_6",
        "BLOCK_7": "BLOCK_7",
        "BLOCK_8": "BLOCK_8",
        "BLOCK_9": "BLOCK_9",
        "BLOCK_10": "BLOCK_10",
        "BLOCK_11": "BLOCK_11",
        "BLOCK_12": "BLOCK_12",
        "BLOCK_13": "BLOCK_13",
    }
    
    # Extract base block name (e.g., "BLOCK_0" from "BLOCK_0_OPT1")
    for key, dir_key in block_map.items():
        if block_name.startswith(key):
            try:
                return Path(BLOCK_DIRS[dir_key])
            except Exception:
                return OUTPUT_ROOT / dir_key
    
    # Fallback
    return OUTPUT_ROOT / block_name


def ensure_directory(path: Union[str, Path]) -> None:
    """
    Ensure a directory exists.
    
    Parameters
    ----------
    path : str or Path
        Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def list_artifacts(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List all artifact files in a directory matching a pattern.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str, optional
        Glob pattern (default: "*")
    
    Returns
    -------
    list
        List of matching file paths
    """
    return list(Path(directory).glob(pattern))


def get_artifact_metadata(directory: Union[str, Path]) -> Optional[Dict]:
    """
    Load artifacts.json metadata from a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing artifacts.json
    
    Returns
    -------
    dict or None
        Metadata dictionary if found
    """
    manifest_path = Path(directory) / "artifacts.json"
    if manifest_path.exists():
        try:
            return load_json(manifest_path)
        except Exception:
            return None
    return None


# ── Main: Print summary when run directly ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("artifact_manager.py - Summary")
    print("=" * 60)
    print(f"Output Root: {OUTPUT_ROOT}")
    print(f"Feature Count: {FEATURE_COUNT}")
    print(f"Feature Library: {FEATURE_LIBRARY_VERSION}")
    print(f"Pipeline Version: {PIPELINE_VERSION}")
    print(f"Available Blocks: {list(BLOCK_DIRS.keys())}")
    print("=" * 60)
    print("Functions available:")
    print("  - save_csv(), save_json(), save_txt(), save_fig()")
    print("  - save_npy(), save_pickle()")
    print("  - load_csv(), load_json(), load_txt()")
    print("  - load_npy(), load_pickle()")
    print("  - load_feature_matrix()")
    print("  - save_artifact_manifest()")
    print("  - get_block_output_dir()")
    print("  - setup_directories()")
    print("  - ensure_directory()")
    print("  - list_artifacts()")
    print("  - get_artifact_metadata()")
    print("=" * 60)
    print("Atomic writes: ENABLED (temporary files prevent partial writes)")
    print("Atomic replace: path.replace() used for atomicity")
    print("Pickle protocol: HIGHEST_PROTOCOL for smaller/faster files")
    print("save_fig: .tmp.{extension} preserves original extension")
    print("save_npy: .tmp.npy avoids double extension issue")
    print("=" * 60)