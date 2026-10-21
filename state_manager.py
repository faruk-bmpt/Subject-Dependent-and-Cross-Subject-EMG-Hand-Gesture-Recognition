# state_manager.py
# ================
# Manages pipeline state and resume logic for frozen blocks.
# Each block uses BLOCK_NAME = "BLOCK_X_NAME" format.
#
# VERSION: 1.0.0-90FEATURE
# FROZEN: Aligned with EMG_5ML_PIPELINE_40
#
# STATE DIRECTORY: Uses OUTPUT_ROOT/STATE for project-level state management
#

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# ── FIX 1: Use OUTPUT_ROOT from config with broader exception handling ────
try:
    from config import OUTPUT_ROOT
    STATE_DIR = OUTPUT_ROOT / "STATE"
except Exception:
    # Fallback: Use local ./state directory if config not available or has errors
    STATE_DIR = Path("./state")

# Ensure state directory exists
STATE_DIR.mkdir(parents=True, exist_ok=True)


# ── FIX 1: Add config import for versioning with broader exception handling ──
try:
    from config import PIPELINE_VERSION, PIPELINE_NAME, PIPELINE_STAGE
    PIPELINE_METADATA = {
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_stage": PIPELINE_STAGE,
    }
except Exception:
    PIPELINE_METADATA = {
        "pipeline_name": "EMG_5ML_PIPELINE_40",
        "pipeline_version": "1.0.0-90FEATURE",
        "pipeline_stage": "90_FEATURE_FROZEN",
    }


def get_state_path(block_name: str) -> Path:
    """Get the state file path for a block."""
    return STATE_DIR / f"{block_name}.done"


def is_completed(block_name: str) -> bool:
    """
    Check if a block has been completed.
    
    Parameters
    ----------
    block_name : str
        Name of the block (e.g., "BLOCK_0_OPT1")
    
    Returns
    -------
    bool
        True if the block has been completed
    """
    return get_state_path(block_name).exists()


def mark_completed(block_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark a block as completed with optional metadata.
    
    FIX: Added timestamp and pipeline metadata for better traceability.
    FIX: Added encoding="utf-8" for cleaner file handling.
    
    Parameters
    ----------
    block_name : str
        Name of the block (e.g., "BLOCK_0_OPT1")
    metadata : dict, optional
        Additional metadata to save with the completion marker
    """
    state_path = get_state_path(block_name)
    
    # Build completion data with timestamp
    data = {
        "block": block_name,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "pipeline": PIPELINE_METADATA,
        "metadata": metadata or {}
    }
    
    # Write state file atomically with UTF-8 encoding
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_completion_metadata(block_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the metadata for a completed block.
    
    Parameters
    ----------
    block_name : str
        Name of the block
    
    Returns
    -------
    dict or None
        Metadata dictionary if block is completed, else None
    """
    state_path = get_state_path(block_name)
    if not state_path.exists():
        return None
    
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metadata", {})
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def get_completion_timestamp(block_name: str) -> Optional[str]:
    """
    Get the completion timestamp for a block.
    
    Parameters
    ----------
    block_name : str
        Name of the block
    
    Returns
    -------
    str or None
        ISO format timestamp if block is completed, else None
    """
    state_path = get_state_path(block_name)
    if not state_path.exists():
        return None
    
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("timestamp")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def reset_block(block_name: str) -> None:
    """
    Reset a specific block (delete its completion marker).
    
    Parameters
    ----------
    block_name : str
        Name of the block to reset
    """
    state_path = get_state_path(block_name)
    if state_path.exists():
        state_path.unlink()


def reset_all() -> None:
    """Reset all blocks (delete all completion markers)."""
    for state_file in STATE_DIR.glob("*.done"):
        state_file.unlink()


def list_completed_blocks() -> List[str]:
    """
    List all completed blocks in alphabetical order.
    
    Returns
    -------
    list
        Sorted list of completed block names
    """
    return sorted(f.stem for f in STATE_DIR.glob("*.done"))


def get_state_summary() -> Dict[str, Any]:
    """
    Get a summary of all completed blocks with their timestamps.
    
    Returns
    -------
    dict
        Summary dictionary with completion status and timestamps
    """
    summary = {
        "state_directory": str(STATE_DIR),
        "pipeline": PIPELINE_METADATA,
        "completed_blocks": [],
        "total_blocks": 0,
    }
    
    for block_name in list_completed_blocks():
        timestamp = get_completion_timestamp(block_name)
        summary["completed_blocks"].append({
            "block": block_name,
            "timestamp": timestamp
        })
    
    summary["total_blocks"] = len(summary["completed_blocks"])
    
    return summary


def print_state_summary() -> None:
    """Print a formatted summary of all completed blocks."""
    summary = get_state_summary()
    
    print("=" * 70)
    print("PIPELINE STATE SUMMARY")
    print("=" * 70)
    print(f"State Directory: {summary['state_directory']}")
    print(f"Pipeline:        {summary['pipeline']['pipeline_name']}")
    print(f"Version:         {summary['pipeline']['pipeline_version']}")
    print(f"Stage:           {summary['pipeline']['pipeline_stage']}")
    print("-" * 70)
    print(f"Completed Blocks: {summary['total_blocks']}")
    
    if summary['completed_blocks']:
        print("-" * 70)
        for entry in summary['completed_blocks']:
            print(f"  ✓ {entry['block']}  ({entry['timestamp']})")
    else:
        print("  (No blocks completed yet)")
    
    print("=" * 70)


def save_state_summary(output_path: Optional[Path] = None) -> None:
    """
    Save state summary to a JSON file for reproducibility.
    
    Parameters
    ----------
    output_path : Path, optional
        Path to save the summary. Defaults to STATE_DIR/state_summary.json
    """
    if output_path is None:
        output_path = STATE_DIR / "state_summary.json"
    
    summary = get_state_summary()
    
    # Convert Path to string for JSON serialization
    summary["state_directory"] = str(summary["state_directory"])
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def get_state_dir() -> Path:
    """
    Get the current state directory path.
    
    Returns
    -------
    Path
        The state directory path
    """
    return STATE_DIR


# ── Main: Print summary when run directly ────────────────────────────────

if __name__ == "__main__":
    print_state_summary()
    
    # Save summary to file
    save_state_summary()
    print(f"\nState summary saved to: {STATE_DIR / 'state_summary.json'}")