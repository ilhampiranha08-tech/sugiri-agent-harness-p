"""
Example: Git checkpoint extension.

Auto-commits before each agent action.
Mirrors pi's git-checkpoint.ts example.
"""

import subprocess
from pathlib import Path

from src.core.types import ExtensionAPI


def default(api: ExtensionAPI):
    """Auto-commit checkpoint on each agent turn."""
    
    def on_turn_start(event, ctx):
        cwd = Path.cwd()
        
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            return  # Not a git repo
        
        # Check if there are changes
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        
        if not status.stdout.strip():
            return  # No changes
        
        # Auto-commit
        msg = f"checkpoint: agent turn {event.get('turn', 'unknown')}"
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", msg, "--allow-empty"],
            capture_output=True,
        )
        
        print(f"\n[Git Checkpoint] Auto-committed: {msg}")
    
    api.on("turn_start", on_turn_start)
    print("[Git Checkpoint] Loaded!")
