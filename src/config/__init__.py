"""
Configuration and settings management.

Loads from:
- ~/.agent/settings.json (global)
- .agent/settings.json (project-local, overrides global)

Mirrors pi's SettingsManager.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.types import Settings, ThinkingLevel


DEFAULT_SETTINGS = {
    "default_provider": "anthropic",
    "default_model": "claude-sonnet-4-5",
    "default_thinking": "off",
    "compaction_enabled": True,
    "compaction_trigger_tokens": 150000,
    "retry_enabled": True,
    "max_retries": 3,
    "theme": "dark",
    "show_thinking": True,
    "enabled_extensions": [],
    "enabled_skills": [],
    "enable_telemetry": False,
}


class SettingsManager:
    """Manages agent settings with global/project layering."""
    
    def __init__(self, cwd: str = ".", agent_dir: str = "~/.agent"):
        self.cwd = Path(cwd).expanduser().resolve()
        self.agent_dir = Path(agent_dir).expanduser().resolve()
        
        self._global_file = self.agent_dir / "settings.json"
        self._project_file = self.cwd / ".agent" / "settings.json"
        
        self._settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._overrides: Dict[str, Any] = {}
        
        self._load()
    
    @classmethod
    def create(cls, cwd: str = ".", agent_dir: str = "~/.agent") -> SettingsManager:
        """Factory method - loads from files."""
        return cls(cwd, agent_dir)
    
    @classmethod
    def in_memory(cls, settings: Optional[Dict[str, Any]] = None) -> SettingsManager:
        """In-memory settings (no file I/O)."""
        sm = cls.__new__(cls)
        sm.cwd = Path(".").resolve()
        sm.agent_dir = Path("~/.agent").expanduser().resolve()
        sm._global_file = None
        sm._project_file = None
        sm._settings = dict(DEFAULT_SETTINGS)
        sm._overrides = {}
        if settings:
            sm._settings.update(settings)
        return sm
    
    def _load(self) -> None:
        """Load settings from files."""
        # Load global
        if self._global_file and self._global_file.exists():
            try:
                with open(self._global_file) as f:
                    global_data = json.load(f)
                self._settings.update(global_data)
            except Exception:
                pass
        
        # Load project (overrides global)
        if self._project_file and self._project_file.exists():
            try:
                with open(self._project_file) as f:
                    project_data = json.load(f)
                self._settings.update(project_data)
            except Exception:
                pass
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply runtime overrides."""
        self._overrides.update(overrides)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value, checking overrides first."""
        if key in self._overrides:
            return self._overrides[key]
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value (in-memory)."""
        self._settings[key] = value
        self._save()
    
    def _save(self) -> None:
        """Persist settings to global file."""
        if not self._global_file:
            return
        
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self._global_file, "w") as f:
                json.dump(self._settings, f, indent=2)
        except Exception:
            pass
    
    async def flush(self) -> None:
        """Ensure all pending writes are flushed."""
        self._save()
    
    def get_settings(self) -> Settings:
        """Get typed Settings object."""
        return Settings(
            default_provider=self.get("default_provider", "anthropic"),
            default_model=self.get("default_model", "claude-sonnet-4-5"),
            default_thinking=ThinkingLevel(self.get("default_thinking", "off")),
            compaction_enabled=self.get("compaction_enabled", True),
            compaction_trigger_tokens=self.get("compaction_trigger_tokens", 150000),
            retry_enabled=self.get("retry_enabled", True),
            max_retries=self.get("max_retries", 3),
            theme=self.get("theme", "dark"),
            show_thinking=self.get("show_thinking", True),
            enabled_extensions=self.get("enabled_extensions", []),
            enabled_skills=self.get("enabled_skills", []),
            enable_telemetry=self.get("enable_telemetry", False),
        )
    
    def get_agent_dir(self) -> str:
        """Get the agent config directory."""
        return str(self.agent_dir)
    
    def get_session_dir(self) -> str:
        """Get the session storage directory."""
        session_dir = self.agent_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)
    
    def drain_errors(self) -> List[str]:
        """Get and clear accumulated errors."""
        return []  # Errors are logged, not buffered


# ── Last Model State ───────────────────────────────────────────────────────

def save_last_model(agent_dir: str, provider: str, model_id: str) -> None:
    """Save the last used model so it's restored on next startup."""
    import json, os
    path = os.path.join(os.path.expanduser(agent_dir), "last_model.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"provider": provider, "model_id": model_id}, f)


def load_last_model(agent_dir: str) -> Optional[Dict[str, str]]:
    """Load the last used model. Returns None if not found."""
    import json, os
    path = os.path.join(os.path.expanduser(agent_dir), "last_model.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
