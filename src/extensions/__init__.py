"""
Extension system for Sugiri.

Extensions are Python modules that can:
- Register custom tools
- Subscribe to lifecycle events
- Register slash commands
- Access session state
- Interact with users via UI

Mirrors pi's extension architecture.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.types import (
    AgentEvent,
    AgentTool,
    EventBus,
    EventType,
    ExtensionAPI,
    ToolCallResult,
)


class ExtensionRuntime:
    """Manages extension loading, lifecycle, and the API they receive."""
    
    def __init__(
        self,
        event_bus: EventBus,
        tools_registry: Dict[str, AgentTool],
        sessions: Optional[Any] = None,
    ):
        self._event_bus = event_bus
        self._tools_registry = tools_registry
        self._sessions = sessions
        self._extensions: List[Any] = []
        self._commands: Dict[str, Callable] = {}
        self._shortcuts: Dict[str, Callable] = {}
        self._loaded_paths: List[str] = []
        
        # Custom tools registered by extensions
        self._custom_tools: Dict[str, AgentTool] = {}
    
    def create_api(self) -> ExtensionAPI:
        """Create the ExtensionAPI object passed to extension factories."""
        return ExtensionAPI(
            register_tool=self.register_tool,
            register_command=self.register_command,
            register_shortcut=self.register_shortcut,
            on=self._event_bus.on,
            emit=self._event_bus.emit,
            append_entry=(lambda *a, **kw: None),  # Set later
            session=None,
            ui=None,
        )
    
    def register_tool(self, tool: AgentTool) -> None:
        """Register a custom tool from an extension."""
        self._custom_tools[tool.name] = tool
        self._tools_registry[tool.name] = tool
    
    def register_command(self, name: str, handler: Callable, description: str = "") -> None:
        """Register a slash command from an extension."""
        self._commands[name] = {"handler": handler, "description": description}
    
    def register_shortcut(self, key: str, handler: Callable) -> None:
        """Register a keyboard shortcut from an extension."""
        self._shortcuts[key] = handler
    
    def get_custom_tools(self) -> Dict[str, AgentTool]:
        """Get all tools registered by extensions."""
        return self._custom_tools
    
    def get_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered commands."""
        return self._commands
    
    def load_extension_from_path(self, path: str) -> Optional[Any]:
        """Load an extension from a file path."""
        ext_path = Path(path).expanduser().resolve()
        
        if not ext_path.exists():
            return None
        
        if str(ext_path) in self._loaded_paths:
            return None
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"extension_{len(self._extensions)}",
                str(ext_path),
            )
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Call the default export (factory function)
            factory = getattr(module, "default", None)
            if factory is None:
                factory = getattr(module, "create_extension", None)
            
            if factory is None:
                return None
            
            api = self.create_api()
            
            # Support both sync and async factories
            import asyncio
            if asyncio.iscoroutinefunction(factory):
                # Async factory - schedule but may need special handling
                result = factory(api)
                self._extensions.append({"module": module, "api": api, "pending": result})
            else:
                result = factory(api)
                self._extensions.append({"module": module, "api": api, "result": result})
            
            self._loaded_paths.append(str(ext_path))
            return module
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def discover_and_load(self, directories: List[str]) -> List[str]:
        """Discover and load extensions from directories."""
        loaded = []
        
        for directory in directories:
            ext_dir = Path(directory).expanduser().resolve()
            if not ext_dir.exists():
                continue
            
            # Load .py files
            for file_path in sorted(ext_dir.glob("*.py")):
                if file_path.name.startswith("_"):
                    continue
                
                result = self.load_extension_from_path(str(file_path))
                if result is not None:
                    loaded.append(str(file_path))
            
            # Load subdirectories with __init__.py or main.py
            for subdir in sorted(ext_dir.iterdir()):
                if not subdir.is_dir() or subdir.name.startswith("_"):
                    continue
                
                init_file = subdir / "__init__.py"
                main_file = subdir / "main.py"
                
                if init_file.exists():
                    result = self.load_extension_from_path(str(init_file))
                    if result is not None:
                        loaded.append(str(init_file))
                elif main_file.exists():
                    result = self.load_extension_from_path(str(main_file))
                    if result is not None:
                        loaded.append(str(main_file))
        
        return loaded
    
    def get_extensions(self) -> List[Dict[str, Any]]:
        """Get info about loaded extensions."""
        return [
            {
                "path": path,
                "has_commands": bool(ext.get("api")),
                "tool_count": len(self._custom_tools),
            }
            for path, ext in zip(self._loaded_paths, self._extensions)
        ]


# ── Resource Loader ──────────────────────────────────────────────────────────

class ResourceLoader:
    """Discovers and loads: extensions, skills, prompts, themes, context files.
    
    Mirrors pi's DefaultResourceLoader.
    """
    
    def __init__(
        self,
        cwd: str = ".",
        agent_dir: str = "~/.agent",
        settings_manager: Optional[Any] = None,
    ):
        self.cwd = Path(cwd).expanduser().resolve()
        self.agent_dir = Path(agent_dir).expanduser().resolve()
        self._settings_manager = settings_manager
        
        # Discovery directories
        self._global_extensions_dir = self.agent_dir / "extensions"
        self._project_extensions_dir = self.cwd / ".agent" / "extensions"
        
        self._global_skills_dir = self.agent_dir / "skills"
        self._project_skills_dir = self.cwd / ".agent" / "skills"
        
        self._global_prompts_dir = self.agent_dir / "prompts"
        self._project_prompts_dir = self.cwd / ".agent" / "prompts"
        
        self._global_themes_dir = self.agent_dir / "themes"
        self._project_themes_dir = self.cwd / ".agent" / "themes"
        
        # Loaded resources
        self._extensions: List[Any] = []
        self._skills: List[Any] = []
        self._prompts: List[Any] = []
        self._themes: Dict[str, Dict] = {}
        self._context_files: List[Dict[str, str]] = []
        # Cache: mtime + content to avoid re-reading unchanged files
        self._context_cache: Dict[str, tuple] = {}  # path -> (mtime, content)
    
    def reload(self) -> None:
        """Reload all resources."""
        # Purge stale cache entries (files that no longer exist)
        stale = [k for k in self._context_cache if not Path(k).exists()]
        for k in stale:
            del self._context_cache[k]
        self._load_context_files()
        self._load_skills()
        self._load_prompts()
        self._load_themes()
    
    def _read_cached(self, file_path: Path, prefix: str = "") -> str:
        """Read file with mtime caching. Returns cached content if unchanged."""
        key = str(file_path)
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            return prefix
        if key in self._context_cache:
            cached_mtime, cached_content = self._context_cache[key]
            if cached_mtime == mtime:
                return cached_content
        content = prefix + file_path.read_text()
        self._context_cache[key] = (mtime, content)
        return content
    
    def _load_context_files(self) -> None:
        """Load AGENTS.md / CLAUDE.md context files and .agent/memory.md.
        Uses mtime caching to avoid re-reading unchanged files."""
        self._context_files = []
        
        # Global
        global_file = self.agent_dir / "AGENTS.md"
        if global_file.exists():
            self._context_files.append({
                "path": str(global_file),
                "content": self._read_cached(global_file),
            })
        
        # Walk up from cwd to find AGENTS.md files
        current = self.cwd
        while True:
            agents_file = current / "AGENTS.md"
            claude_file = current / "CLAUDE.md"
            
            for f in [agents_file, claude_file]:
                if f.exists() and f not in [Path(c["path"]) for c in self._context_files]:
                    self._context_files.append({
                        "path": str(f),
                        "content": self._read_cached(f),
                    })
            
            # Also check .agent/memory.md for session memory
            memory_file = current / ".agent" / "memory.md"
            if memory_file.exists():
                self._context_files.append({
                    "path": str(memory_file),
                    "content": self._read_cached(memory_file, prefix="## Session Memory\n\n"),
                })
            
            parent = current.parent
            if parent == current:
                break
            current = parent
    
    def get_context_files(self) -> List[Dict[str, str]]:
        return self._context_files
    
    def _load_skills(self) -> None:
        """Load skills from skill directories."""
        self._skills = []
        
        for skills_dir in [self._global_skills_dir, self._project_skills_dir]:
            if not skills_dir.exists():
                continue
            
            # Direct .md files as individual skills
            for md_file in sorted(skills_dir.glob("*.md")):
                self._load_skill_file(md_file)
            
            # Directories containing SKILL.md
            for subdir in sorted(skills_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                skill_file = subdir / "SKILL.md"
                if skill_file.exists():
                    self._load_skill_file(skill_file, base_dir=str(subdir))
    
    def _load_skill_file(self, path: Path, base_dir: Optional[str] = None) -> None:
        """Load a single skill from a markdown file."""
        try:
            content = path.read_text()
            
            # Parse frontmatter
            name = path.stem
            description = ""
            
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = {}
                    for line in parts[1].strip().split("\n"):
                        if ":" in line:
                            key, _, value = line.partition(":")
                            frontmatter[key.strip()] = value.strip()
                    
                    name = frontmatter.get("name", name)
                    description = frontmatter.get("description", "")
            
            if not description:
                # Use first heading as description
                for line in content.split("\n"):
                    if line.startswith("# "):
                        description = line[2:].strip()
                        break
            
            from core.types import Skill
            self._skills.append(Skill(
                name=name,
                description=description[:1024],
                file_path=str(path),
                base_dir=base_dir or str(path.parent),
                source="local",
                content=content,
            ))
        except Exception:
            pass
    
    def get_skills(self) -> List[Any]:
        return self._skills
    
    def _load_prompts(self) -> None:
        """Load prompt templates."""
        self._prompts = []
        
        for prompts_dir in [self._global_prompts_dir, self._project_prompts_dir]:
            if not prompts_dir.exists():
                continue
            
            for md_file in sorted(prompts_dir.glob("*.md")):
                try:
                    content = md_file.read_text()
                    name = md_file.stem
                    description = ""
                    
                    if content.startswith("<!--"):
                        # Extract description from HTML comment
                        end = content.find("-->")
                        if end > 0:
                            description = content[4:end].strip()
                    
                    if not description:
                        for line in content.split("\n"):
                            if line.startswith("# "):
                                description = line[2:].strip()
                                break
                    
                    from core.types import PromptTemplate
                    self._prompts.append(PromptTemplate(
                        name=name,
                        description=description,
                        source=str(md_file),
                        content=content,
                    ))
                except Exception:
                    pass
    
    def get_prompts(self) -> List[Any]:
        return self._prompts
    
    def _load_themes(self) -> None:
        """Load theme files."""
        self._themes = {"dark": {}, "light": {}}
        
        for themes_dir in [self._global_themes_dir, self._project_themes_dir]:
            if not themes_dir.exists():
                continue
            
            for theme_file in sorted(themes_dir.glob("*.json")):
                try:
                    import json
                    theme_data = json.loads(theme_file.read_text())
                    theme_name = theme_file.stem
                    self._themes[theme_name] = theme_data
                except Exception:
                    pass
    
    def get_themes(self) -> Dict[str, Dict]:
        return self._themes
    
    def get_extension_directories(self) -> List[str]:
        """Get directories to scan for extensions."""
        dirs = []
        if self._global_extensions_dir.exists():
            dirs.append(str(self._global_extensions_dir))
        if self._project_extensions_dir.exists():
            dirs.append(str(self._project_extensions_dir))
        return dirs
    
    def get_skill_directories(self) -> List[str]:
        """Get directories to scan for skills."""
        dirs = []
        if self._global_skills_dir.exists():
            dirs.append(str(self._global_skills_dir))
        if self._project_skills_dir.exists():
            dirs.append(str(self._project_skills_dir))
        return dirs
