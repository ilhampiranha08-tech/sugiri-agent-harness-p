"""
Package management for Sugiri.

Supports installing, removing, listing, and updating packages.
Packages can be:
- Extensions (Python modules)
- Skills (Markdown-based skill packs)
- Prompts (Prompt templates)
- Themes (JSON theme files)

Sources:
- Local path (copy to agent dir)
- Git repositories (clone & install)
- npm packages (download & extract)

Mirrors pi's package management system.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class PackageManager:
    """Manages package installation, removal, listing, and updates."""

    def __init__(self, agent_dir: str = "~/.agent", cwd: str = "."):
        self.agent_dir = Path(agent_dir).expanduser().resolve()
        self.cwd = Path(cwd).resolve()
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Registry file
        self._registry_file = self.agent_dir / "packages.json"

        # Subdirectories for package types
        self._extensions_dir = self.agent_dir / "extensions"
        self._skills_dir = self.agent_dir / "skills"
        self._prompts_dir = self.agent_dir / "prompts"
        self._themes_dir = self.agent_dir / "themes"

        for d in [self._extensions_dir, self._skills_dir,
                   self._prompts_dir, self._themes_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._project_dir = self.cwd / ".agent"

        self._registry = self._load_registry()

    # ── Public API ─────────────────────────────────────────────────────────

    def install(self, source: str, local: bool = False) -> bool:
        """Install a package from source.

        Args:
            source: URL, git repo, npm package, or local path
            local: If True, install to project .agent/ instead of ~/.agent/

        Returns:
            True if installed successfully
        """
        target_base = self._project_dir if local else self.agent_dir

        # 1. Local path
        if os.path.exists(os.path.expanduser(source)):
            return self._install_from_path(source, target_base)

        # 2. Git repository
        if source.startswith("git+") or source.startswith("https://github.com") or source.startswith("git@"):
            return self._install_from_git(source, target_base)

        # 3. npm package
        if source.startswith("npm:") or source.startswith("@"):
            return self._install_from_npm(source, target_base)

        print(f"Error: Cannot determine source type for: {source}")
        print("  Use a local path, git+https://..., or npm:package-name")
        return False

    def remove(self, name: str) -> bool:
        """Remove an installed package by name."""
        removed = False

        # Remove from registry
        if name in self._registry:
            entry = self._registry.pop(name)
            self._save_registry()

            # Remove files
            for fpath in entry.get("files", []):
                p = Path(fpath).expanduser()
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(str(p))
                    else:
                        p.unlink()
                    print(f"  Removed: {fpath}")
                    removed = True

        if not removed:
            print(f"Package '{name}' not found in registry.")

        return removed

    def list(self) -> List[Dict[str, Any]]:
        """List all installed packages."""
        result = []
        for name, entry in self._registry.items():
            result.append({
                "name": name,
                "version": entry.get("version", "unknown"),
                "source": entry.get("source", "unknown"),
                "type": entry.get("type", "unknown"),
                "location": entry.get("location", "global"),
                "installed": entry.get("installed", ""),
            })
        return result

    def update(self, name: Optional[str] = None, update_all: bool = False) -> bool:
        """Update a specific package or all packages.

        Args:
            name: Package name to update (or None for all)
            update_all: If True, update all packages

        Returns:
            True if any updates were performed
        """
        if name:
            return self._update_one(name)

        if update_all:
            updated_any = False
            for pkg_name in list(self._registry.keys()):
                if self._update_one(pkg_name):
                    updated_any = True
            return updated_any

        print("Specify a package name or --all to update all packages.")
        return False

    def detect_and_install(self, path: str, local: bool = False) -> List[str]:
        """Detect package type from a local path and install.

        Looks for extensions/ skills/ prompts/ themes/ subdirectories
        and installs each found package.

        Returns list of installed package names.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            print(f"Error: Path not found: {path}")
            return []

        installed = []

        # Single file: extension (.py) or skill (.md) or prompt (.md) or theme (.json)
        if p.is_file():
            name = p.stem
            if p.suffix == ".py":
                installed.append(self._install_extension(p, local, name))
            elif p.suffix == ".md":
                # Try as skill first, then prompt
                installed.append(self._install_skill(p, local, name))
            elif p.suffix == ".json":
                installed.append(self._install_theme(p, local, name))
            return [x for x in installed if x]

        # Directory with pi package structure
        for sub in ["extensions", "skills", "prompts", "themes"]:
            subdir = p / sub
            if subdir.exists() and subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_file() and not item.name.startswith("_"):
                        n = item.stem
                        if sub == "extensions" and item.suffix == ".py":
                            installed.append(self._install_extension(item, local, n))
                        elif sub == "skills" and item.suffix == ".md":
                            installed.append(self._install_skill(item, local, n))
                        elif sub == "prompts" and item.suffix == ".md":
                            installed.append(self._install_prompt(item, local, n))
                        elif sub == "themes" and item.suffix == ".json":
                            installed.append(self._install_theme(item, local, n))

        return [x for x in installed if x]

    # ── Internal: Install Methods ──────────────────────────────────────────

    def _install_from_path(self, source: str, target_base: Path) -> bool:
        """Install from a local path."""
        source_path = Path(source).expanduser().resolve()
        installed = self.detect_and_install(str(source_path),
                                            local=(target_base == self._project_dir))
        return len(installed) > 0

    def _install_from_git(self, source: str, target_base: Path) -> bool:
        """Install from a git repository."""
        # Normalize git URL
        git_url = source
        if git_url.startswith("git+"):
            git_url = git_url[4:]

        repo_name = git_url.rstrip("/").split("/")[-1].replace(".git", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            print(f"Cloning {git_url}...")

            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", git_url, str(tmp_path)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    print(f"Error cloning: {result.stderr}")
                    return False
            except FileNotFoundError:
                print("Error: git not found. Install git to use this feature.")
                return False
            except subprocess.TimeoutExpired:
                print("Error: Clone timed out.")
                return False
            except Exception as e:
                print(f"Error cloning: {e}")
                return False

            print(f"Cloned. Detecting package structure...")
            installed = self.detect_and_install(str(tmp_path),
                                                local=(target_base == self._project_dir))
            if installed:
                # Record git source
                for name in installed:
                    if name in self._registry:
                        self._registry[name]["source"] = git_url
                        self._registry[name]["source_type"] = "git"
                self._save_registry()

            return len(installed) > 0

    def _install_from_npm(self, source: str, target_base: Path) -> bool:
        """Install from an npm package."""
        # Normalize npm source
        pkg_name = source
        if pkg_name.startswith("npm:"):
            pkg_name = pkg_name[4:]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            print(f"Downloading {pkg_name} from npm...")

            try:
                result = subprocess.run(
                    ["npm", "pack", pkg_name, "--pack-destination", str(tmp_path)],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(tmpdir),
                )
                if result.returncode != 0:
                    print(f"Error: npm pack failed: {result.stderr}")
                    return False
            except FileNotFoundError:
                print("Error: npm not found. Install Node.js to use this feature.")
                return False
            except subprocess.TimeoutExpired:
                print("Error: npm pack timed out.")
                return False
            except Exception as e:
                print(f"Error downloading: {e}")
                return False

            # Find the downloaded .tgz file
            tgz_files = list(tmp_path.glob("*.tgz"))
            if not tgz_files:
                print("Error: No .tgz file found after npm pack.")
                return False

            tgz_file = tgz_files[0]
            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir()

            # Extract
            import tarfile
            try:
                with tarfile.open(str(tgz_file), "r:gz") as tar:
                    tar.extractall(path=str(extract_dir))
            except Exception as e:
                print(f"Error extracting: {e}")
                return False

            # Find package directory inside
            pkg_dir = extract_dir / "package"
            if not pkg_dir.exists():
                dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                if dirs:
                    pkg_dir = dirs[0]

            installed = self.detect_and_install(str(pkg_dir),
                                                local=(target_base == self._project_dir))
            if installed:
                for name in installed:
                    if name in self._registry:
                        self._registry[name]["source"] = f"npm:{pkg_name}"
                        self._registry[name]["source_type"] = "npm"
                self._save_registry()

            return len(installed) > 0

    # ── Internal: Single File Installers ───────────────────────────────────

    def _install_extension(self, src: Path, local: bool, name: str) -> str:
        """Install a single extension (.py file)."""
        target_dir = (self._project_dir / "extensions") if local else self._extensions_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        shutil.copy2(str(src), str(target))
        self._register(name, "extension", str(target),
                       files=[str(target)],
                       location="project" if local else "global")
        print(f"  ✅ Extension '{name}' installed to {target}")
        return name

    def _install_skill(self, src: Path, local: bool, name: str) -> str:
        """Install a single skill (.md file)."""
        target_dir = (self._project_dir / "skills") if local else self._skills_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        shutil.copy2(str(src), str(target))
        self._register(name, "skill", str(target),
                       files=[str(target)],
                       location="project" if local else "global")
        print(f"  ✅ Skill '{name}' installed to {target}")
        return name

    def _install_prompt(self, src: Path, local: bool, name: str) -> str:
        """Install a single prompt (.md file)."""
        target_dir = (self._project_dir / "prompts") if local else self._prompts_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        shutil.copy2(str(src), str(target))
        self._register(name, "prompt", str(target),
                       files=[str(target)],
                       location="project" if local else "global")
        print(f"  ✅ Prompt '{name}' installed to {target}")
        return name

    def _install_theme(self, src: Path, local: bool, name: str) -> str:
        """Install a single theme (.json file)."""
        target_dir = (self._project_dir / "themes") if local else self._themes_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        shutil.copy2(str(src), str(target))
        self._register(name, "theme", str(target),
                       files=[str(target)],
                       location="project" if local else "global")
        print(f"  ✅ Theme '{name}' installed to {target}")
        return name

    # ── Internal: Registry ─────────────────────────────────────────────────

    def _register(self, name: str, pkg_type: str, target: str,
                   files: List[str], location: str = "global",
                   source: str = "", version: str = "0.0.0"):
        """Register a package in the registry."""
        # If package already exists, preserve its source info
        existing = self._registry.get(name, {})

        self._registry[name] = {
            "name": name,
            "type": pkg_type,
            "location": location,
            "files": files,
            "source": source or existing.get("source", target),
            "source_type": existing.get("source_type", "local"),
            "version": version or existing.get("version", "0.0.0"),
            "installed": existing.get("installed", ""),
        }
        self._save_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the package registry from JSON."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_registry(self):
        """Save the package registry to JSON."""
        with open(self._registry_file, "w") as f:
            json.dump(self._registry, f, indent=2)

    def _update_one(self, name: str) -> bool:
        """Update a single package."""
        if name not in self._registry:
            print(f"Package '{name}' not installed.")
            return False

        entry = self._registry[name]
        source = entry.get("source", "")
        source_type = entry.get("source_type", "local")
        location = entry.get("location", "global")
        local = location == "project"

        if source_type == "local":
            # Re-copy from source
            if os.path.exists(source):
                return self.install(source, local=local)
            else:
                print(f"Cannot update '{name}': source not found at {source}")
                return False

        elif source_type == "git":
            print(f"Updating '{name}' from git: {source}")
            # Remove old files
            for f in entry.get("files", []):
                p = Path(f).expanduser()
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(str(p))
                    else:
                        p.unlink()
            return self.install(source, local=local)

        elif source_type == "npm":
            print(f"Updating '{name}' from npm: {source}")
            for f in entry.get("files", []):
                p = Path(f).expanduser()
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(str(p))
                    else:
                        p.unlink()
            return self.install(source, local=local)

        else:
            print(f"Cannot update '{name}': unknown source type '{source_type}'")
            return False


# ── Convenience ────────────────────────────────────────────────────────────

def get_package_manager(agent_dir: str = "~/.agent", cwd: str = ".") -> PackageManager:
    """Get a PackageManager instance."""
    return PackageManager(agent_dir, cwd)
