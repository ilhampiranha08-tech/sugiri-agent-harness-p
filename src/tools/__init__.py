"""
Built-in tool implementations: read, write, edit, bash.

Mirrors pi's core tool set. All tools are async and return ToolCallResult.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.types import AgentTool, ToolCallResult


# ── Read Tool ────────────────────────────────────────────────────────────────

class ReadTool(AgentTool):
    name = "read"
    label = "Read"
    description = """Read the contents of a file. Supports text files and images (jpg, png, gif, webp).
Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB
(whichever is hit first). Use offset/limit for large files."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    }
    
    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd).resolve()
    
    @staticmethod
    async def _aio_read_text(full_path: Path) -> str:
        """Async file read using thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(lambda: full_path.read_text(encoding="utf-8", errors="replace"))
    
    @staticmethod
    async def _aio_read_bytes(full_path: Path) -> bytes:
        return await asyncio.to_thread(lambda: full_path.read_bytes())
    
    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable] = None,
    ) -> ToolCallResult:
        start = time.time()
        path = params["path"]
        offset = params.get("offset", 1)
        limit = params.get("limit")
        
        full_path = self.cwd / path
        full_path = full_path.resolve()
        
        # Security: resolve and validate path is within cwd or absolute
        try:
            full_path.relative_to(self.cwd)
        except ValueError:
            # Absolute paths: check if within project boundaries
            # For now, allow but log (future: strict mode)
            pass
        
        if not full_path.exists():
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name="read",
                params=params,
                content=[{"type": "text", "text": f"Error: File not found: {path}"}],
                is_error=True,
                duration_ms=(time.time() - start) * 1000,
            )
        
        if full_path.is_dir():
            # List directory
            items = sorted(full_path.iterdir())
            lines = []
            for item in items[:200]:
                prefix = "📁 " if item.is_dir() else "📄 "
                lines.append(f"{prefix}{item.name}")
            content = "\n".join(lines)
            if len(items) > 200:
                content += f"\n... and {len(items) - 200} more items"
        else:
            try:
                # Check if it's an image
                suffix = full_path.suffix.lower()
                if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    size_bytes = full_path.stat().st_size
                    # Try to get image dimensions if PIL is available
                    dims = ""
                    try:
                        from PIL import Image
                        with Image.open(full_path) as img:
                            dims = f", {img.size[0]}x{img.size[1]}"
                    except ImportError:
                        pass
                    content = f"[Image: {full_path.name} ({size_bytes} bytes{dims}, {suffix[1:].upper()})]"
                else:
                    lines = (await self._aio_read_text(full_path)).splitlines(keepends=True)
                    
                    total_lines = len(lines)
                    
                    if offset > 1:
                        lines = lines[offset - 1:]
                    if limit:
                        lines = lines[:limit]
                    
                    content = "".join(lines[:2000])  # Max 2000 lines
                    
                    if len(content) > 50000:
                        content = content[:50000] + "\n... (truncated at 50KB)"
                    
                    if len(lines) >= 2000 and not limit:
                        content += "\n... (output truncated to 2000 lines)"
            except UnicodeDecodeError:
                content = f"[Binary file: {full_path.name} ({full_path.stat().st_size} bytes)]"
        
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            params=params,
            content=[{"type": "text", "text": content}],
            details={"path": str(full_path), "size": full_path.stat().st_size},
            duration_ms=(time.time() - start) * 1000,
        )


# ── Write Tool ───────────────────────────────────────────────────────────────

class WriteTool(AgentTool):
    name = "write"
    label = "Write"
    description = """Write content to a file. Creates the file if it doesn't exist, overwrites if it does.
Automatically creates parent directories."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["path", "content"],
    }
    
    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd).resolve()
    
    @staticmethod
    async def _aio_write(full_path: Path, content: str) -> None:
        """Async file write using thread pool."""
        await asyncio.to_thread(lambda: full_path.write_text(content, encoding="utf-8"))
    
    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable] = None,
    ) -> ToolCallResult:
        start = time.time()
        path = params["path"]
        content = params["content"]
        
        full_path = self.cwd / path
        full_path = full_path.resolve()
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        existed = full_path.exists()
        
        await self._aio_write(full_path, content)
        
        lines = content.count("\n") + 1
        
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name="write",
            params=params,
            content=[{
                "type": "text",
                "text": f"{'Updated' if existed else 'Created'} {path} ({lines} lines, {len(content)} bytes)",
            }],
            details={
                "path": str(full_path),
                "size": len(content),
                "lines": lines,
                "existed": existed,
            },
            duration_ms=(time.time() - start) * 1000,
        )


# ── Edit Tool ────────────────────────────────────────────────────────────────

class EditTool(AgentTool):
    name = "edit"
    label = "Edit"
    description = """Edit a single file using exact text replacement. Every edits[].oldText must match
a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines,
merge them into one edit instead of emitting overlapping edits."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string", "description": "Exact text to replace"},
                        "newText": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    
    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd).resolve()
    
    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable] = None,
    ) -> ToolCallResult:
        start = time.time()
        path = params["path"]
        edits = params["edits"]
        
        full_path = self.cwd / path
        full_path = full_path.resolve()
        
        if not full_path.exists():
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name="edit",
                params=params,
                content=[{"type": "text", "text": f"Error: File not found: {path}"}],
                is_error=True,
                duration_ms=(time.time() - start) * 1000,
            )
        
        original = await asyncio.to_thread(full_path.read_text, encoding="utf-8")
        
        modified = original
        for edit in edits:
            old_text = edit["oldText"]
            new_text = edit["newText"]
            
            if old_text not in modified:
                return ToolCallResult(
                    tool_call_id=tool_call_id,
                    tool_name="edit",
                    params=params,
                    content=[{
                        "type": "text",
                        "text": f"Error: Could not find oldText in file:\n```\n{old_text[:200]}\n```\nFile content at path {path} has changed or the text doesn't match exactly.",
                    }],
                    is_error=True,
                    duration_ms=(time.time() - start) * 1000,
                )
            
            # Warn if oldText appears multiple times (only 1st occurrence replaced)
            count = modified.count(old_text)
            if count > 1:
                import warnings
                warnings.warn(f"oldText appears {count} times in {path}. Only first occurrence replaced.")
            
            modified = modified.replace(old_text, new_text, 1)
        
        await asyncio.to_thread(full_path.write_text, modified, encoding="utf-8")
        
        # Generate proper unified diff
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff_lines = list(difflib.unified_diff(
            original_lines, modified_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}",
            n=3,  # context lines
        ))
        
        changes = len(edits)
        added = max(0, len(modified_lines) - len(original_lines))
        removed = max(0, len(original_lines) - len(modified_lines))
        
        # Format diff for display (LLM sees plain +/- text)
        if diff_lines:
            diff_text = "".join(diff_lines)
            # Truncate if too long (max ~3000 chars for LLM context)
            if len(diff_text) > 3000:
                diff_text = diff_text[:3000] + "\n... (diff truncated)"
        else:
            diff_text = f"{changes} change(s), +{added}/-{removed} lines (no diff)"
        
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name="edit",
            params=params,
            content=[{
                "type": "text",
                "text": f"Edited {path}: {changes} change(s), +{added}/-{removed} lines\n\n{diff_text}",
            }],
            details={
                "path": str(full_path),
                "changes": changes,
                "added_lines": added,
                "removed_lines": removed,
                "diff": diff_text,
            },
            duration_ms=(time.time() - start) * 1000,
        )


# ── Bash Tool ────────────────────────────────────────────────────────────────

class BashTool(AgentTool):
    name = "bash"
    label = "Bash"
    description = """Execute a bash command in the current working directory. Returns stdout and stderr.
Output is truncated to last 2000 lines or 50KB (whichever is hit first).
Optionally provide a timeout in seconds."""
    
    DESTRUCTIVE_PATTERNS = [
        "rm -rf", "rm -r", "sudo rm", "mkfs", "dd if=",
        ":(){ :|:& };", "chmod 777", "chown -R", "> /dev/",
        "shutdown", "reboot", "init 0", "init 6",
    ]
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
        },
        "required": ["command"],
    }
    
    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd).resolve()
    
    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable] = None,
    ) -> ToolCallResult:
        start = time.time()
        command = params["command"]
        timeout = params.get("timeout")
        
        # Check for destructive patterns - warn but don't block
        warning = ""
        for pattern in self.DESTRUCTIVE_PATTERNS:
            if pattern in command:
                warning = f"⚠️  Destructive command detected: `{pattern}`"
                break
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolCallResult(
                    tool_call_id=tool_call_id,
                    tool_name="bash",
                    params=params,
                    content=[{
                        "type": "text",
                        "text": f"Command timed out after {timeout}s: {command}",
                    }],
                    is_error=True,
                    duration_ms=timeout * 1000,
                )
            
            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                if output:
                    output += "\n[stderr]\n"
                output += stderr.decode("utf-8", errors="replace")
            
            # Truncate if too large
            lines = output.split("\n")
            if len(lines) > 2000:
                output = "\n".join(lines[:2000]) + "\n... (output truncated to 2000 lines)"
            if len(output) > 50000:
                output = output[:50000] + "\n... (output truncated at 50KB)"
            
            if not output.strip():
                output = "(no output)"
            
            # Prepend destructive warning if detected
            if warning:
                output = warning + "\n" + output
            
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name="bash",
                params=params,
                content=[{"type": "text", "text": output}],
                details={
                    "exit_code": process.returncode,
                    "command": command,
                },
                duration_ms=(time.time() - start) * 1000,
            )
        
        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name="bash",
                params=params,
                content=[{"type": "text", "text": f"Error executing command: {e}"}],
                is_error=True,
                duration_ms=(time.time() - start) * 1000,
            )


# ── Tool Factory ─────────────────────────────────────────────────────────────

def create_default_tools(cwd: str = ".") -> List[AgentTool]:
    """Create the default set of built-in tools."""
    return [
        ReadTool(cwd),
        WriteTool(cwd),
        EditTool(cwd),
        BashTool(cwd),
    ]


def create_readonly_tools(cwd: str = ".") -> List[AgentTool]:
    """Create read-only tools (no write/edit/bash)."""
    return [ReadTool(cwd)]
