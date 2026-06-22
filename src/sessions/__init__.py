"""
Session management with JSONL tree-structured storage.

Mirrors pi's session format: each entry has id, parentId for branching.
Sessions are stored as JSONL files in ~/.agent/sessions/.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import AgentMessage, ContentBlock, ImageContent, SessionEntry


def _serialize_content(msg: AgentMessage):
    """Serialize message content to JSON-safe format. Caches result on message."""
    cache_key = '_cached_serialized'
    if hasattr(msg, cache_key):
        return getattr(msg, cache_key)
    content = msg.content
    if isinstance(content, list):
        content = [dataclasses.asdict(block) if hasattr(block, '__dataclass_fields__') else block for block in content]
    setattr(msg, cache_key, content)
    return content


class SessionManager:
    """Manages session persistence with tree-structured JSONL format."""
    
    def __init__(self, session_dir: str, cwd: str = "."):
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.cwd = Path(cwd).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self._file_path: Optional[Path] = None
        self._entries: Dict[str, SessionEntry] = {}
        self._active_path_ids: List[str] = []
        self._root_id: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
    
    @property
    def session_id(self) -> str:
        return self._file_path.stem if self._file_path else "unsaved"
    
    @property
    def session_file(self) -> Optional[str]:
        return str(self._file_path) if self._file_path else None
    
    def create_new(self, name: Optional[str] = None) -> SessionManager:
        """Create a new empty session."""
        session_id = name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._file_path = self.session_dir / f"{session_id}.jsonl"
        self._entries = {}
        self._active_path_ids = []
        self._root_id = None
        self._metadata = {
            "created": datetime.now().isoformat(),
            "cwd": str(self.cwd),
            "session_id": session_id,
        }
        self._save()
        return self
    
    @classmethod
    def open(cls, path: str) -> SessionManager:
        """Open an existing session file."""
        file_path = Path(path).expanduser().resolve()
        sm = cls(str(file_path.parent))
        sm._file_path = file_path
        sm._load()
        return sm
    
    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str) -> Optional[SessionManager]:
        """Continue the most recent session for the given cwd."""
        session_path = Path(session_dir).expanduser().resolve()
        if not session_path.exists():
            return None
        
        # Find sessions matching this cwd
        sessions = sorted(
            session_path.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for session_file in sessions:
            try:
                sm = cls.open(str(session_file))
                if sm._metadata.get("cwd") == str(Path(cwd).resolve()):
                    return sm
            except Exception:
                continue
        
        return None
    
    @classmethod
    def in_memory(cls) -> SessionManager:
        """Create an in-memory session (no persistence)."""
        sm = cls("")
        sm._file_path = None
        return sm
    
    @classmethod
    def list_sessions(cls, session_dir: str, cwd: Optional[str] = None) -> List[Dict[str, Any]]:
        """List saved sessions, optionally filtered by cwd."""
        session_path = Path(session_dir).expanduser().resolve()
        if not session_path.exists():
            return []
        
        sessions = []
        for f in sorted(session_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                sm = cls.open(str(f))
                entry = {
                    "id": sm.session_id,
                    "file": str(f),
                    "created": sm._metadata.get("created", ""),
                    "cwd": sm._metadata.get("cwd", ""),
                    "message_count": len(sm._entries),
                }
                if cwd is None or sm._metadata.get("cwd") == str(Path(cwd).resolve()):
                    sessions.append(entry)
            except Exception:
                continue
        
        return sessions
    
    @classmethod
    def search_sessions(cls, session_dir: str, keyword: str = "", 
                        cwd: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search sessions by keyword in description or messages.
        
        Returns list of session info dicts sorted by recency.
        """
        session_path = Path(session_dir).expanduser().resolve()
        if not session_path.exists():
            return []
        
        keyword_lower = keyword.lower() if keyword else ""
        results = []
        
        for f in sorted(session_path.glob("*.jsonl"), 
                       key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                sm = cls.open(str(f))
                
                # Filter by cwd
                if cwd and sm._metadata.get("cwd") != str(Path(cwd).resolve()):
                    continue
                
                desc = sm._metadata.get("description", "")
                session_id = sm._metadata.get("session_id", f.stem)
                
                # Match keyword against description, session_id, or message content
                if keyword_lower:
                    matched = keyword_lower in desc.lower() or keyword_lower in session_id.lower()
                    if not matched:
                        # Search first 3 messages
                        for entry in sorted(sm._entries.values(), 
                                           key=lambda e: e.message.timestamp)[:3]:
                            content = entry.message.content
                            if isinstance(content, str) and keyword_lower in content.lower():
                                matched = True
                                break
                    if not matched:
                        continue
                
                results.append({
                    "id": session_id,
                    "file": str(f),
                    "created": sm._metadata.get("created", ""),
                    "cwd": sm._metadata.get("cwd", ""),
                    "message_count": len(sm._entries),
                    "description": desc[:60] or "(no description)",
                })
            except Exception:
                continue
        
        return results
    
    def _load(self) -> None:
        """Load session from JSONL file."""
        if not self._file_path or not self._file_path.exists():
            return
        
        with open(self._file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry_data = json.loads(line)
                    
                    if entry_data.get("type") == "metadata":
                        self._metadata = entry_data
                        continue
                    
                    if entry_data.get("type") == "message":
                        msg_data = entry_data.get("message", {})
                        raw_content = msg_data.get("content", "")
                        # Reconstruct ContentBlock list from saved dicts
                        if isinstance(raw_content, list):
                            content_blocks = []
                            for item in raw_content:
                                if isinstance(item, dict):
                                    img_data = item.get("image")
                                    image = None
                                    if img_data and isinstance(img_data, dict):
                                        image = ImageContent(
                                            type=img_data.get("type", "image"),
                                            source_type=img_data.get("source_type", "base64"),
                                            media_type=img_data.get("media_type", "image/png"),
                                            data=img_data.get("data", ""),
                                        )
                                    content_blocks.append(ContentBlock(
                                        type=item.get("type", "text"),
                                        text=item.get("text"),
                                        image=image,
                                    ))
                            content = content_blocks
                        else:
                            content = raw_content
                        message = AgentMessage(
                            id=entry_data["id"],
                            role=msg_data.get("role", "user"),
                            content=content,
                            name=msg_data.get("name"),
                            tool_call_id=msg_data.get("tool_call_id"),
                            parent_id=entry_data.get("parentId"),
                            timestamp=entry_data.get("timestamp", ""),
                            metadata=msg_data.get("metadata", {}),
                        )
                        
                        entry = SessionEntry(
                            id=entry_data["id"],
                            parent_id=entry_data.get("parentId"),
                            message=message,
                            label=entry_data.get("label"),
                        )
                        
                        self._entries[entry.id] = entry
                        
                        # Set root
                        if entry.parent_id is None:
                            self._root_id = entry.id
                        
                        # Build children
                        if entry.parent_id and entry.parent_id in self._entries:
                            self._entries[entry.parent_id].children_ids.append(entry.id)
                
                except json.JSONDecodeError:
                    continue
        
        # Compute active path (leaf traversal from root)
        self._compute_active_path()
    
    def _save(self) -> None:
        """Save session to JSONL file atomically (crash-safe).
        
        Writes to a .tmp file first, then renames over the original.
        Keeps a .bak backup of the previous version.
        """
        if not self._file_path:
            return
        
        import shutil
        
        lines = []
        
        # Write metadata
        lines.append(json.dumps({**self._metadata, "type": "metadata"}))
        
        # Write entries in creation order
        for entry in sorted(self._entries.values(), key=lambda e: e.message.timestamp):
            # Convert content to JSON-serializable format
            content = _serialize_content(entry.message)
            
            lines.append(json.dumps({
                "type": "message",
                "id": entry.id,
                "parentId": entry.parent_id,
                "label": entry.label,
                "message": {
                    "role": entry.message.role,
                    "content": content,
                    "name": entry.message.name,
                    "tool_call_id": entry.message.tool_call_id,
                    "metadata": entry.message.metadata,
                },
                "timestamp": entry.message.timestamp,
            }))
        
        # Atomic write: tmp → rename (crash-safe)
        tmp_path = str(self._file_path) + ".tmp"
        bak_path = str(self._file_path) + ".bak"
        
        with open(tmp_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        
        # Backup old file before overwriting
        if self._file_path.exists():
            try:
                shutil.copy2(str(self._file_path), bak_path)
            except Exception:
                pass
        
        # Atomic rename (on same filesystem, this is instant and crash-safe)
        os.replace(tmp_path, str(self._file_path))
    
    def _compute_active_path(self) -> None:
        """Compute the active path from root to current leaf."""
        self._active_path_ids = []
        
        if not self._root_id:
            return
        
        current = self._root_id
        self._active_path_ids.append(current)
        
        while current in self._entries:
            entry = self._entries[current]
            if entry.children_ids:
                # Follow the last child (latest branch)
                current = entry.children_ids[-1]
                self._active_path_ids.append(current)
            else:
                break
    
    def append_message(self, message: AgentMessage, parent_id: Optional[str] = None) -> str:
        """Append a message to the session. Returns the entry ID."""
        if parent_id is None and self._active_path_ids:
            parent_id = self._active_path_ids[-1]
        
        # Auto-description: use first user message as session description
        if not self._metadata.get("description") and message.role == "user":
            desc = message.content if isinstance(message.content, str) else ""
            self._metadata["description"] = desc[:80].strip() or "(empty)"
        
        entry = SessionEntry(
            id=message.id,
            parent_id=parent_id,
            message=message,
        )
        
        self._entries[entry.id] = entry
        
        if parent_id and parent_id in self._entries:
            parent = self._entries[parent_id]
            if entry.id not in parent.children_ids:
                parent.children_ids.append(entry.id)
        
        if self._root_id is None:
            self._root_id = entry.id
        
        # Update active path
        if parent_id and parent_id in self._active_path_ids:
            # If we're on the active path, extend it
            idx = self._active_path_ids.index(parent_id)
            self._active_path_ids = self._active_path_ids[:idx + 1] + [entry.id]
        else:
            self._compute_active_path()
        
        # Append-only: serialize and write single line (O(1) instead of O(n))
        self._append_entry_line(entry)
        return entry.id
    
    def _append_entry_line(self, entry: SessionEntry) -> None:
        """Append a single entry line to the session file (O(1))."""
        if not self._file_path:
            return
        
        content = _serialize_content(entry.message)
        
        line = json.dumps({
            "type": "message",
            "id": entry.id,
            "parentId": entry.parent_id,
            "label": entry.label,
            "message": {
                "role": entry.message.role,
                "content": content,
                "name": entry.message.name,
                "tool_call_id": entry.message.tool_call_id,
                "metadata": entry.message.metadata,
            },
            "timestamp": entry.message.timestamp,
        })
        
        # Track append count for periodic full-save (refreshes metadata header)
        if not hasattr(self, '_append_count'):
            self._append_count = 0
        self._append_count += 1
        
        # Write metadata + full file on first save; append-only thereafter
        # Every 50 appends, do an atomic full save to refresh metadata header
        if self._file_path.exists() and self._file_path.stat().st_size > 0:
            if self._append_count % 50 == 0:
                self._save()  # Atomic full rewrite
            else:
                # Append-only: write + fsync for crash-safety
                with open(self._file_path, "a") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        else:
            self._append_count = 0
            self._save()
    
    def branch(self, entry_id: str) -> bool:
        """Set active path leaf to the given entry (in-place branching)."""
        if entry_id not in self._entries:
            return False
        
        # Find path from root to entry
        path = [entry_id]
        current = entry_id
        
        while current in self._entries:
            entry = self._entries[current]
            if entry.parent_id:
                path.insert(0, entry.parent_id)
                current = entry.parent_id
            else:
                break
        
        self._active_path_ids = path
        self._save()
        return True
    
    def get_entry(self, entry_id: str) -> Optional[SessionEntry]:
        """Get an entry by ID."""
        return self._entries.get(entry_id)
    
    def get_children(self, entry_id: str) -> List[SessionEntry]:
        """Get child entries."""
        entry = self._entries.get(entry_id)
        if not entry:
            return []
        return [self._entries[cid] for cid in entry.children_ids if cid in self._entries]
    
    def get_leaf_entry(self) -> Optional[SessionEntry]:
        """Get the current leaf entry."""
        if self._active_path_ids:
            return self._entries.get(self._active_path_ids[-1])
        return None
    
    def get_active_messages(self) -> List[AgentMessage]:
        """Get all messages along the active path."""
        messages = []
        for entry_id in self._active_path_ids:
            if entry_id in self._entries:
                messages.append(self._entries[entry_id].message)
        return messages
    
    def get_all_entries(self) -> List[SessionEntry]:
        """Get all entries in the session."""
        return list(self._entries.values())
    
    def set_label(self, entry_id: str, label: str) -> None:
        """Set a label on an entry."""
        if entry_id in self._entries:
            self._entries[entry_id].label = label
            self._save()
    
    def get_label(self, entry_id: str) -> Optional[str]:
        """Get a label for an entry."""
        entry = self._entries.get(entry_id)
        return entry.label if entry else None
    
    def fork(self, from_entry_id: str, new_session_name: Optional[str] = None) -> SessionManager:
        """Create a new session from a specific entry, copying the path."""
        new_sm = SessionManager(str(self.session_dir))
        new_sm.create_new(new_session_name)
        new_sm._metadata["forked_from"] = str(self._file_path)
        new_sm._metadata["forked_from_entry"] = from_entry_id
        new_sm._metadata["cwd"] = self._metadata.get("cwd", str(self.cwd))
        
        # Copy entries along the path to from_entry_id
        path = [from_entry_id]
        current = from_entry_id
        while current in self._entries:
            entry = self._entries[current]
            if entry.parent_id:
                path.insert(0, entry.parent_id)
                current = entry.parent_id
            else:
                break
        
        # Copy entries
        id_map = {}
        for old_id in path:
            if old_id in self._entries:
                old_entry = self._entries[old_id]
                new_msg = AgentMessage(
                    role=old_entry.message.role,
                    content=old_entry.message.content,
                    name=old_entry.message.name,
                    tool_call_id=old_entry.message.tool_call_id,
                    parent_id=id_map.get(old_entry.parent_id) if old_entry.parent_id else None,
                )
                new_id = new_sm.append_message(new_msg, new_msg.parent_id)
                id_map[old_id] = new_id
        
        new_sm._save()
        return new_sm
    
    def import_from_jsonl(self, path: str) -> None:
        """Import a session from another JSONL file."""
        source = SessionManager.open(path)
        target_dir = str(self.session_dir) if self.session_dir else "/tmp"
        new_sm = SessionManager(target_dir)
        new_sm.create_new(f"imported-{source.session_id}")
        new_sm._metadata["cwd"] = source._metadata.get("cwd", str(self.cwd))
        
        id_map = {}
        # Import in order
        for entry in sorted(source._entries.values(), key=lambda e: e.message.timestamp):
            new_msg = AgentMessage(
                role=entry.message.role,
                content=entry.message.content,
                name=entry.message.name,
                tool_call_id=entry.message.tool_call_id,
                parent_id=id_map.get(entry.parent_id) if entry.parent_id else None,
            )
            new_id = new_sm.append_message(new_msg, new_msg.parent_id)
            id_map[entry.id] = new_id
        
        new_sm._save()
        
        # Replace current session with imported one
        self._file_path = new_sm._file_path
        self._entries = new_sm._entries
        self._active_path_ids = new_sm._active_path_ids
        self._root_id = new_sm._root_id
        self._metadata = new_sm._metadata
