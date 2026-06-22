"""
Core type definitions and abstractions for Sugiri.

Mirrors pi's architecture: AgentSession, events, tools, models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from datetime import datetime
import asyncio


class AbortError(Exception):
    """Raised when an agent operation is aborted."""
    pass
import uuid


# ── Model & Thinking ────────────────────────────────────────────────────────

class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class Model:
    """Represents an LLM model."""
    provider: str
    model_id: str
    display_name: Optional[str] = None
    context_window: int = 200000
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_images: bool = False
    supports_thinking: bool = False


# ── Messages ─────────────────────────────────────────────────────────────────

@dataclass
class ImageContent:
    type: str = "image"
    source_type: str = "base64"  # base64 or url
    media_type: str = "image/png"
    data: str = ""


@dataclass
class ContentBlock:
    type: str  # "text" or "image"
    text: Optional[str] = None
    image: Optional[ImageContent] = None


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str = "user"  # user, assistant, system, tool
    content: Union[str, List[ContentBlock]] = ""
    name: Optional[str] = None  # tool name for tool results
    tool_call_id: Optional[str] = None
    parent_id: Optional[str] = None  # for tree structure in session
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Tools ────────────────────────────────────────────────────────────────────

class ToolResult(ABC):
    """Base class for tool results."""
    pass


@dataclass
class ToolCallResult:
    tool_call_id: str
    tool_name: str
    params: Dict[str, Any]
    content: List[Dict[str, Any]]
    is_error: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0


class AgentTool(ABC):
    """Abstract base for all tools."""
    
    name: str = "unknown"
    label: str = "Unknown"
    description: str = "No description"
    
    @abstractmethod
    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        signal: Optional[Any] = None,
        on_update: Optional[Callable] = None,
    ) -> ToolCallResult:
        ...


# ── Events ───────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"
    SESSION_START = "session_start"
    QUEUE_UPDATE = "queue_update"
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"


@dataclass
class AgentEvent:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


# ── Provider Interface ──────────────────────────────────────────────────────

class Provider(ABC):
    """Abstract LLM provider."""
    
    name: str = "unknown"
    
    @abstractmethod
    async def chat(
        self,
        model: Model,
        messages: List[AgentMessage],
        tools: List[AgentTool],
        system_prompt: str,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
        max_tokens: Optional[int] = None,
        signal: Optional[Any] = None,
    ) -> AgentMessage:
        """Send a chat request and return assistant response."""
        ...
    
    @abstractmethod
    async def stream_chat(
        self,
        model: Model,
        messages: List[AgentMessage],
        tools: List[AgentTool],
        system_prompt: str,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
        max_tokens: Optional[int] = None,
        signal: Optional[Any] = None,
    ):
        """Stream chat response. Yields text deltas, tool calls, etc."""
        ...


# ── Skill ────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str = "local"
    content: Optional[str] = None


# ── Prompt Template ──────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    name: str
    description: str
    source: str
    content: str


# ── Extension Context ────────────────────────────────────────────────────────

@dataclass
class ExtensionAPI:
    """API passed to extensions, mirrors pi's ExtensionAPI."""
    
    register_tool: Callable
    register_command: Callable
    register_shortcut: Callable
    on: Callable  # event subscription
    emit: Callable  # event emission
    append_entry: Callable  # session persistence
    
    # Access to parent session
    session: Optional[Any] = None
    
    # UI context available in interactive mode
    ui: Optional[Any] = None


# ── Session ──────────────────────────────────────────────────────────────────

@dataclass
class SessionEntry:
    id: str
    parent_id: Optional[str]
    message: AgentMessage
    children_ids: List[str] = field(default_factory=list)
    label: Optional[str] = None


# ── Settings ─────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    # Model settings
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-5"
    default_thinking: ThinkingLevel = ThinkingLevel.OFF
    
    # Compaction
    compaction_enabled: bool = True
    compaction_trigger_tokens: int = 150000
    
    # Retry
    retry_enabled: bool = True
    max_retries: int = 3
    
    # UI
    theme: str = "dark"
    show_thinking: bool = True
    
    # Extensions & skills
    enabled_extensions: List[str] = field(default_factory=list)
    enabled_skills: List[str] = field(default_factory=list)
    
    # Telemetry
    enable_telemetry: bool = False


# ── Event Bus ────────────────────────────────────────────────────────────────

class EventBus:
    """Simple event bus for extension communication."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def off(self, event: str, callback: Callable) -> None:
        if event in self._listeners:
            self._listeners[event] = [
                c for c in self._listeners[event] if c != callback
            ]
    
    def emit(self, event: str, *args, **kwargs) -> None:
        if event in self._listeners:
            for callback in self._listeners[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    # Don't let listener errors propagate
                    pass
