"""
Agent - the core LLM interaction loop.

Manages:
- Message history
- Tool calling loop
- Event streaming
- Thinking level
- Compaction detection

Mirrors pi's Agent class from @earendil-works/pi-agent-core.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from providers import ProviderAuthError, ProviderRegistry, get_auth_storage, get_provider_registry
from core.types import (
    AbortError,
    AgentEvent,
    AgentMessage,
    AgentTool,
    EventBus,
    EventType,
    Model,
    ThinkingLevel,
    ToolCallResult,
)
from tools import create_default_tools
from config.pricing import calculate_cost

# Cache tiktoken import at module level (avoid repeated import attempts)
try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except (ImportError, Exception):
    _TIKTOKEN_ENC = None
    _HAS_TIKTOKEN = False


class AgentState:
    """Mutable agent state."""
    
    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.model: Optional[Model] = None
        self.thinking_level: ThinkingLevel = ThinkingLevel.OFF
        self.system_prompt: str = ""
        self.tools: List[AgentTool] = []
        self.streaming_message: Optional[AgentMessage] = None
        self.error_message: Optional[str] = None


class Agent:
    """Core agent that manages LLM interaction."""
    
    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry] = None,
        tools: Optional[List[AgentTool]] = None,
        system_prompt: str = "",
        event_bus: Optional[EventBus] = None,
    ):
        self.provider_registry = provider_registry or get_provider_registry(get_auth_storage())
        self.state = AgentState()
        self.state.tools = tools or create_default_tools()
        self.state.system_prompt = system_prompt
        self.event_bus = event_bus or EventBus()
        
        self._is_streaming = False
        self._abort_flag = False
        self._abort_event: asyncio.Event = asyncio.Event()
        self._steering_queue: List[str] = []
        self._follow_up_queue: List[str] = []
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        
        # Token counting & auto-compaction
        self._token_threshold: int = 150000
        self._on_compact: Optional[Callable] = None
        self._last_token_count: int = 0
        self._token_count_dirty: bool = True  # Dirty flag for caching
        
        # Permission gate
        self.permission_gate_enabled: bool = False
        self._permission_allow_all: bool = False  # Allow all for this session
        self._permission_callback: Optional[Callable] = None  # Async callback(tool_name, params) -> bool
        
        # Cost tracking
        self._total_cost: float = 0.0
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_cache_read_tokens: int = 0
        
        # Subscribers
        self._subscribers: List[Callable] = []
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    def _count_tokens(self) -> int:
        """Estimate token count for current messages + system prompt.
        
        Uses tiktoken if available, otherwise char/4 heuristic.
        Cached via dirty flag; reset on message add or compaction.
        """
        if not self._token_count_dirty:
            return self._last_token_count
        
        text_parts = [self.state.system_prompt, "\n"]
        for msg in self.state.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            text_parts.append(content)
            text_parts.append("\n")
            for tc in msg.metadata.get("tool_calls", []):
                text_parts.append(json.dumps(tc))
                text_parts.append("\n")
        
        text = "".join(text_parts)
        
        # Use cached tiktoken import
        if _HAS_TIKTOKEN and _TIKTOKEN_ENC is not None:
            self._last_token_count = len(_TIKTOKEN_ENC.encode(text))
        else:
            # Fallback: ~4 chars per token (reasonable for English)
            self._last_token_count = len(text) // 4
        
        self._token_count_dirty = False
        return self._last_token_count
    
    def set_compaction_callback(self, threshold: int, callback: Callable) -> None:
        """Set auto-compaction threshold and callback.
        
        Args:
            threshold: Token count threshold to trigger compaction
            callback: Async callable to perform compaction
        """
        self._token_threshold = threshold
        self._on_compact = callback
    
    async def _maybe_compact(self) -> bool:
        """Check token count and auto-compact if needed. Returns True if compacted."""
        count = self._count_tokens()
        self._last_token_count = count
        
        if count >= self._token_threshold and self._on_compact:
            self._emit(AgentEvent(
                type=EventType.COMPACTION_START,
                data={"token_count": count, "threshold": self._token_threshold},
            ))
            await self._on_compact()
            # Re-count after compaction (dirty flag reset inside _count_tokens)
            self._token_count_dirty = True
            self._last_token_count = self._count_tokens()
            self._emit(AgentEvent(
                type=EventType.COMPACTION_END,
                data={"token_count": self._last_token_count},
            ))
            return True
        return False
    
    @property
    def token_count(self) -> int:
        return self._last_token_count or self._count_tokens()
    
    @property
    def total_cost(self) -> float:
        return self._total_cost
    
    @property
    def session_input_tokens(self) -> int:
        return self._session_input_tokens
    
    @property
    def session_output_tokens(self) -> int:
        return self._session_output_tokens
    
    def _track_usage(self, model: Model, usage: dict) -> None:
        """Track token usage and cost from a provider response."""
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0) or usage.get("cached_tokens", 0)
        
        self._session_input_tokens += input_tokens
        self._session_output_tokens += output_tokens
        self._session_cache_read_tokens += cache_read
        
        cost = calculate_cost(model.provider, model.model_id,
                              input_tokens, output_tokens, cache_read)
        self._total_cost += cost
    
    def subscribe(self, listener: Callable) -> Callable:
        """Subscribe to agent events. Returns unsubscribe function."""
        self._subscribers.append(listener)
        return lambda: self._subscribers.remove(listener)
    
    def _emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribers and the event bus."""
        for listener in self._subscribers:
            try:
                listener(event)
            except Exception:
                pass
        
        self.event_bus.emit(event.type.value, event)
    
    async def _run_tool(self, tool_call: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a tool call and return results."""
        tool_name = tool_call["name"]
        tool_input = tool_call.get("input", tool_call.get("arguments", {}))
        call_id = tool_call.get("id", "unknown")
        
        self._emit(AgentEvent(
            type=EventType.TOOL_EXECUTION_START,
            data={"tool_name": tool_name, "tool_call_id": call_id, "input": tool_input},
        ))
        
        # Find the tool
        tool = None
        for t in self.state.tools:
            if t.name == tool_name:
                tool = t
                break
        
        # Permission gate: ask user before executing
        if self.permission_gate_enabled and not self._permission_allow_all:
            if self._permission_callback is not None:
                allowed = await self._permission_callback(tool_name, tool_input)
                if not allowed:
                    return [{"type": "text", "text": f"Permission denied for '{tool_name}'"}]
                if allowed == "allow_all":
                    self._permission_allow_all = True
        
        if tool is None:
            result = ToolCallResult(
                tool_call_id=call_id,
                tool_name=tool_name,
                params=tool_input,
                content=[{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}],
                is_error=True,
            )
        else:
            try:
                result = await tool.execute(call_id, tool_input)
            except Exception as e:
                result = ToolCallResult(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    params=tool_input,
                    content=[{"type": "text", "text": f"Error: {e}\n{traceback.format_exc()}"}],
                    is_error=True,
                )
        
        self._emit(AgentEvent(
            type=EventType.TOOL_EXECUTION_END,
            data={
                "tool_name": tool_name,
                "tool_call_id": call_id,
                "is_error": result.is_error,
                "result": result,
            },
        ))
        
        return [{"type": "text", "text": result.content[0]["text"]} if result.content else []]
    
    async def _process_tool_calls(
        self,
        assistant_msg: AgentMessage,
    ) -> List[AgentMessage]:
        """Process tool calls from an assistant message.
        
        Independent tools are executed in parallel via asyncio.gather
        for 2-3x faster multi-tool turns.
        """
        tool_calls = assistant_msg.metadata.get("tool_calls", [])
        if not tool_calls:
            return []
        
        # Single tool: execute directly (no overhead)
        if len(tool_calls) == 1:
            tc = tool_calls[0]
            results = await self._run_tool(tc)
            return [AgentMessage(
                role="tool",
                content=json.dumps(results) if (results is not None and (not isinstance(results, list) or results)) else "(no output)",
                name=tc["name"],
                tool_call_id=tc["id"],
                parent_id=assistant_msg.id,
            )]
        
        # Multiple tools: execute in parallel
        async def run_one(tc):
            results = await self._run_tool(tc)
            return AgentMessage(
                role="tool",
                content=json.dumps(results) if (results is not None and (not isinstance(results, list) or results)) else "(no output)",
                name=tc["name"],
                tool_call_id=tc["id"],
                parent_id=assistant_msg.id,
            )
        
        return list(await asyncio.gather(*[run_one(tc) for tc in tool_calls]))
    
    async def run(
        self,
        model: Optional[Model] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        max_turns: int = 25,
    ) -> List[AgentMessage]:
        """Run the agent with the current state. Returns new messages."""
        if model:
            self.state.model = model
        if thinking_level:
            self.state.thinking_level = thinking_level
        
        if not self.state.model:
            raise ValueError("No model set")
        
        self._abort_flag = False
        self._abort_event.clear()
        self._is_streaming = True
        self._idle_event.clear()
        
        self._emit(AgentEvent(
            type=EventType.AGENT_START,
            data={"model": self.state.model.model_id},
        ))
        
        new_messages = []
        turns = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while turns < max_turns and not self._abort_flag:
                turns += 1
                
                # Auto-compact if token threshold reached
                await self._maybe_compact()
                
                self._emit(AgentEvent(type=EventType.TURN_START, data={"turn": turns}))
                
                provider = await self.provider_registry.get_provider(
                    self.state.model.provider
                )
                
                # Retry loop with exponential backoff
                max_retries = 3
                retry_delay = 1.0
                assistant_msg = None
                last_error = None
                
                for attempt in range(max_retries + 1):
                    if self._abort_flag or self._abort_event.is_set():
                        raise AbortError("Agent aborted")
                    try:
                        assistant_msg = await provider.chat(
                            model=self.state.model,
                            messages=self.state.messages,
                            tools=self.state.tools,
                            system_prompt=self.state.system_prompt,
                            thinking_level=self.state.thinking_level,
                            signal=self._abort_event,
                        )
                        break  # Success
                    except (AbortError, ProviderAuthError):
                        raise
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            self._emit(AgentEvent(
                                type=EventType.TURN_END,
                                data={"warning": f"Retry {attempt+1}/{max_retries} after error: {e}"},
                            ))
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                
                if assistant_msg is None:
                    raise RuntimeError(f"Provider failed after {max_retries} retries: {last_error}")
                
                assistant_msg.parent_id = self.state.messages[-1].id if self.state.messages else None
                self._track_usage(self.state.model, assistant_msg.metadata.get("usage", {}))
                self.state.messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                
                self._emit(AgentEvent(
                    type=EventType.MESSAGE_END,
                    data={"message": assistant_msg},
                ))
                
                # Check for tool calls
                tool_calls = assistant_msg.metadata.get("tool_calls", [])
                if not tool_calls:
                    # No tool calls - agent is done
                    break
                
                # Execute tools and add results
                tool_messages = await self._process_tool_calls(assistant_msg)
                for tm in tool_messages:
                    self.state.messages.append(tm)
                    new_messages.append(tm)
                    # Track tool errors without full JSON deserialize
                    content = tm.content if isinstance(tm.content, str) else ""
                    if "Error" in content[:200]:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                
                # Stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self._emit(AgentEvent(
                        type=EventType.MESSAGE_END,
                        data={"warning": "Too many consecutive tool errors, stopping"},
                    ))
                    break
                
                # Check for steering messages
                if self._steering_queue:
                    steering_text = self._steering_queue.pop(0)
                    user_msg = AgentMessage(role="user", content=steering_text)
                    user_msg.parent_id = self.state.messages[-1].id if self.state.messages else None
                    self.state.messages.append(user_msg)
                    new_messages.append(user_msg)
                
                self._emit(AgentEvent(
                    type=EventType.TURN_END,
                    data={"turn": turns, "message": assistant_msg},
                ))
            
            # Process follow-up messages
            while self._follow_up_queue and turns < max_turns and not self._abort_flag:
                follow_up = self._follow_up_queue.pop(0)
                user_msg = AgentMessage(role="user", content=follow_up)
                user_msg.parent_id = self.state.messages[-1].id if self.state.messages else None
                self.state.messages.append(user_msg)
                new_messages.append(user_msg)
                
                # Re-run agent for follow-up
                provider = await self.provider_registry.get_provider(self.state.model.provider)
                assistant_msg = await provider.chat(
                    model=self.state.model,
                    messages=self.state.messages,
                    tools=self.state.tools,
                    system_prompt=self.state.system_prompt,
                    thinking_level=self.state.thinking_level,
                    signal=self._abort_event,
                )
                
                assistant_msg.parent_id = user_msg.id
                self.state.messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                turns += 1
        
        except AbortError:
            # Graceful abort - just stop
            pass
        finally:
            self._is_streaming = False
            self._idle_event.set()
            
            self._emit(AgentEvent(
                type=EventType.AGENT_END,
                data={"messages": new_messages},
            ))
        
        return new_messages
    
    def abort(self) -> None:
        """Abort the current agent run."""
        self._abort_flag = True
        self._abort_event.set()
    
    async def wait_for_idle(self) -> None:
        """Wait until the agent is idle."""
        await self._idle_event.wait()
    
    def steer(self, text: str) -> None:
        """Queue a steering message for the next turn."""
        self._steering_queue.append(text)
    
    def follow_up(self, text: str) -> None:
        """Queue a follow-up message for after the agent finishes."""
        self._follow_up_queue.append(text)
    
    def add_to_history(self, message: AgentMessage) -> None:
        """Add a message to the conversation history."""
        self.state.messages.append(message)
        self._token_count_dirty = True
    
    async def stream_run(
        self,
        model: Optional[Model] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        max_turns: int = 25,
    ):
        """Run the agent with streaming. Yields events."""
        if model:
            self.state.model = model
        if thinking_level:
            self.state.thinking_level = thinking_level
        
        if not self.state.model:
            raise ValueError("No model set")
        
        self._abort_flag = False
        self._abort_event.clear()
        self._is_streaming = True
        self._idle_event.clear()
        
        self._emit(AgentEvent(
            type=EventType.AGENT_START,
            data={"model": self.state.model.model_id},
        ))
        
        new_messages = []
        turns = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while turns < max_turns and not self._abort_flag:
                turns += 1
                
                # Auto-compact if token threshold reached
                await self._maybe_compact()
                
                yield {"type": "turn_start", "turn": turns}
                self._emit(AgentEvent(type=EventType.TURN_START, data={"turn": turns}))
                
                provider = await self.provider_registry.get_provider(self.state.model.provider)
                
                # Stream the assistant response (with retry on failure)
                text_buffer = ""
                tool_calls: List[Dict[str, Any]] = []
                tool_input_buffers: Dict[int, str] = {}
                
                stream_ok = False
                max_stream_retries = 2
                stream_retry_delay = 1.0
                last_stream_error = None
                
                for stream_attempt in range(max_stream_retries + 1):
                    if self._abort_flag or self._abort_event.is_set():
                        raise AbortError("Agent aborted")
                    try:
                        text_buffer = ""
                        tool_calls = []
                        tool_input_buffers = {}
                        stream_usage = {}  # Capture usage from streaming events
                        
                        async for event in provider.stream_chat(
                            model=self.state.model,
                            messages=self.state.messages,
                            tools=self.state.tools,
                            system_prompt=self.state.system_prompt,
                            thinking_level=self.state.thinking_level,
                        ):
                            yield event
                            
                            if event["type"] == "text_delta":
                                text_buffer += event["text"]
                                self._emit(AgentEvent(
                                    type=EventType.MESSAGE_UPDATE,
                                    data={"text": event["text"]},
                                ))
                            elif event["type"] == "thinking_delta":
                                self._emit(AgentEvent(
                                    type=EventType.MESSAGE_UPDATE,
                                    data={"thinking": event["text"]},
                                ))
                            elif event["type"] == "tool_use_start":
                                idx = event.get("index", len(tool_calls))
                                call_id = event.get("id") or f"call_{event.get('name','unknown')}_{idx}_{id(event)}"
                                tool_calls.append({
                                    "id": call_id,
                                    "name": event["name"],
                                    "input": {},
                                })
                                tool_input_buffers[idx] = ""
                            elif event["type"] == "tool_input_delta":
                                idx = event.get("index", 0)
                                tool_input_buffers[idx] = tool_input_buffers.get(idx, "") + event["text"]
                                if idx < len(tool_calls):
                                    try:
                                        tool_calls[idx]["input"] = json.loads(tool_input_buffers[idx])
                                    except json.JSONDecodeError:
                                        pass
                            elif event["type"] == "message_end":
                                # Capture usage from streaming for cost tracking
                                if "usage" in event:
                                    stream_usage = event["usage"]
                        
                        stream_ok = True
                        break  # Success
                    except (AbortError, ProviderAuthError):
                        raise
                    except Exception as e:
                        last_stream_error = e
                        if stream_attempt < max_stream_retries:
                            self._emit(AgentEvent(
                                type=EventType.TURN_END,
                                data={"warning": f"Stream retry {stream_attempt+1}/{max_stream_retries} after error: {e}"},
                            ))
                            await asyncio.sleep(stream_retry_delay)
                            stream_retry_delay *= 2
                
                if not stream_ok:
                    raise RuntimeError(f"Stream provider failed after {max_stream_retries} retries: {last_stream_error}")
                
                # Check for abort after streaming
                if self._abort_flag or self._abort_event.is_set():
                    raise AbortError("Agent aborted")
                
                # Build assistant message
                assistant_msg = AgentMessage(
                    role="assistant",
                    content=text_buffer,
                )
                if tool_calls:
                    assistant_msg.metadata["tool_calls"] = tool_calls
                
                # Apply captured streaming usage for cost tracking
                if stream_usage:
                    assistant_msg.metadata["usage"] = stream_usage
                
                assistant_msg.parent_id = self.state.messages[-1].id if self.state.messages else None
                self._track_usage(self.state.model, assistant_msg.metadata.get("usage", {}))
                self.state.messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                
                yield {"type": "message_end", "message": assistant_msg}
                
                # Execute tools (parallel for multiple independent calls)
                if tool_calls:
                    # Announce all tools
                    for tc in tool_calls:
                        yield {"type": "tool_start", "name": tc["name"], "input": tc.get("input", {})}
                    
                    # Run tools in parallel
                    async def run_stream_tool(tc):
                        results = await self._run_tool(tc)
                        return AgentMessage(
                            role="tool",
                            content=json.dumps(results),
                            name=tc["name"],
                            tool_call_id=tc["id"],
                            parent_id=assistant_msg.id,
                        )
                    
                    tool_msgs = list(await asyncio.gather(*[run_stream_tool(tc) for tc in tool_calls]))
                    
                    # Check for abort after tool execution
                    if self._abort_flag or self._abort_event.is_set():
                        raise AbortError("Agent aborted")
                    
                    for tm in tool_msgs:
                        self.state.messages.append(tm)
                        new_messages.append(tm)
                        yield {"type": "tool_end", "name": tm.name}
                        # Track consecutive errors
                        content = tm.content if isinstance(tm.content, str) else ""
                        if "Error" in content[:200]:
                            consecutive_errors += 1
                        else:
                            consecutive_errors = 0
                    
                    # Stop on too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        yield {"type": "error", "message": "Too many consecutive tool errors, stopping"}
                        break
                else:
                    # No tool calls - agent done
                    break
                
                if self._steering_queue:
                    steering_text = self._steering_queue.pop(0)
                    user_msg = AgentMessage(role="user", content=steering_text)
                    user_msg.parent_id = self.state.messages[-1].id
                    self.state.messages.append(user_msg)
                    new_messages.append(user_msg)
                
                yield {"type": "turn_end", "turn": turns}
                self._emit(AgentEvent(type=EventType.TURN_END, data={"turn": turns}))
        
        finally:
            self._is_streaming = False
            self._idle_event.set()
            
            self._emit(AgentEvent(
                type=EventType.AGENT_END,
                data={"messages": new_messages},
            ))
            yield {"type": "agent_end", "messages": new_messages}
    
    def reset(self) -> None:
        """Reset agent state (clear history)."""
        self.state = AgentState()
        self._steering_queue.clear()
        self._follow_up_queue.clear()
        self._abort_flag = False
        self._abort_event.clear()
        self._token_count_dirty = True
