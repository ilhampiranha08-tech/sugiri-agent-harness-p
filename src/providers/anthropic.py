"""
Anthropic provider implementation using the Messages API.

Supports: streaming, tool use, thinking (extended reasoning), image inputs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from . import ProviderAuthError
from core.types import (
    AbortError,
    AgentMessage,
    AgentTool,
    ContentBlock,
    ImageContent,
    Model,
    Provider as ProviderInterface,
    ThinkingLevel,
)


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

THINKING_BUDGETS = {
    ThinkingLevel.OFF: None,
    ThinkingLevel.MINIMAL: 512,
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MEDIUM: 4096,
    ThinkingLevel.HIGH: 8192,
    ThinkingLevel.XHIGH: 16384,
}

# Adaptive thinking effort levels (Claude Opus 4.6+)
ADAPTIVE_EFFORT = {
    ThinkingLevel.OFF: None,
    ThinkingLevel.MINIMAL: "low",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.XHIGH: "max",
}

# Models that use adaptive thinking instead of budget_tokens
ADAPTIVE_THINKING_MODELS = {"claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"}


class AnthropicProvider(ProviderInterface):
    """Anthropic Claude API provider."""
    
    name = "anthropic"
    
    def __init__(self, api_key: Optional[str] = None, shared_client: Optional[httpx.AsyncClient] = None):
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = shared_client
        self._owns_client = shared_client is None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
            self._owns_client = True
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
    
    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ProviderAuthError("anthropic")
        return key
    
    def _convert_messages(self, messages: List[AgentMessage]) -> List[Dict]:
        """Convert our AgentMessage format to Anthropic API format."""
        converted = []
        for msg in messages:
            if msg.role == "system":
                continue

            # Tool result messages
            if msg.role == "tool" and msg.tool_call_id:
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    }]
                })
                continue

            role = "user" if msg.role == "user" else "assistant"

            # Assistant message with tool_use blocks from previous turns
            if msg.role == "assistant" and msg.metadata.get("tool_calls"):
                blocks = []
                # Text content
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    blocks.append({"type": "text", "text": text})
                # Tool use blocks from history
                for tc in msg.metadata["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    })
                converted.append({"role": role, "content": blocks})
                continue

            # Regular text content
            if isinstance(msg.content, str):
                converted.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list):
                blocks = []
                for block in msg.content:
                    if block.type == "text":
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.image:
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": block.image.source_type,
                                "media_type": block.image.media_type,
                                "data": block.image.data,
                            }
                        })
                converted.append({"role": role, "content": blocks})

        return converted
    
    def _convert_tools(self, tools: List[AgentTool]) -> List[Dict]:
        """Convert our tool format to Anthropic API format."""
        tool_defs = []
        for tool in tools:
            tool_defs.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": getattr(tool, "parameters_schema", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            })
        return tool_defs
    
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
        """Non-streaming chat request."""
        client = await self._get_client()
        api_key = self._get_api_key()
        
        body = {
            "model": model.model_id,
            "max_tokens": max_tokens or model.max_output_tokens,
            "system": system_prompt,
            "messages": self._convert_messages(messages),
        }
        
        if tools:
            body["tools"] = self._convert_tools(tools)
        
        # Thinking configuration — adaptive for Opus 4.6+, budget for older models
        if thinking_level != ThinkingLevel.OFF and model.supports_thinking:
            if model.model_id in ADAPTIVE_THINKING_MODELS:
                effort = ADAPTIVE_EFFORT.get(thinking_level, "high")
                body["thinking"] = {"type": "adaptive", "effort": effort}
            else:
                budget = THINKING_BUDGETS.get(thinking_level)
                if budget:
                    body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        
        if signal and signal.is_set():
            raise AbortError("Agent aborted")

        response = await client.post(
            ANTHROPIC_API_URL,
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        
        return self._parse_response(data)
    
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
        """Streaming chat request. Yields events."""
        client = await self._get_client()
        api_key = self._get_api_key()
        
        body = {
            "model": model.model_id,
            "max_tokens": max_tokens or model.max_output_tokens,
            "system": system_prompt,
            "messages": self._convert_messages(messages),
            "stream": True,
        }
        
        if tools:
            body["tools"] = self._convert_tools(tools)
        
        # Thinking configuration — adaptive for Opus 4.6+, budget for older models
        if thinking_level != ThinkingLevel.OFF and model.supports_thinking:
            if model.model_id in ADAPTIVE_THINKING_MODELS:
                effort = ADAPTIVE_EFFORT.get(thinking_level, "high")
                body["thinking"] = {"type": "adaptive", "effort": effort}
            else:
                budget = THINKING_BUDGETS.get(thinking_level)
                if budget:
                    body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        
        async with client.stream(
            "POST",
            ANTHROPIC_API_URL,
            json=body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if signal and signal.is_set():
                    raise AbortError("Agent aborted")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield self._parse_stream_event(data)
                    except json.JSONDecodeError:
                        continue
    
    def _parse_response(self, data: Dict) -> AgentMessage:
        """Parse a non-streaming response into AgentMessage."""
        text_parts = []
        tool_calls = []
        
        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "input": block["input"],
                })
        
        content = "".join(text_parts) if text_parts else ""
        
        msg = AgentMessage(role="assistant", content=content)
        
        if tool_calls:
            msg.metadata["tool_calls"] = tool_calls
        
        # Include thinking content if present
        thinking = data.get("thinking", "")
        if thinking:
            msg.metadata["thinking"] = thinking
        
        msg.metadata["usage"] = data.get("usage", {})
        return msg
    
    def _parse_stream_event(self, data: Dict) -> Dict[str, Any]:
        """Parse a streaming event."""
        event_type = data.get("type", "")
        
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return {"type": "text_delta", "text": delta.get("text", "")}
            elif delta.get("type") == "thinking_delta":
                return {"type": "thinking_delta", "text": delta.get("thinking", "")}
            elif delta.get("type") == "input_json_delta":
                return {"type": "tool_input_delta", "text": delta.get("partial_json", "")}
        
        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                return {
                    "type": "tool_use_start",
                    "id": block.get("id"),
                    "name": block.get("name"),
                }
        
        elif event_type == "content_block_stop":
            return {"type": "content_block_stop"}
        
        elif event_type == "message_start":
            msg = data.get("message", {})
            return {"type": "message_start", "message": msg}
        
        elif event_type == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            return {
                "type": "message_end",
                "stop_reason": delta.get("stop_reason"),
                "usage": usage,
            }
        
        elif event_type == "message_stop":
            return {"type": "message_stop"}
        
        elif event_type == "error":
            return {"type": "error", "error": data.get("error", {})}
        
        return {"type": event_type, "raw": data}


# ── Helper: system prompt builder ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are Sugiri, an AI coding agent created by Ilham Sugiri.
You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands
- edit: Make precise file edits
- write: Create or overwrite files

Guidelines:
- Be concise in your responses
- Show file paths clearly when working with files
- Use bash for file operations like ls, rg, find
- Use read to examine files
- Use edit for precise changes
"""
