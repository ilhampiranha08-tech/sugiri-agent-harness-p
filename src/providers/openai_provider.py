"""OpenAI provider implementation."""

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
    Model,
    Provider as ProviderInterface,
    ThinkingLevel,
)

# OpenAI reasoning effort mapping
# GPT-5.x / o-series: none, low, medium, high, max
OPENAI_THINKING_EFFORT = {
    ThinkingLevel.OFF: None,
    ThinkingLevel.MINIMAL: "low",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.XHIGH: "max",
}


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(ProviderInterface):
    """OpenAI API provider (GPT-4, etc.)."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, shared_client: Optional[httpx.AsyncClient] = None):
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = shared_client
        self._owns_client = shared_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))  # 5 min for long-thinking models
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
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderAuthError("openai")
        return key

    def _convert_messages(self, messages: List[AgentMessage], system_prompt: str) -> List[Dict]:
        """Convert to OpenAI format, including tool_calls from history."""
        converted = []

        if system_prompt:
            # If first message is already a system message (from compaction),
            # prepend to it instead of creating a duplicate
            if messages and messages[0].role == "system":
                combined = system_prompt + "\n\n" + (messages[0].content if isinstance(messages[0].content, str) else "")
                converted.append({"role": "system", "content": combined})
                messages = messages[1:]  # Skip the already-processed system message
            else:
                converted.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })
                continue

            # Assistant message with tool_calls from previous turns
            if msg.role == "assistant" and msg.metadata.get("tool_calls"):
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content if isinstance(msg.content, str) else None,
                    "tool_calls": [],
                }
                for tc in msg.metadata["tool_calls"]:
                    entry["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    })
                # If no text content, set content to None (not empty string)
                if not entry["content"]:
                    entry["content"] = None
                converted.append(entry)
                continue

            # Regular text content
            if isinstance(msg.content, str):
                converted.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if block.type == "text":
                        parts.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.image:
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.image.media_type};base64,{block.image.data}"},
                        })
                converted.append({"role": msg.role, "content": parts})

        return converted

    def _convert_tools(self, tools: List[AgentTool]) -> List[Dict]:
        """Convert tools to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": getattr(tool, "parameters_schema", {
                        "type": "object",
                        "properties": {},
                    }),
                }
            }
            for tool in tools
        ]

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
        if signal and signal.is_set():
            raise AbortError("Agent aborted")
        client = await self._get_client()
        api_key = self._get_api_key()

        body = {
            "model": model.model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "max_tokens": max_tokens or model.max_output_tokens,
        }

        if tools:
            body["tools"] = self._convert_tools(tools)
            body["tool_choice"] = "auto"

        # OpenAI reasoning effort (o4-mini, o1, etc.)
        if model.supports_thinking and thinking_level != ThinkingLevel.OFF:
            effort = OPENAI_THINKING_EFFORT.get(thinking_level, "medium")
            if effort:
                body["reasoning_effort"] = effort

        response = await client.post(
            OPENAI_API_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        msg_data = choice["message"]

        content = msg_data.get("content") or ""
        result = AgentMessage(role="assistant", content=content)

        if msg_data.get("tool_calls"):
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args,
                })
            result.metadata["tool_calls"] = tool_calls

        result.metadata["usage"] = data.get("usage", {})
        return result

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
        client = await self._get_client()
        api_key = self._get_api_key()

        body = {
            "model": model.model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "max_tokens": max_tokens or model.max_output_tokens,
            "stream": True,
        }

        if tools:
            body["tools"] = self._convert_tools(tools)
            body["tool_choice"] = "auto"

        # OpenAI reasoning effort
        if model.supports_thinking and thinking_level != ThinkingLevel.OFF:
            effort = OPENAI_THINKING_EFFORT.get(thinking_level, "medium")
            if effort:
                body["reasoning_effort"] = effort

        async with client.stream(
            "POST",
            OPENAI_API_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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
                        delta = data["choices"][0].get("delta", {})

                        if "content" in delta and delta["content"]:
                            yield {"type": "text_delta", "text": delta["content"]}

                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                tc_index = tc.get("index", 0)
                                # First chunk: contains id and name → tool_use_start
                                if tc.get("id") and tc.get("function") and tc["function"].get("name"):
                                    yield {
                                        "type": "tool_use_start",
                                        "id": tc["id"],
                                        "name": tc["function"]["name"],
                                        "index": tc_index,
                                    }
                                # Subsequent chunks: contains arguments fragments → tool_input_delta
                                if tc.get("function") and tc["function"].get("arguments"):
                                    yield {
                                        "type": "tool_input_delta",
                                        "text": tc["function"]["arguments"],
                                        "index": tc_index,
                                    }
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
