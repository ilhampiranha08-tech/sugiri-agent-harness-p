"""Google Gemini provider implementation."""

from __future__ import annotations

import json
import os
import uuid
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

# Gemini thinking budget mapping (mirrors Anthropic)
THINKING_BUDGETS = {
    ThinkingLevel.OFF: None,
    ThinkingLevel.MINIMAL: 512,
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MEDIUM: 4096,
    ThinkingLevel.HIGH: 8192,
    ThinkingLevel.XHIGH: 16384,
}


class GoogleProvider(ProviderInterface):
    """Google Gemini API provider."""

    name = "google"

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
        key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ProviderAuthError("google")
        return key

    def _convert_messages(self, messages: List[AgentMessage]) -> List[Dict]:
        """Convert AgentMessage list to Gemini contents format.

        Handles: user, assistant, tool (functionResponse) roles.
        """
        contents = []
        for msg in messages:
            if msg.role == "user":
                parts = []
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == "text":
                            parts.append({"text": block.text})
                        elif block.type == "image" and block.image:
                            parts.append({
                                "inline_data": {
                                    "mime_type": block.image.media_type,
                                    "data": block.image.data,
                                }
                            })
                contents.append({"role": "user", "parts": parts})

            elif msg.role == "assistant":
                parts = []
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == "text":
                            parts.append({"text": block.text})

                # Include previous tool calls so Gemini sees them
                if msg.metadata.get("tool_calls"):
                    for tc in msg.metadata["tool_calls"]:
                        parts.append({
                            "functionCall": {
                                "name": tc["name"],
                                "args": tc["input"],
                            }
                        })

                contents.append({"role": "model", "parts": parts})

            elif msg.role == "tool":
                # Gemini expects functionResponse in a "user" turn.
                # Extract the actual text content from the tool result format
                # (which is a JSON array like [{"type":"text","text":"..."}])
                try:
                    parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(parsed, list):
                        # Extract text from structured tool result
                        parts_list = []
                        for item in parsed:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts_list.append(item.get("text", ""))
                        response_text = "\n".join(parts_list) if parts_list else json.dumps(parsed)
                    elif isinstance(parsed, dict):
                        response_text = parsed.get("text", json.dumps(parsed))
                    else:
                        response_text = str(parsed)
                except (json.JSONDecodeError, TypeError):
                    response_text = str(msg.content)

                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.name or "unknown",
                            "response": {"result": response_text},
                        }
                    }]
                })

        return contents

    def _convert_tools(self, tools: List[AgentTool]) -> List[Dict]:
        """Convert tools to Gemini function declarations."""
        declarations = []
        for tool in tools:
            # Gemini requires type: OBJECT (uppercase) and no required field in properties
            schema = dict(getattr(tool, "parameters_schema", {
                "type": "OBJECT",
                "properties": {},
            }))
            if "type" in schema:
                schema["type"] = schema["type"].upper()
            declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            })
        return declarations

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

        contents = self._convert_messages(messages)

        gen_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens or model.max_output_tokens,
        }

        # Gemini thinking (Gemini 2.5 Pro supports thinkingConfig)
        if model.supports_thinking and thinking_level != ThinkingLevel.OFF:
            budget = THINKING_BUDGETS.get(thinking_level, 4096)
            gen_config["thinkingConfig"] = {"thinkingBudget": budget}

        body: Dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": gen_config,
        }

        if tools:
            body["tools"] = [{"function_declarations": self._convert_tools(tools)}]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.model_id}:generateContent?key={api_key}"

        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No response from Gemini")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        text_parts = []
        tool_calls = []
        call_index = 0

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                # Gemini doesn't return a unique call id; generate one
                call_id = f"call_{fc['name']}_{call_index}_{uuid.uuid4().hex[:8]}"
                call_index += 1
                tool_calls.append({
                    "id": call_id,
                    "name": fc["name"],
                    "input": fc.get("args", {}),
                })

        # Capture thoughts if present (Gemini returns thought in parts)
        thought_parts = []
        for part in parts:
            if "thought" in part:
                thought_parts.append(part["thought"])

        content = "".join(text_parts) if text_parts else ""
        msg = AgentMessage(role="assistant", content=content)
        if tool_calls:
            msg.metadata["tool_calls"] = tool_calls

        if thought_parts:
            msg.metadata["thinking"] = "\n".join(thought_parts)
        msg.metadata["usage"] = data.get("usageMetadata", {})
        return msg

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
        """Streaming with Gemini (server-sent events)."""
        if signal and signal.is_set():
            raise AbortError("Agent aborted")
        client = await self._get_client()
        api_key = self._get_api_key()

        contents = self._convert_messages(messages)

        gen_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens or model.max_output_tokens,
        }

        # Gemini thinking (Gemini 2.5 Pro supports thinkingConfig)
        if model.supports_thinking and thinking_level != ThinkingLevel.OFF:
            budget = THINKING_BUDGETS.get(thinking_level, 4096)
            gen_config["thinkingConfig"] = {"thinkingBudget": budget}

        body: Dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": gen_config,
        }

        if tools:
            body["tools"] = [{"function_declarations": self._convert_tools(tools)}]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.model_id}:streamGenerateContent?alt=sse&key={api_key}"

        tool_seen: List[str] = []
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        candidates = data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield {"type": "text_delta", "text": part["text"]}
                                elif "thought" in part:
                                    yield {"type": "thinking_delta", "text": part["thought"]}
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    # Gemini doesn't return a unique call id; generate one
                                    call_id = f"call_{fc['name']}_{uuid.uuid4().hex[:8]}"
                                    args = fc.get("args", {})
                                    yield {
                                        "type": "tool_use_start",
                                        "id": call_id,
                                        "name": fc["name"],
                                        "index": len(tool_seen),
                                    }
                                    # Gemini sends complete args at once → emit as single delta
                                    yield {
                                        "type": "tool_input_delta",
                                        "text": json.dumps(args),
                                        "index": len(tool_seen),
                                    }
                                    tool_seen.append(call_id)
                    except json.JSONDecodeError:
                        continue
