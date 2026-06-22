"""
Run modes for Sugiri.

- interactive: Full TUI with chat, commands, keyboard shortcuts
- print: Single-shot, prints response and exits
- json: Outputs events as JSON lines

Mirrors pi's InteractiveMode, runPrintMode, runJsonMode.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from core.types import (
    AgentEvent,
    Model,
    ThinkingLevel,
    EventType,
)
from core.session import AgentSession
from providers import ProviderRegistry, get_provider_registry


class InteractiveMode:
    """Full interactive TUI mode."""
    
    def __init__(
        self,
        session: AgentSession,
        initial_message: Optional[str] = None,
        initial_images: Optional[List] = None,
    ):
        self._session = session
        self._initial_message = initial_message
        self._initial_images = initial_images or []
    
    async def run(self) -> None:
        """Start the interactive mode."""
        from ui import AgentTUI
        
        tui = AgentTUI(self._session)
        await tui.run(self._initial_message)


class PrintMode:
    """Print mode - single-shot, streams output to stdout in real-time."""
    
    def __init__(
        self,
        session: AgentSession,
        initial_message: str,
        mode: str = "text",
    ):
        self._session = session
        self._message = initial_message
        self._mode = mode
    
    async def run(self) -> None:
        """Run in print mode with real-time streaming output."""
        first_text = True
        async for event in self._session.prompt_stream(self._message):
            if event.get("type") == "text_delta":
                if first_text:
                    print()  # newline before first output
                    first_text = False
                print(event["text"], end="", flush=True)
            elif event.get("type") == "tool_start":
                print(f"\n🔧 {event.get('name', 'unknown')}...", flush=True)
            elif event.get("type") == "thinking_delta":
                # Thinking output: show dimmed in stderr so pipes stay clean
                import sys
                sys.stderr.write(event["text"])
                sys.stderr.flush()
        if not first_text:
            print()  # final newline


class JSONMode:
    """JSON mode - streams all events as JSON lines in real-time."""
    
    def __init__(self, session: AgentSession, initial_message: str):
        self._session = session
        self._message = initial_message
    
    async def run(self) -> None:
        """Run in JSON mode with real-time streaming."""
        async for event in self._session.prompt_stream(self._message):
            output = {
                "type": event.get("type", "unknown"),
                "data": {k: v for k, v in event.items() if k != "type"},
            }
            print(json.dumps(output, default=str), flush=True)


class RPCMode:
    """RPC mode - stdin/stdout JSONL-based protocol."""
    
    def __init__(self, session: AgentSession):
        self._session = session
        self._running = False
    
    async def run(self) -> None:
        """Run in RPC mode. Reads JSONL commands from stdin, writes responses to stdout."""
        self._running = True
        
        # Setup event output
        def on_event(event: AgentEvent) -> None:
            output = json.dumps({
                "type": event.type.value,
                "data": event.data,
            }, default=str)
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
        
        self._session.subscribe(on_event)
        
        # Read commands from stdin
        import sys
        while self._running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                cmd_type = cmd.get("type", "")
                
                if cmd_type == "prompt":
                    text = cmd.get("text", "")
                    async for event in self._session.prompt_stream(text):
                        output = json.dumps({
                            "type": event.get("type", "unknown"),
                            "data": {k: v for k, v in event.items() if k != "type"},
                        }, default=str)
                        sys.stdout.write(output + "\n")
                        sys.stdout.flush()
                
                elif cmd_type == "abort":
                    await self._session.abort()
                
                elif cmd_type == "set_model":
                    provider = cmd.get("provider", "")
                    model_id = cmd.get("model", "")
                    registry = get_provider_registry()
                    model = registry.find_model(provider, model_id)
                    if model:
                        await self._session.set_model(model)
                
                elif cmd_type == "set_thinking":
                    level = cmd.get("level", "off")
                    try:
                        self._session.set_thinking_level(ThinkingLevel(level))
                    except ValueError:
                        pass
                
                elif cmd_type == "new_session":
                    await self._session.new_session(cmd.get("name"))
                
                elif cmd_type == "quit":
                    self._running = False
                    break
            
            except Exception as e:
                error_output = json.dumps({
                    "type": "error",
                    "data": {"message": str(e)},
                })
                sys.stdout.write(error_output + "\n")
                sys.stdout.flush()
