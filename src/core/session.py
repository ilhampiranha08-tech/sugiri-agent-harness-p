"""
AgentSession - combines Agent with session persistence.

The main high-level API for interacting with the agent.
Mirrors pi's AgentSession interface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import SettingsManager
from core.agent import Agent
from core.types import (
    AgentMessage,
    AgentTool,
    EventBus,
    ExtensionAPI,
    ImageContent,
    Model,
    ThinkingLevel,
)
from extensions import ExtensionRuntime, ResourceLoader
from providers import AuthStorage, ProviderAuthError, ProviderRegistry, get_auth_storage, get_provider_registry
from sessions import SessionManager
from tools import create_default_tools


# Pre-compiled regex for prompt template placeholder replacement
import re as _re
_PLACEHOLDER_RE = _re.compile(r'\{\{(\w+)\}\}')

class AgentSession:
    """High-level agent session with persistence, extensions, and event streaming.

    Usage:
        session = await AgentSession.create(
            model=some_model,
            tools=["read", "bash", "edit", "write"],
        )

        session.subscribe(lambda event: print(event))

        await session.prompt("What files are in this directory?")
    """

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager,
        settings_manager: SettingsManager,
        resource_loader: ResourceLoader,
        extension_runtime: ExtensionRuntime,
        provider_registry: ProviderRegistry,
        auth_storage: AuthStorage,
        model: Optional[Model] = None,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
    ):
        self._agent = agent
        self._session_manager = session_manager
        self._settings_manager = settings_manager
        self._resource_loader = resource_loader
        self._extension_runtime = extension_runtime
        self._provider_registry = provider_registry
        self._auth_storage = auth_storage

        if model:
            self._agent.state.model = model
        self._agent.state.thinking_level = thinking_level

        self._subscribers: List[Callable] = []
        
        # Remember mode: auto-summarize session on dispose
        self.remember_enabled: bool = False

        # Load existing messages from session
        for msg in session_manager.get_active_messages():
            self._agent.add_to_history(msg)

    @staticmethod
    async def _auto_compact(agent: Agent) -> None:
        """Auto-compact: summarize older messages, keep recent ones."""
        msgs = agent.state.messages
        if len(msgs) < 10:
            return

        keep_count = min(5, len(msgs) // 3)
        keep = msgs[-keep_count:]
        to_summarize = msgs[:-keep_count]

        # Build conversation text
        lines = []
        for msg in to_summarize:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 1000:
                content = content[:1000] + "..."
            role = msg.role
            if msg.name:
                role += f"({msg.name})"
            lines.append(f"[{role}]: {content}")

        summary_prompt = (
            "Summarize this conversation. Focus on: 1) what was asked, "
            "2) what was done, 3) key decisions, 4) current state. Be concise.\n\n"
            + "\n".join(lines)
        )

        model = agent.state.model
        if model is None:
            summary = f"[Previous conversation - {len(to_summarize)} messages]"
        else:
            try:
                provider = await agent.provider_registry.get_provider(model.provider)
                summary_msg = AgentMessage(role="user", content=summary_prompt)
                summary_response = await asyncio.wait_for(
                    provider.chat(
                        model=model, messages=[summary_msg], tools=[],
                        system_prompt="Summarize concisely.",
                        thinking_level=ThinkingLevel.OFF, max_tokens=1024,
                    ),
                    timeout=30.0,  # Prevent hanging the agent on compaction
                )
                summary = summary_response.content if isinstance(summary_response.content, str) else str(summary_response.content)
                summary = f"## Previous conversation summary\n\n{summary.strip()}"
            except Exception:
                summary = f"[Previous conversation - {len(to_summarize)} messages]"

        agent.state.messages = [AgentMessage(role="system", content=summary), *keep]
        agent._token_count_dirty = True  # Force recount on next check

    @classmethod
    async def create(
        cls,
        *,
        cwd: str = ".",
        agent_dir: str = "~/.agent",
        model: Optional[Model] = None,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
        tools: Optional[List[str]] = None,
        custom_tools: Optional[List[AgentTool]] = None,
        session_manager: Optional[SessionManager] = None,
        settings_manager: Optional[SettingsManager] = None,
        resource_loader: Optional[ResourceLoader] = None,
        extension_runtime: Optional[ExtensionRuntime] = None,
        event_bus: Optional[EventBus] = None,
        disable_extensions: bool = False,
        disable_skills: bool = False,
        disable_context_files: bool = False,
        system_prompt_override: Optional[str] = None,
        no_session: bool = False,
        session_name: Optional[str] = None,
    ) -> AgentSession:
        """Factory method to create an AgentSession."""

        # Set up config
        sm = settings_manager or SettingsManager.create(cwd, agent_dir)
        auth = get_auth_storage(agent_dir)
        registry = get_provider_registry(auth)
        eb = event_bus or EventBus()

        # Set up sessions
        if session_manager:
            sess_mgr = session_manager
        elif no_session:
            sess_mgr = SessionManager.in_memory()
        elif session_name:
            sess_mgr = SessionManager(sm.get_session_dir(), cwd)
            sess_mgr.create_new(session_name)
        else:
            sess_mgr = SessionManager(sm.get_session_dir(), cwd)
            sess_mgr.create_new()

        # Build system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = cls._build_system_prompt(
                resource_loader or ResourceLoader(cwd, agent_dir),
                disable_context_files=disable_context_files,
                disable_skills=disable_skills,
            )

        # Build tools
        all_tools = create_default_tools(cwd)

        if tools:
            all_tools = [t for t in all_tools if t.name in tools]

        if custom_tools:
            all_tools.extend(custom_tools)

        # Set up extensions
        ext_runtime = extension_runtime or ExtensionRuntime(eb, {t.name: t for t in all_tools})

        if resource_loader and not disable_extensions:
            ext_dirs = resource_loader.get_extension_directories()
            # Also load from settings
            for ext_path in sm.get("enabled_extensions", []):
                ext_runtime.load_extension_from_path(ext_path)
            ext_runtime.discover_and_load(ext_dirs)

        # Add extension tools
        for tool in ext_runtime.get_custom_tools().values():
            if tool.name not in [t.name for t in all_tools]:
                all_tools.append(tool)

        # Create agent
        agent = Agent(
            provider_registry=registry,
            tools=all_tools,
            system_prompt=system_prompt,
            event_bus=eb,
        )

        # Wire up auto-compaction
        settings = sm.get_settings()
        if settings.compaction_enabled:
            agent.set_compaction_callback(
                threshold=settings.compaction_trigger_tokens,
                callback=lambda: cls._auto_compact(agent),
            )

        # Set up permission gate callback (will be overridden by TUI)
        # In non-interactive mode with permission gate on:
        # - bash commands: warn but allow (no terminal to confirm)
        # - other tools: allow
        async def _default_permission(tool_name: str, params: dict) -> bool:
            import sys
            if not sys.stdin.isatty() and tool_name == "bash":
                cmd = params.get("command", "")
                dangerous = ["rm -rf", "sudo rm", "mkfs", "dd if=", "> /dev/", 
                            "shutdown", "reboot", "chmod 777"]
                for pat in dangerous:
                    if pat in cmd:
                        return False  # Deny destructive commands in non-TTY mode
            return True
        agent._permission_callback = _default_permission

        # Determine model
        final_model = model
        if not final_model:
            settings = sm.get_settings()
            final_model = registry.find_model(
                settings.default_provider,
                settings.default_model,
            )

        if not final_model:
            # Fall back to first available
            models = registry.list_models()
            if models:
                final_model = models[0]

        session = cls(
            agent=agent,
            session_manager=sess_mgr,
            settings_manager=sm,
            resource_loader=resource_loader or ResourceLoader(cwd, agent_dir),
            extension_runtime=ext_runtime,
            provider_registry=registry,
            auth_storage=auth,
            model=final_model,
            thinking_level=thinking_level,
        )
        
        # Restore persistent settings
        if sm.get("memory_enabled", False):
            session.remember_enabled = True
        
        return session

    @staticmethod
    def _build_system_prompt(
        loader: ResourceLoader,
        disable_context_files: bool = False,
        disable_skills: bool = False,
    ) -> str:
        """Build the system prompt from defaults, context files, and skills."""
        from providers.anthropic import DEFAULT_SYSTEM_PROMPT

        parts = [DEFAULT_SYSTEM_PROMPT]

        # Add context files (AGENTS.md)
        if not disable_context_files:
            context_files = loader.get_context_files()
            if context_files:
                parts.append("\n\n---\n## Project Context\n")
                for cf in context_files:
                    parts.append(f"\n<!-- From {cf['path']} -->\n{cf['content']}")

        # Add skills
        if not disable_skills:
            skills = loader.get_skills()
            if skills:
                parts.append("\n\n---\n## Available Skills\n")
                for skill in skills:
                    parts.append(f"\n### {skill.name}\n{skill.description}")
                parts.append(
                    "\nTo use a skill, read its SKILL.md file with the read tool."
                )

        return "".join(parts)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def model(self) -> Optional[Model]:
        return self._agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._agent.state.thinking_level

    @property
    def messages(self) -> List[AgentMessage]:
        return self._agent.state.messages

    @property
    def is_streaming(self) -> bool:
        return self._agent.is_streaming

    @property
    def session_id(self) -> str:
        return self._session_manager.session_id

    @property
    def session_file(self) -> Optional[str]:
        return self._session_manager.session_file

    def subscribe(self, listener: Callable) -> Callable:
        """Subscribe to agent events. Returns unsubscribe function."""
        self._subscribers.append(listener)
        return self._agent.subscribe(listener)

    async def prompt(
        self,
        text: str,
        *,
        images: Optional[List[ImageContent]] = None,
        expand_prompt_templates: bool = False,
        model: Optional[Model] = None,
        thinking_level: Optional[ThinkingLevel] = None,
    ) -> None:
        """Send a prompt to the agent and wait for completion."""

        # Handle slash commands
        if text.startswith("/"):
            await self._handle_command(text)
            return

        # Handle prompt templates
        if expand_prompt_templates:
            text = self._expand_prompt_templates(text)

        # Build user message
        content: Any = text
        if images:
            from core.types import ContentBlock
            blocks = [ContentBlock(type="text", text=text)]
            for img in images:
                blocks.append(ContentBlock(type="image", image=img))
            content = blocks

        user_msg = AgentMessage(role="user", content=content)

        # Add to session
        last_id = None
        active_msgs = self._session_manager.get_active_messages()
        if active_msgs:
            last_id = active_msgs[-1].id

        entry_id = self._session_manager.append_message(user_msg, last_id)
        self._agent.add_to_history(user_msg)

        # Run agent
        try:
            new_messages = await self._agent.run(
                model=model or self.model,
                thinking_level=thinking_level or self.thinking_level,
            )
        except ProviderAuthError as e:
            self._notify(f"\n⚠️  No API key configured for provider \"{e.provider}\".\n"
                         f"    Please run /login to set your API key first.\n")
            return

        # Save assistant messages to session
        for msg in new_messages:
            last_id = self._session_manager.get_active_messages()[-1].id if self._session_manager.get_active_messages() else None
            self._session_manager.append_message(msg, last_id)

    async def prompt_stream(
        self,
        text: str,
        *,
        images: Optional[List[ImageContent]] = None,
        model: Optional[Model] = None,
        thinking_level: Optional[ThinkingLevel] = None,
    ):
        """Send a prompt and yield streaming events."""

        content: Any = text
        if images:
            from core.types import ContentBlock
            blocks = [ContentBlock(type="text", text=text)]
            for img in images:
                blocks.append(ContentBlock(type="image", image=img))
            content = blocks

        user_msg = AgentMessage(role="user", content=content)

        last_id = None
        active_msgs = self._session_manager.get_active_messages()
        if active_msgs:
            last_id = active_msgs[-1].id

        self._session_manager.append_message(user_msg, last_id)
        self._agent.add_to_history(user_msg)

        new_messages = []
        saved_ids = set()
        try:
            async for event in self._agent.stream_run(
                model=model or self.model,
                thinking_level=thinking_level or self.thinking_level,
            ):
                yield event
                # Save messages incrementally as they appear (survives early abort)
                msg = event.get("message")
                if msg and event.get("type") in ("message_end", "tool_end"):
                    if msg.id not in saved_ids:
                        saved_ids.add(msg.id)
                        last_id = None
                        active_msgs = self._session_manager.get_active_messages()
                        if active_msgs:
                            last_id = active_msgs[-1].id
                        self._session_manager.append_message(msg, last_id)
                # Also capture from agent_end (full list for completeness)
                if event.get("type") == "agent_end":
                    new_messages = event.get("messages", [])

            # Save any remaining messages not yet saved
            for msg in new_messages:
                if msg.id not in saved_ids:
                    last_id = None
                    active_msgs = self._session_manager.get_active_messages()
                    if active_msgs:
                        last_id = active_msgs[-1].id
                    self._session_manager.append_message(msg, last_id)
        except ProviderAuthError as e:
            self._notify(f"\n⚠️  No API key configured for provider \"{e.provider}\".\n"
                         f"    Please run /login to set your API key first.\n")

    def steer(self, text: str) -> None:
        """Queue a steering message during streaming."""
        if not self._agent.is_streaming:
            raise RuntimeError("Agent is not streaming. Use prompt() instead.")
        self._agent.steer(text)

    def follow_up(self, text: str) -> None:
        """Queue a follow-up message during streaming."""
        if not self._agent.is_streaming:
            raise RuntimeError("Agent is not streaming. Use prompt() instead.")
        self._agent.follow_up(text)

    async def abort(self) -> None:
        """Abort current agent operation."""
        self._agent.abort()

    async def set_model(self, model: Model) -> None:
        """Switch the model and remember it for next startup."""
        self._agent.state.model = model

        # Save as last used model
        from config import save_last_model
        save_last_model(
            self._settings_manager.get_agent_dir(),
            model.provider,
            model.model_id,
        )

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking level."""
        self._agent.state.thinking_level = level

    def toggle_thinking(self) -> str:
        """Toggle/cCycle thinking based on current model.
        Returns a human-readable description of the new state.

        - Anthropic/Google: cycle 6 levels (off→minimal→low→medium→high→xhigh→off)
        - DeepSeek V4: 3-level cycle (off → high → max → off)
        - OpenAI/Gemini Flash: not supported
        """
        model = self.model
        if not model:
            return "No model selected"

        provider = model.provider
        model_id = model.model_id

        # DeepSeek V4 - 3-level cycle (off → high → max → off)
        if provider == "deepseek":
            current = self._agent.state.thinking_level
            if current == ThinkingLevel.OFF:
                self._agent.state.thinking_level = ThinkingLevel.HIGH
                return "thinking:high"
            elif current in (ThinkingLevel.HIGH, ThinkingLevel.MINIMAL, ThinkingLevel.LOW, ThinkingLevel.MEDIUM):
                self._agent.state.thinking_level = ThinkingLevel.XHIGH
                return "thinking:max"
            else:
                self._agent.state.thinking_level = ThinkingLevel.OFF
                return "thinking:off"

        # Providers with 6-level thinking (Anthropic, Google Pro)
        if model.supports_thinking:
            levels = list(ThinkingLevel)
            current_idx = levels.index(self._agent.state.thinking_level)
            next_idx = (current_idx + 1) % len(levels)
            self._agent.state.thinking_level = levels[next_idx]
            return f"thinking:{levels[next_idx].value}"

        # Not supported
        return "thinking not supported for this model"

    def cycle_thinking_level(self) -> ThinkingLevel:
        """Cycle through thinking levels (legacy, for Anthropic-style)."""
        levels = list(ThinkingLevel)
        current_idx = levels.index(self._agent.state.thinking_level)
        next_idx = (current_idx + 1) % len(levels)
        self._agent.state.thinking_level = levels[next_idx]
        return levels[next_idx]

    async def compact(self, custom_instructions: Optional[str] = None) -> None:
        """Compact conversation using LLM summarization.

        Keeps the last few messages intact, summarizes older messages
        via the current LLM provider into a concise system message.
        """
        if self._agent.is_streaming:
            self._notify("Cannot compact while agent is working. Wait for it to finish.")
            return
        msgs = self._agent.state.messages
        if len(msgs) < 10:
            return

        # Keep last 5 messages for immediate context
        keep_count = min(5, len(msgs) // 3)
        keep = msgs[-keep_count:]
        to_summarize = msgs[:-keep_count]

        # Build text of messages to summarize
        lines = []
        for msg in to_summarize:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate very long tool outputs
            if len(content) > 1000:
                content = content[:1000] + "..."
            role = msg.role
            if msg.name:
                role += f"({msg.name})"
            lines.append(f"[{role}]: {content}")

        conversation_text = "\n".join(lines)

        # Build summarization prompt
        summary_prompt = (
            "Summarize the following conversation between a user and a coding agent. "
            "Focus on: 1) what the user asked for, 2) what was done/accomplished, "
            "3) key decisions made, 4) current state of work. "
            "Be concise but complete. This summary will replace the conversation history."
        )
        if custom_instructions:
            summary_prompt += f"\n\nAdditional context: {custom_instructions}"

        summary_prompt += f"\n\nConversation to summarize:\n{conversation_text}"

        # Use LLM to summarize
        model = self.model
        if model is None:
            # Fallback to basic truncation
            summary = f"[Previous conversation - {len(to_summarize)} messages summarized]"
        else:
            summary = None
            max_retries = 2
            retry_delay = 1.0

            for attempt in range(max_retries + 1):
                try:
                    provider = await self._provider_registry.get_provider(model.provider)

                    summary_msg = AgentMessage(role="user", content=summary_prompt)

                    summary_response = await asyncio.wait_for(
                        provider.chat(
                            model=model,
                            messages=[summary_msg],
                            tools=[],
                            system_prompt="You are a conversation summarizer. Output only the summary.",
                            thinking_level=ThinkingLevel.OFF,
                            max_tokens=2048,
                        ),
                        timeout=30.0,
                    )
                    summary = summary_response.content if isinstance(summary_response.content, str) else str(summary_response.content)
                    summary = f"## Previous conversation summary\n\n{summary.strip()}"
                    break
                except Exception as e:
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        summary = None

            if summary is None:
                summary = f"[Previous conversation - {len(to_summarize)} messages summarized]"

        # Replace with summary + recent messages
        self._agent.state.messages = [
            AgentMessage(role="system", content=summary),
            *keep,
        ]
        self._agent._token_count_dirty = True

    def navigate_tree(self, entry_id: str) -> bool:
        """Navigate to a different point in the session tree."""
        return self._session_manager.branch(entry_id)

    async def new_session(self, name: Optional[str] = None) -> None:
        """Start a new session."""
        self._session_manager.create_new(name)
        self._agent.reset()

    async def fork(self, entry_id: str) -> None:
        """Fork from a specific entry into a new session."""
        new_sm = self._session_manager.fork(entry_id)
        self._session_manager = new_sm
        self._agent.reset()
        for msg in new_sm.get_active_messages():
            self._agent.add_to_history(msg)

    async def dispose(self) -> None:
        """Clean up resources. Auto-summarize if remember mode is on."""
        try:
            if self.remember_enabled and len(self._agent.state.messages) >= 3:
                await self._remember_session()
        except Exception:
            pass  # Don't block cleanup if memory save fails
        try:
            self._agent.reset()
            await self._provider_registry.close_all()
        except Exception:
            pass  # Best-effort cleanup
    
    async def _remember_session(self) -> None:
        """Summarize current session and save to .agent/memory.md."""
        import os
        from datetime import datetime
        
        msgs = self._agent.state.messages
        # Build conversation text (last 20 messages max)
        lines = []
        for msg in msgs[-20:]:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 500:
                content = content[:500] + "..."
            role = msg.role
            if msg.name:
                role += f"({msg.name})"
            lines.append(f"[{role}]: {content}")
        
        conversation = "\n".join(lines)
        summary_prompt = (
            "Summarize this conversation in 3 short bullet points. "
            "Focus on what was asked and what was done. Be concise.\n\n"
            + conversation
        )
        
        summary = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Unknown summary"
        model = self.model
        if model:
            try:
                provider = await self._provider_registry.get_provider(model.provider)
                summary_msg = AgentMessage(role="user", content=summary_prompt)
                summary_response = await asyncio.wait_for(
                    provider.chat(
                        model=model, messages=[summary_msg], tools=[],
                        system_prompt="Summarize briefly.",
                        thinking_level=ThinkingLevel.OFF, max_tokens=256,
                    ),
                    timeout=15.0,
                )
                text = summary_response.content if isinstance(summary_response.content, str) else str(summary_response.content)
                summary = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {text.strip()}"
            except Exception:
                pass
        
        # Append to memory file
        cwd = self._session_manager.cwd if hasattr(self._session_manager, 'cwd') else Path.cwd()
        memory_file = Path(str(cwd)) / ".agent" / "memory.md"
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        existing = ""
        if memory_file.exists():
            lines = memory_file.read_text().strip().split("\n")
            # Keep only last 50 entries (line-based rotation, prevents bloat)
            if len(lines) > 50:
                lines = ["... (older entries truncated)"] + lines[-50:]
            existing = "\n".join(lines) + "\n"
        
        new_entry = f"- {summary}\n"
        memory_file.write_text(existing + new_entry)

    # ── Commands ────────────────────────────────────────────────────────────

    async def _export_session(self, path: str = "") -> None:
        """Export current session to a Markdown file."""
        import os
        from datetime import datetime
        
        if not path:
            name = self.session_id or "session"
            path = f"{name}.md"
        
        lines = []
        lines.append(f"# Sugiri Session Export")
        lines.append(f"")
        lines.append(f"- **Session**: {self.session_id}")
        lines.append(f"- **Model**: {self.model.display_name if self.model else 'none'}")
        lines.append(f"- **Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        for msg in self.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.role == "user":
                lines.append(f"### 🧑 You")
                lines.append(f"")
                lines.append(content)
                lines.append(f"")
            elif msg.role == "assistant":
                lines.append(f"### 🤖 Assistant")
                lines.append(f"")
                lines.append(content)
                lines.append(f"")
            elif msg.role == "tool":
                # Escape backticks in tool name/content to avoid markdown injection
                safe_name = msg.name.replace("`", "\\`") if msg.name else "unknown"
                short = content[:200] + ("..." if len(content) > 200 else "")
                short = short.replace("`", "\\`")
                lines.append(f"*🔧 `{safe_name}`*: {short}")
                lines.append(f"")
            elif msg.role == "system":
                safe_content = content[:200].replace("`", "\\`")
                lines.append(f"*System: {safe_content}*")
                lines.append(f"")
        
        text = "\n".join(lines)
        
        # Warn if file already exists
        if os.path.exists(path):
            self._notify(f"⚠️  File already exists, overwriting: {os.path.abspath(path)}")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        
        self._notify(f"Session exported to: {os.path.abspath(path)}")
    
    async def _handle_login(self) -> None:
        """Interactive API key setup like pi's /login."""
        import sys

        self._notify("""
=== Login / Set API Key ===

Select provider:
  [1] Anthropic (Claude)
  [2] OpenAI (GPT)
  [3] Google (Gemini)
  [4] DeepSeek (V4)
  [5] Cancel
""")

        try:
            choice = await self._read_input("Provider [1-5]: ")
        except (KeyboardInterrupt, EOFError):
            self._notify("Cancelled.")
            return

        provider_map = {"1": "anthropic", "2": "openai", "3": "google", "4": "deepseek"}
        provider = provider_map.get(choice.strip())

        if not provider:
            self._notify("Cancelled.")
            return

        self._notify(f"\nEnter API key for {provider}:")
        self._notify("(input will be hidden)\n")

        try:
            api_key = await self._read_secret("API key: ")
        except (KeyboardInterrupt, EOFError):
            self._notify("Cancelled.")
            return

        if not api_key.strip():
            self._notify("Cancelled - empty key.")
            return

        # Save to auth storage (persistent)
        self._auth_storage.save_auth(provider, api_key.strip())

        # Also set as runtime key for immediate use
        self._auth_storage.set_runtime_api_key(provider, api_key.strip())

        # Clear cached provider so it picks up the new key
        self._provider_registry.clear_cached_provider(provider)

        self._notify(f"\n✅ API key for '{provider}' saved to ~/.agent/auth.json")
        self._notify("You can now use /model to select a model.")

    async def _read_input(self, prompt: str) -> str:
        """Read a line of input."""
        import sys
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return (await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)).strip()

    async def _read_secret(self, prompt: str) -> str:
        """Read a secret (hidden) input if terminal supports it."""
        import sys
        try:
            import termios
            import tty
            sys.stdout.write(prompt)
            sys.stdout.flush()
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                result = ""
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ('\n', '\r'):
                        sys.stdout.write('\n')
                        break
                    elif ch == '\x7f':  # backspace
                        if result:
                            result = result[:-1]
                            sys.stdout.write('\b \b')
                    elif ch == '\x03':  # Ctrl+C
                        sys.stdout.write('\n')
                        raise KeyboardInterrupt
                    else:
                        result += ch
                        sys.stdout.write('*')
                    sys.stdout.flush()
                return result
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (ImportError, termios.error):
            # Fallback: visible input - warn BEFORE showing input
            self._notify("\n⚠️  Terminal does not support hidden input. Key will be visible below.")
            sys.stdout.write(prompt)
            sys.stdout.flush()
            result = (await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)).strip()
            return result

    async def _select_thinking_interactive(self) -> None:
        """Show interactive thinking level selector.

        Options depend on provider:
          - Claude/Gemini Pro: off, minimal, low, medium, high, xhigh
          - DeepSeek V4: off, high, max
          - Others: not supported
        """
        model = self.model
        if not model:
            self._notify("No model selected.")
            return

        provider = model.provider
        model_id = model.model_id

        # Determine available levels
        if provider == "deepseek":
            # DeepSeek V4: off, high, max
            levels = ["off", "high", "max"]
        elif model.supports_thinking:
            # Anthropic / Google Pro: 6-level cycle
            levels = [l.value for l in ThinkingLevel]
        else:
            self._notify("thinking not supported for this model")
            return

        from ui.selector import select_thinking
        import asyncio

        current = self.thinking_level.value
        loop = asyncio.get_event_loop()
        chosen = await loop.run_in_executor(None, select_thinking, current, levels)

        if chosen is None:
            self._notify("Thinking selection cancelled.")
            return

        try:
            # Map display names to enum values (DeepSeek uses "max" for xhigh)
            level_map = {"max": "xhigh", "high": "high", "off": "off"}
            chosen_value = level_map.get(chosen, chosen)
            level = ThinkingLevel(chosen_value)
            self.set_thinking_level(level)
            self._notify(f"thinking:{level.value}")
        except ValueError:
            self._notify(f"Invalid level: {chosen}")

    async def _select_model_interactive(self, models: List) -> None:
        """Show interactive model selector."""
        import sys

        if not models:
            self._notify("No models available.")
            return

        # Sort models by provider then name
        models = sorted(models, key=lambda m: (m.provider, m.model_id))

        # Run the selector in a thread (it uses raw terminal mode)
        from ui.selector import select_model

        # We need to run blocking terminal I/O in executor
        loop = asyncio.get_event_loop()
        chosen = await loop.run_in_executor(None, select_model, models)

        if chosen:
            await self.set_model(chosen)
            self._notify(f"Switched to {chosen.display_name or chosen.model_id} "
                        f"({chosen.provider})")
        else:
            self._notify("Model selection cancelled.")

    async def _handle_command(self, text: str) -> None:
        """Handle slash commands."""
        parts = text.split(maxsplit=1)
        command = parts[0].lstrip("/").lower()
        args = parts[1] if len(parts) > 1 else ""

        # Built-in commands
        if command == "model":
            models = self._provider_registry.list_models()
            if args:
                # Quick switch by name
                for m in models:
                    if args in m.model_id:
                        await self.set_model(m)
                        self._notify(f"Switched to {m.display_name or m.model_id}")
                        return
                self._notify(f"Model '{args}' not found")
            else:
                # Interactive selector
                await self._select_model_interactive(models)

        elif command == "thinking":
            await self._select_thinking_interactive()

        elif command == "session":
            info = (
                f"Session: {self.session_id}\n"
                f"File: {self.session_file or 'in-memory'}\n"
                f"Messages: {len(self.messages)}\n"
                f"Model: {self.model.model_id if self.model else 'none'}\n"
                f"Thinking: {self.thinking_level.value}"
            )
            self._notify(info)

        elif command == "compact":
            await self.compact(args if args else None)
            self._notify("Conversation compacted")
        
        elif command == "permission":
            self._agent.permission_gate_enabled = not self._agent.permission_gate_enabled
            self._agent._permission_allow_all = False  # Reset allow-all on toggle
            status = "on" if self._agent.permission_gate_enabled else "off"
            self._notify(f"Permission gate: {status}")
        
        elif command == "remember":
            self.remember_enabled = not self.remember_enabled
            status = "on" if self.remember_enabled else "off"
            # Persist the setting so it survives restarts
            self._settings_manager.set("memory_enabled", self.remember_enabled)
            self._notify(f"Session memory: {status}")
        
        elif command == "clear":
            self._agent.state.messages.clear()
            self._notify("Conversation history cleared.")
        
        elif command == "new":
            await self.new_session(args if args else None)
            self._notify("Started new session")

        elif command == "export":
            await self._export_session(args)
        
        elif command == "cost":
            agent = self._agent
            cost = agent.total_cost
            inp = agent.session_input_tokens
            out = agent.session_output_tokens
            total_tk = inp + out
            model = self.model
            model_name = model.display_name or model.model_id if model else "none"
            from config.pricing import format_cost
            info = (
                f"Session cost\n"
                f"  Model: {model_name}\n"
                f"  Input tokens: {inp:,}\n"
                f"  Output tokens: {out:,}\n"
                f"  Total tokens: {total_tk:,}\n"
                f"  Cost: {format_cost(cost)}"
            )
            self._notify(info)
        
        elif command == "sessions":
            # /sessions        → list all
            # /sessions <kw>   → search (non-digit = keyword, digit = resume from last list)
            if args and args.isdigit():
                await self._resume_session_by_index(int(args))
            else:
                await self._search_sessions(args if args else "")
        
        elif command == "login":
            await self._handle_login()

        elif command == "help" or command == "?":
            help_text = """
Available commands:
  /login           - Set API key for a provider
  /model [name]    - Switch or list models
  /thinking        - Toggle thinking (on/off or cycle level)
  /permission      - Toggle permission gate on/off
  /remember        - Toggle session memory on/off
  /session         - Show session info
  /compact         - Compact conversation history
  /clear           - Clear conversation history
  /new [name]      - Start new session
  /export [path]   - Export session to Markdown
  /cost            - Show session token usage and cost
  /sessions [kw]   - List or search past sessions
  /help            - Show this help
  /quit, /exit     - Exit
"""
            self._notify(help_text)

        elif command in ("quit", "exit"):
            self._notify("Goodbye!")
            raise SystemExit(0)

        else:
            # Check extension commands
            ext_commands = self._extension_runtime.get_commands()
            if command in ext_commands:
                await ext_commands[command]["handler"](args, None)
            else:
                # Check skill commands
                if command.startswith("skill:"):
                    skill_name = command[6:]
                    skills = self._resource_loader.get_skills()
                    for skill in skills:
                        if skill.name == skill_name:
                            await self.prompt(
                                f"Use the '{skill_name}' skill. "
                                f"Read {skill.file_path} for instructions.\n"
                                f"User request: {args}"
                            )
                            return
                    self._notify(f"Skill '{skill_name}' not found")
                else:
                    self._notify(f"Unknown command: /{command}. Type /help for available commands.")

    async def _search_sessions(self, keyword: str = "") -> None:
        """Search and list sessions interactively."""
        from sessions import SessionManager
        import os
        
        session_dir = self._settings_manager.get_session_dir()
        cwd = os.getcwd()
        
        results = SessionManager.search_sessions(session_dir, keyword, cwd)
        
        if not results:
            self._notify(f"No sessions found" + (f" matching '{keyword}'" if keyword else ""))
            return
        
        # Format results
        lines = [f"\nSessions" + (f" matching '{keyword}':" if keyword else ":")]
        for i, s in enumerate(results[:20]):
            desc = s.get("description", "")[:50]
            msgs = s.get("message_count", 0)
            created = s.get("created", "")[:16]
            lines.append(f"  [{i}] {created}  msgs:{msgs:>4}  {desc}")
        
        if len(results) > 20:
            lines.append(f"  ... and {len(results) - 20} more")
        lines.append(f"\n  Type /sessions <num> to resume, or /sessions <keyword> to filter")
        
        self._notify("\n".join(lines))
        
        # Store results for later resume by index
        self._last_session_results = results
    
    async def _resume_session_by_index(self, idx: int) -> None:
        """Resume a session from the last search results by index."""
        results = getattr(self, '_last_session_results', None)
        if not results:
            self._notify("No session list available. Type /sessions first.")
            return
        if idx < 0 or idx >= len(results):
            self._notify(f"Invalid index: {idx}. Choose 0-{len(results)-1}.")
            return
        target = results[idx]
        filepath = target["file"]
        self._notify(f"Opening session: {target['id']}")
        from sessions import SessionManager as SM
        source_mgr = SM.open(filepath)
        if source_mgr._active_path_ids:
            new_sm = source_mgr.fork(source_mgr._active_path_ids[-1])
            self._session_manager = new_sm
            self._agent.reset()
            for msg in new_sm.get_active_messages():
                self._agent.add_to_history(msg)
            self._notify(f"Resumed: {target.get('description', target['id'])[:60]}")
    
    def _notify(self, text: str) -> None:
        """Display a notification."""
        # In interactive mode, this would show in the UI
        print(f"\n[Agent] {text}")

    def _expand_prompt_templates(self, text: str) -> str:
        """Expand prompt template references."""
        if not text.startswith("/"):
            return text

        template_name = text[1:].split()[0]
        prompts = self._resource_loader.get_prompts()

        for prompt in prompts:
            if prompt.name == template_name:
                content = prompt.content
                args = text[len(template_name) + 2:] if len(text) > len(template_name) + 1 else ""

                def replace_placeholder(match):
                    key = match.group(1)
                    if key == "args":
                        return args
                    return f"[{key}]"

                content = _PLACEHOLDER_RE.sub(replace_placeholder, content)
                return content

        return text


# ── Convenience function ────────────────────────────────────────────────────

async def create_session(
    *,
    cwd: str = ".",
    agent_dir: str = "~/.agent",
    model: Optional[Model] = None,
    thinking_level: ThinkingLevel = ThinkingLevel.OFF,
    tools: Optional[List[str]] = None,
    no_session: bool = False,
    **kwargs,
) -> AgentSession:
    """Quick convenience function to create an AgentSession."""
    return await AgentSession.create(
        cwd=cwd,
        agent_dir=agent_dir,
        model=model,
        thinking_level=thinking_level,
        tools=tools,
        no_session=no_session,
        **kwargs,
    )
