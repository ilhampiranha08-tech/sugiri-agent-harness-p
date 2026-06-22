"""
Terminal UI for interactive agent mode.

Built with Textual, provides:
- Chat message display
- Input editor
- Tool call visualization
- Status bar
- Command handling

Mirrors pi's interactive mode TUI.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box

from core.types import (
    AgentEvent,
    AgentMessage,
    EventType,
    ThinkingLevel,
)
from core.agent import Agent
from core.session import AgentSession


class AgentTUI:
    """Rich-based Terminal UI for interactive mode.
    
    Provides a chat-like interface with:
    - Message history display
    - Input editor with history, line editing, paste support
    - Status line with model/thinking/token info
    - Tool call visualization
    """
    
    def __init__(self, session: AgentSession):
        self._session = session
        self._console = Console()
        self._running = False
        self._line_buffer: List[str] = []
        self._last_assistant_text = ""
        # Command history
        self._history: List[str] = []
        self._history_cursor: int = -1
        self._history_saved_line: str = ""
        self._old_termios = None  # Unix only
        # Command suggestions state
        self._suggestions_visible = False
        self._suggestions_lines = 0
        self._suggestion_selected = 0
        self._suggestion_matches: List[tuple] = []
        self._suggestion_navigated = False
        self._cached_commands: Optional[List[tuple]] = None
        # Loading state
        self._loading = False
        self._spinner_task = None
        self._spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._spinner_idx = 0
        self._spinner_clear_on_exit = True
    
    # ── Cross-platform raw input helpers ──────────────────────────
    
    @staticmethod
    def _getch() -> str:
        """Read a single character from stdin. Cross-platform."""
        if sys.platform == 'win32':
            import msvcrt
            return msvcrt.getwch()
        else:
            return sys.stdin.read(1)
    
    def _enable_raw_mode(self) -> None:
        """Enable raw terminal mode for character-by-character input."""
        if sys.platform != 'win32':
            import termios, tty
            self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
    
    def _disable_raw_mode(self) -> None:
        """Restore terminal settings."""
        if sys.platform != 'win32' and self._old_termios is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None
    
    @staticmethod
    def _kbhit() -> bool:
        """Check if a keypress is available. Cross-platform."""
        if sys.platform == 'win32':
            import msvcrt
            return msvcrt.kbhit()
        else:
            import select
            return select.select([sys.stdin], [], [], 0)[0] != []
    
    async def _spinner_loop(self):
        """Animate a loading spinner while agent is working."""
        import sys
        while self._loading:
            ch = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
            self._spinner_idx += 1
            sys.stdout.write(f'\r  {ch} Thinking...')
            sys.stdout.flush()
            await asyncio.sleep(0.12)
        # Only clear if _stop_loading hasn't already cleared synchronously
        if self._spinner_clear_on_exit:
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
    
    async def _bash_progress_loop(self, tool_name: str):
        """Show elapsed time + spinner for long-running bash commands."""
        import sys, time
        start = time.time()
        while self._loading:
            elapsed = int(time.time() - start)
            ch = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
            self._spinner_idx += 1
            sys.stdout.write(f'\r  {ch} {tool_name} ({elapsed}s)...')
            sys.stdout.flush()
            await asyncio.sleep(0.2)
    
    def _start_loading(self, tool_name: str = ""):
        """Start the loading spinner. If tool_name given, shows elapsed time."""
        if not self._loading:
            self._loading = True
            self._spinner_clear_on_exit = True
            self._spinner_idx = 0
            if tool_name:
                self._spinner_task = asyncio.ensure_future(self._bash_progress_loop(tool_name))
            else:
                self._spinner_task = asyncio.ensure_future(self._spinner_loop())
    
    def _stop_loading(self):
        """Stop the loading spinner and clear its line immediately."""
        if self._loading:
            self._loading = False
            self._spinner_clear_on_exit = False  # Prevent async spinner from clearing again
            # Clear spinner line synchronously — don't wait for async task
            import sys
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
    
    def _get_commands(self) -> List[tuple]:
        """Get all available slash commands with descriptions. Cached after first call."""
        if self._cached_commands is not None:
            return self._cached_commands
        
        commands = [
            ("/model", "Switch or list AI models"),
            ("/thinking", "Toggle thinking level"),
            ("/permission", "Toggle permission gate on/off"),
            ("/remember", "Toggle session memory on/off"),
            ("/session", "Show session info"),
            ("/cost", "Show token usage and session cost"),
            ("/sessions", "List or search past sessions"),
            ("/compact", "Compact conversation history"),
            ("/clear", "Clear conversation history"),
            ("/new", "Start new session"),
            ("/export", "Export session to Markdown"),
            ("/login", "Set API key for a provider"),
            ("/help", "Show all commands"),
            ("/quit", "Exit Sugiri"),
            ("/exit", "Exit Sugiri"),
        ]
        # Add extension commands
        try:
            ext_commands = self._session._extension_runtime.get_commands()
            for name, info in ext_commands.items():
                commands.append((f"/{name}", info.get("description", "")))
        except Exception:
            pass
        self._cached_commands = commands
        return commands
    
    async def _permission_prompt(self, tool_name: str, params: dict) -> str:
        """Show permission confirmation with arrow-selectable options.
        Returns True (allow), False (deny), or 'allow_all'."""
        import sys
        
        # Build short description
        cmd = params.get("command", params.get("path", ""))
        if tool_name == "bash":
            desc = f"bash: {cmd[:80]}"
        elif tool_name == "write":
            desc = f"write: {cmd}"
        elif tool_name == "edit":
            desc = f"edit: {cmd}"
        else:
            desc = f"{tool_name}: {str(params)[:80]}"
        
        options = [("Y", "Yes, allow"), ("N", "No, deny"), ("A", "Allow all session")]
        selected = 0
        _lines = 7  # Fixed height of permission prompt area
        _drawn = False
        
        def draw():
            nonlocal _drawn
            if _drawn:
                sys.stdout.write(f'\033[{_lines}A')
            sys.stdout.write('\r\033[J')
            sys.stdout.write(f'  \033[33m⚠\033[0m  \033[33m{desc}\033[0m\n\n')
            for i, (key, label) in enumerate(options):
                if i == selected:
                    sys.stdout.write(f'  \033[7m {key}  {label} \033[0m\n')
                else:
                    sys.stdout.write(f'  \033[2m[{key}] {label}\033[0m\n')
            sys.stdout.write(f'\n  \033[2m↑↓ choose  Enter confirm  y/n/a quick\033[0m')
            sys.stdout.flush()
            _drawn = True
        
        self._enable_raw_mode()
        try:
            draw()
            
            while True:
                ch = self._getch()
                
                if ch == '\x1b':
                    # Read full escape sequence
                    seq = ch
                    for _ in range(6):
                        if not self._kbhit():
                            break
                        c = self._getch()
                        seq += c
                        if c.isalpha() or c == '~':
                            break
                    
                    if seq in ('\x1b[A', '\x1bOA'):  # Up
                        selected = (selected - 1) % len(options)
                        draw()
                    elif seq in ('\x1b[B', '\x1bOB'):  # Down
                        selected = (selected + 1) % len(options)
                        draw()
                    elif len(seq) == 1:  # Esc
                        sys.stdout.write('\n  Cancelled\n')
                        sys.stdout.flush()
                        return False
                
                elif ch in ('\r', '\n'):
                    key = options[selected][0]
                    if key == 'Y': return True
                    elif key == 'N': return False
                    elif key == 'A': return "allow_all"
                
                elif ch.lower() == 'y':
                    return True
                elif ch.lower() == 'n':
                    return False
                elif ch.lower() == 'a':
                    return "allow_all"
                
                elif ch == '\x03':  # Ctrl+C
                    return False
        finally:
            self._disable_raw_mode()
    
    async def run(self, initial_message: Optional[str] = None) -> None:
        """Run the interactive TUI."""
        self._running = True
        
        # Set terminal title
        self._set_terminal_title()
        
        # First-run wizard if no API key configured
        if self._needs_wizard():
            await self._run_wizard()
        
        # Subscribe to agent events
        self._session.subscribe(self._on_event)
        
        # Wire up permission gate callback for interactive confirmation
        self._session.agent._permission_callback = self._permission_prompt
        
        # Print header
        self._print_header()
        
        if initial_message:
            async for _ in self._session.prompt_stream(initial_message):
                pass  # Events handled by _on_event
        
        # Main input loop
        while self._running:
            try:
                user_input = await self._get_input()
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self._session.prompt(user_input)
                elif user_input.startswith("!"):
                    # Bash command passthrough
                    await self._session.prompt(
                        f"Run this command and show me the output: {user_input[1:]}"
                    )
                else:
                    async for _ in self._session.prompt_stream(user_input):
                        pass  # Events handled by _on_event
            
            except KeyboardInterrupt:
                self._console.print("\n[yellow]Use /quit to exit, Esc or Ctrl+C again to force[/]")
                try:
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    self._running = False
                    break
            except SystemExit:
                self._running = False
                break
            except Exception as e:
                self._console.print(f"[red]Error: {e}[/]")
    
    def _set_terminal_title(self):
        """Set terminal window title to Sugiri - model (session)."""
        import sys
        model = self._session.model
        model_name = model.display_name or model.model_id if model else "no-model"
        session_id = self._session.session_id[:8] if self._session.session_id else "?"
        title = f"Sugiri - {model_name} ({session_id})"
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    
    def _needs_wizard(self) -> bool:
        """Check if first-run wizard should be shown."""
        import os, json
        auth_file = os.path.join(
            os.path.expanduser(self._session._settings_manager.get_agent_dir()),
            "auth.json"
        )
        if os.path.exists(auth_file):
            try:
                with open(auth_file) as f:
                    if json.load(f):
                        return False
            except Exception:
                pass
        return True
    
    async def _run_wizard(self) -> None:
        """First-run setup wizard: provider → API key → model."""
        import os, sys, asyncio
        from providers import BUILTIN_MODELS
        from ui.selector import TerminalSelector
        
        self._console.print()
        self._console.print("[bold cyan]╔══════════════════════════════════════════════╗[/]")
        self._console.print("[bold cyan]║[/]   [bold]Welcome to Sugiri v1.2.2![/]              [bold cyan]║[/]")
        self._console.print("[bold cyan]║[/]   Let's set up your AI provider.         [bold cyan]║[/]")
        self._console.print("[bold cyan]╚══════════════════════════════════════════════╝[/]")
        self._console.print()
        
        # Step 1: Choose provider
        providers = [
            {"id": "anthropic", "name": "Anthropic (Claude)"},
            {"id": "openai", "name": "OpenAI (GPT)"},
            {"id": "google", "name": "Google (Gemini)"},
            {"id": "deepseek", "name": "DeepSeek (V4)"},
        ]
        
        selector = TerminalSelector(
            items=providers,
            title="Select AI Provider",
            display=lambda p: p["name"],
            search_key=lambda p: p["name"],
            prompt="Provider: ",
        )
        
        loop = asyncio.get_event_loop()
        chosen = await loop.run_in_executor(None, selector.run)
        
        if chosen is None:
            self._console.print("\n[yellow]Setup cancelled. You can run /login later.[/]")
            self._console.print()
            return
        
        provider = chosen["id"]
        self._console.print(f"\n[green]Provider: {chosen['name']}[/]")
        
        # Step 2: Enter API key
        self._console.print()
        self._console.print(f"Enter your [cyan]{chosen['name']}[/] API key:")
        self._console.print("[dim](input will be hidden)[/]")
        
        api_key = await self._session._read_secret("API key: ")
        
        if not api_key or not api_key.strip():
            self._console.print("\n[yellow]Setup cancelled - empty key.[/]")
            self._console.print()
            return
        
        # Save key
        self._session._auth_storage.save_auth(provider, api_key.strip())
        self._session._auth_storage.set_runtime_api_key(provider, api_key.strip())
        self._console.print("[green]API key saved![/]")
        
        # Step 3: Choose default model
        models = [m for m in BUILTIN_MODELS.get(provider, [])]
        if models:
            self._console.print()
            selector2 = TerminalSelector(
                items=models,
                title=f"Select Default Model ({chosen['name']})",
                display=lambda m: f"{m.display_name or m.model_id}",
                search_key=lambda m: m.model_id,
                prompt="Model: ",
            )
            chosen_model = await loop.run_in_executor(None, selector2.run)
            
            if chosen_model:
                await self._session.set_model(chosen_model)
                self._console.print(f"\n[green]Model: {chosen_model.display_name or chosen_model.model_id}[/]")
        
        self._console.print()
        self._console.print("[bold green]Setup complete![/] You're ready to code.")
        self._console.print("[dim]Type /help for commands, /model to switch models.[/]")
        self._console.print()
    
    def _print_header(self) -> None:
        """Print the startup header."""
        self._console.print()
        self._console.print(
            Panel(
                Text("Sugiri - Coding Agent TUI", style="bold cyan"),
                box=box.ROUNDED,
            )
        )
        
        # Model info
        model = self._session.model
        self._console.print(
            f"Model: [green]{model.display_name or model.model_id}[/] "
            f"(Provider: {model.provider})"
        )
        
        # Show thinking status for all models that support it
        if model.supports_thinking:
            provider = model.provider
            level = self._session.thinking_level.value
            
            if provider == "deepseek":
                effort = "max" if level == "xhigh" else level
                style = "yellow" if level != "off" else "dim"
                self._console.print(f"Thinking: [{style}]{effort}[/]")
            else:
                style = "yellow" if level != "off" else "dim"
                self._console.print(f"Thinking: [{style}]{level}[/]")
        
        self._console.print(f"Session: [dim]{self._session.session_id}[/]")
        
        # Context files info
        context_files = self._session._resource_loader.get_context_files()
        if context_files:
            paths = [cf["path"] for cf in context_files]
            self._console.print(f"Context files: [dim]{', '.join(paths[:3])}[/]")
        
        # Skills info
        skills = self._session._resource_loader.get_skills()
        if skills:
            skill_names = [s.name for s in skills[:5]]
            self._console.print(f"Skills: [dim]{', '.join(skill_names)}[/]")
        
        self._console.print()
        self._console.print("[dim]Type /login for login AI Provider[/]")
        self._console.print("[dim]Type /model for change AI Model[/]")
        self._console.print("[dim]Type /help for commands, Esc/Ctrl+C to quit[/]")
        self._console.print("─" * self._console.width)
        self._console.print()
    
    def _on_event(self, event: AgentEvent) -> None:
        """Handle agent events for display."""
        etype = event.type
        
        if etype == EventType.MESSAGE_UPDATE:
            data = event.data
            if "text" in data:
                self._stop_loading()
                self._console.print(data["text"], end="")
        
        elif etype == EventType.MESSAGE_END:
            self._stop_loading()
            msg = event.data.get("message")
            if msg:
                self._print_message(msg)
        
        elif etype == EventType.TOOL_EXECUTION_START:
            self._stop_loading()
            tool_name = event.data.get("tool_name", "unknown")
            self._console.print(f"\n[dim]🔧 Running [cyan]{tool_name}[/]...[/]")
            self._start_loading(tool_name)  # Shows elapsed time for bash commands
        
        elif etype == EventType.TOOL_EXECUTION_END:
            self._stop_loading()
            result = event.data.get("result")
            is_error = event.data.get("is_error", False)
            tool_name = event.data.get("tool_name", "")
            
            if result and result.content:
                for block in result.content:
                    text = block.get("text", "")
                    if is_error:
                        self._console.print(f"[red]{text}[/]")
                    elif tool_name == "edit":
                        # Render diff with colors
                        self._print_diff(text)
                    else:
                        if len(text) > 500:
                            text = text[:500] + "\n... (truncated)"
                        self._console.print(f"[dim]{text}[/]")
        
        elif etype == EventType.AGENT_START:
            self._console.print()
            self._start_loading()
        
        elif etype == EventType.TURN_START:
            self._start_loading()
        
        elif etype == EventType.TURN_END:
            self._stop_loading()
        
        elif etype == EventType.AGENT_END:
            self._stop_loading()
            self._console.print("[dim]Done.[/]")
            self._console.print()
    
    def _print_message(self, msg: AgentMessage) -> None:
        """Print a message in chat format."""
        if msg.role == "user":
            self._console.print()
            self._console.print(
                Panel(
                    Markdown(msg.content if isinstance(msg.content, str) else str(msg.content)),
                    title="You",
                    title_align="left",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )
        
        elif msg.role == "assistant":
            content = msg.content
            if isinstance(content, str):
                self._console.print(Markdown(content))
            else:
                self._console.print(str(content))
        
        elif msg.role == "tool":
            self._console.print(
                f"[dim italic]Tool result ({msg.name}): {str(msg.content)[:200]}[/]"
            )
    
    def _print_diff(self, text: str):
        """Print tool output with diff highlighting."""
        lines = text.split('\n')
        # Print summary line (first line)
        if lines:
            self._console.print(f"[dim]{lines[0]}[/]")
        # Print diff with colors
        for line in lines[1:]:
            if line.startswith('+++') or line.startswith('---'):
                self._console.print(f"[bold dim]{line}[/]")
            elif line.startswith('@@'):
                self._console.print(f"[cyan]{line}[/]")
            elif line.startswith('+'):
                self._console.print(f"[green]{line}[/]")
            elif line.startswith('-'):
                self._console.print(f"[red]{line}[/]")
            else:
                self._console.print(f"[dim]{line}[/]")
    
    async def _get_input(self) -> str:
        """Get input with keyboard shortcuts support.
        
        Supports:
        - Ctrl+L: Open model selector
        - Ctrl+C: Clear input / Exit
        - Esc: Cancel / Abort
        """
        import sys
        
        # Use raw writes exclusively — mixing Rich breaks cursor tracking
        # Two newlines to cleanly separate from any previous output
        sys.stdout.write('\n\n')
        self._suggestions_visible = False
        self._suggestions_lines = 0
        self._suggestion_selected = 0
        self._suggestion_matches = []
        self._suggestion_navigated = False
        
        model = self._session.model
        if model:
            parts = [f'\033[42;30m {model.model_id} \033[0m']
            
            provider = model.provider
            level = self._session.thinking_level.value
            if model.supports_thinking:
                if provider == "deepseek" and model.model_id.endswith("-reasoner"):
                    parts.append('\033[43;30m reasoning \033[0m')
                if provider == "deepseek":
                    # DeepSeek V4: off / high / max
                    if level == "off":
                        parts.append('\033[2m thinking:off \033[0m')
                    else:
                        effort = "max" if level == "xhigh" else "high"
                        parts.append(f'\033[43;30m thinking:{effort} \033[0m')
                else:
                    # Anthropic, OpenAI, Google
                    if level == "off":
                        parts.append('\033[2m thinking:off \033[0m')
                    else:
                        parts.append(f'\033[43;30m thinking:{level} \033[0m')
            
            parts.append(f'\033[2m msgs:{len(self._session.messages)} \033[0m')
            tokens = self._session.agent.token_count
            if tokens > 0:
                parts.append(f'\033[2m tk:{tokens//1000}k \033[0m')
            # Cost display — always visible, dim when $0
            cost = self._session.agent.total_cost
            from config.pricing import format_cost
            if cost > 0:
                parts.append(f' \033[33m{format_cost(cost)}\033[0m')
            else:
                parts.append(f' \033[2m$0.00\033[0m')
            parts.append('\033[2m Ctrl+L=model \033[0m')
            
            sys.stdout.write(''.join(parts) + '\n')
        
        sys.stdout.write('> ')
        sys.stdout.flush()
        
        try:
            result = await self._raw_input()
            return result
        except KeyboardInterrupt:
            raise
    
    async def _raw_input(self) -> str:
        """Read input in raw terminal mode with full line editing.
        
        Features:
        - Command history (Up/Down arrows)
        - Cursor movement (Left/Right, Home/End, Ctrl+A/E)
        - Word navigation (Alt+B/F, Ctrl+Left/Right)
        - Delete (Del key, Ctrl+D at cursor)
        - Kill line (Ctrl+K), Kill word backward (Ctrl+W)
        - Paste support (bracket paste mode + multiline)
        - Multiline input (backslash continuation)
        - @ file picker
        - Ctrl+L model selector
        """
        import sys
        import os
        
        buffer = ""
        lines: List[str] = []
        cursor_pos = 0
        in_paste = False
        paste_buffer = ""
        
        # Clear any leftover suggestion artifacts
        self._suggestions_visible = False
        self._suggestions_lines = 0
        
        saved_history_pos = self._history_cursor
        
        def clamp_cursor():
            nonlocal cursor_pos
            cursor_pos = max(0, min(cursor_pos, len(buffer)))
        
        def word_left(pos: int) -> int:
            """Move cursor to start of current/previous word."""
            # Skip trailing whitespace
            while pos > 0 and buffer[pos - 1].isspace():
                pos -= 1
            # Skip word characters
            while pos > 0 and not buffer[pos - 1].isspace():
                pos -= 1
            return pos
        
        def word_right(pos: int) -> int:
            """Move cursor to end of current/next word."""
            # Skip current word
            while pos < len(buffer) and not buffer[pos].isspace():
                pos += 1
            # Skip whitespace
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            return pos
        
        def add_to_history(text: str):
            """Add a non-empty, non-duplicate line to history."""
            t = text.strip()
            if t and (not self._history or self._history[-1] != t):
                self._history.append(t)
                if len(self._history) > 1000:
                    self._history.pop(0)
            self._history_cursor = -1
        
        def history_up():
            nonlocal cursor_pos
            if not self._history:
                return
            if self._history_cursor == -1:
                self._history_saved_line = buffer
                self._history_cursor = len(self._history) - 1
            elif self._history_cursor > 0:
                self._history_cursor -= 1
            buffer_new = self._history[self._history_cursor]
            return buffer_new
        
        def history_down():
            nonlocal cursor_pos
            if self._history_cursor == -1:
                return buffer
            self._history_cursor += 1
            if self._history_cursor >= len(self._history):
                self._history_cursor = -1
                return self._history_saved_line
            return self._history[self._history_cursor]
        
        def apply_history(entry: str):
            nonlocal buffer, cursor_pos
            buffer = entry
            cursor_pos = len(buffer)
        
        def insert_text(text: str):
            nonlocal buffer, cursor_pos
            buffer = buffer[:cursor_pos] + text + buffer[cursor_pos:]
            cursor_pos += len(text)
        
        try:
            self._enable_raw_mode()
            
            while self._running:
                # Redraw input line
                self._redraw_input_line(buffer, cursor_pos, len(lines))
                # Show command suggestions if typing /
                self._render_suggestions(buffer)
                
                # Read next character
                ch = await asyncio.get_event_loop().run_in_executor(None, self._getch)
                
                if not ch:
                    self._running = False
                    return ""
                
                # ── Bracket paste mode ───────────────────────────────
                if in_paste:
                    if ch == '\x1b':
                        # Might be paste end marker \e[201~
                        if self._kbhit():
                            seq = ch
                            while True:
                                c = self._getch()
                                if not c:
                                    break
                                seq += c
                                if c == '~' or c.isalpha():
                                    break
                            if seq == '\x1b[201~':
                                in_paste = False
                                # Insert accumulated paste
                                insert_text(paste_buffer)
                                paste_buffer = ""
                            else:
                                paste_buffer += seq
                        else:
                            paste_buffer += ch
                    elif ch == '\r':
                        # Enter during paste - add newline to paste buffer
                        paste_buffer += '\n'
                    elif ch.isprintable() or ch in ('\t',):
                        paste_buffer += ch
                    else:
                        paste_buffer += ch
                    continue
                
                # ── Enter ───────────────────────────────────────────
                if ch in ('\r', '\n'):
                    if buffer.endswith('\\'):
                        # Multiline continuation
                        buffer = buffer[:-1]
                        lines.append(buffer)
                        buffer = ""
                        cursor_pos = 0
                        sys.stdout.write('\n│ ')
                        sys.stdout.flush()
                    else:
                        # Auto-complete if buffer is shorter than selected suggestion
                        if self._suggestions_visible and self._suggestion_matches:
                            cmd = self._suggestion_matches[self._suggestion_selected][0]
                            if buffer != cmd:
                                buffer = cmd + ' '
                                cursor_pos = len(buffer)
                                self._clear_suggestions()
                                continue
                        
                        lines.append(buffer)
                        full_input = '\n'.join(lines)
                        if full_input.strip():
                            add_to_history(full_input)
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        return full_input
                
                # ── Ctrl+C: Clear / Exit ────────────────────────────
                elif ch == '\x03':
                    if buffer or lines:
                        buffer = ""
                        cursor_pos = 0
                        lines = []
                        self._history_cursor = -1
                        self._clear_input_line()
                        sys.stdout.write('\n\033[33mInput cleared. Ctrl+C again to quit.\033[0m\n> ')
                        sys.stdout.flush()
                    else:
                        raise KeyboardInterrupt
                
                # ── Tab: Auto-complete command / path ──────────
                elif ch == '\t':
                    if buffer.startswith('/') and ' ' not in buffer:
                        # Command completion: find matching commands
                        cmds = self._get_commands()
                        prefix = buffer[1:]  # strip leading /
                        matches = [c[0] for c in cmds if c[0].startswith(buffer)]
                        if len(matches) == 1:
                            # Exact match: complete and add space
                            buffer = matches[0] + ' '
                            cursor_pos = len(buffer)
                        elif len(matches) > 1:
                            # Multiple matches: complete to longest common prefix
                            lcp = buffer
                            for i in range(len(buffer), min(len(m[0]) for m in matches) + 1):
                                chars = {m[0][i:i+1] for m in matches if i < len(m[0])}
                                if len(chars) == 1:
                                    lcp += list(chars)[0]
                                else:
                                    break
                            buffer = lcp
                            cursor_pos = len(buffer)
                
                # ── Backspace ────────────────────────────────────────
                elif ch in ('\x7f', '\x08'):
                    if cursor_pos > 0:
                        buffer = buffer[:cursor_pos - 1] + buffer[cursor_pos:]
                        cursor_pos -= 1
                        if self._suggestions_visible:
                            self._suggestion_selected = 0
                            self._suggestion_navigated = False
                
                # ── Escape sequences ────────────────────────────────
                elif ch == '\x1b':
                    # Read the rest of the escape sequence
                    seq = ch
                    try:
                        # Read up to 12 more bytes (max CSI sequence length)
                        for _ in range(12):
                            if not self._kbhit():
                                break
                            try:
                                c = self._getch()
                                if not c:
                                    break
                                seq += c
                                if c.isalpha() or c == '~':
                                    break
                            except Exception:
                                break
                    except Exception:
                        pass
                    if len(seq) == 1:
                        # Plain Esc - no follow-up bytes
                        if self._session.is_streaming:
                            await self._session.abort()
                            sys.stdout.write('\n\033[33mAborted.\033[0m\n')
                            sys.stdout.flush()
                        # Clear suggestions
                        self._clear_suggestions()
                        if buffer:
                            buffer = ""
                            cursor_pos = 0
                            lines = []
                            self._history_cursor = -1
                            # Don't print new prompt — let _redraw_input_line handle it
                        else:
                            raise KeyboardInterrupt
                        continue
                    
                    # ── Arrow keys ───────────────────────────────
                    if seq in ('\x1b[A', '\x1bOA'):    # Up (ANSI + application mode)
                        if self._suggestions_visible and self._suggestion_matches:
                            self._suggestion_navigated = True
                            # Navigate suggestion list up (wrap around)
                            if self._suggestion_selected > 0:
                                self._suggestion_selected -= 1
                            else:
                                self._suggestion_selected = len(self._suggestion_matches) - 1
                        else:
                            entry = history_up()
                            if entry is not None:
                                apply_history(entry)
                    elif seq in ('\x1b[B', '\x1bOB'):  # Down (ANSI + application mode)
                        if self._suggestions_visible and self._suggestion_matches:
                            self._suggestion_navigated = True
                            # Navigate suggestion list down (wrap around)
                            max_idx = len(self._suggestion_matches) - 1
                            if self._suggestion_selected < max_idx:
                                self._suggestion_selected += 1
                            else:
                                self._suggestion_selected = 0
                        else:
                            entry = history_down()
                            if entry is not None:
                                apply_history(entry)
                    elif seq == '\x1b[C':    # Right
                        cursor_pos = min(len(buffer), cursor_pos + 1)
                    elif seq == '\x1b[D':    # Left
                        cursor_pos = max(0, cursor_pos - 1)
                    
                    # ── Home/End ─────────────────────────────────
                    elif seq in ('\x1b[H', '\x1b[1~'):   # Home
                        cursor_pos = 0
                    elif seq in ('\x1b[F', '\x1b[4~'):   # End
                        cursor_pos = len(buffer)
                    elif seq == '\x1bOH':    # Home (tmux)
                        cursor_pos = 0
                    elif seq == '\x1bOF':    # End (tmux)
                        cursor_pos = len(buffer)
                    
                    # ── Delete ──────────────────────────────────
                    elif seq == '\x1b[3~':   # Delete
                        if cursor_pos < len(buffer):
                            buffer = buffer[:cursor_pos] + buffer[cursor_pos + 1:]
                    
                    # ── Ctrl+Arrow (word navigation) ────────────
                    elif seq in ('\x1b[1;5C', '\x1b[1;2C', '\x1bOC'):  # Ctrl+Right / Shift+Right
                        cursor_pos = word_right(cursor_pos)
                    elif seq in ('\x1b[1;5D', '\x1b[1;2D', '\x1bOD'):  # Ctrl+Left / Shift+Left
                        cursor_pos = word_left(cursor_pos)
                    
                    # ── Alt+B/F (word navigation, some terminals) ──
                    elif seq in ('\x1bb', '\x1bB'):   # Alt+B / Alt+Shift+B
                        cursor_pos = word_left(cursor_pos)
                    elif seq in ('\x1bf', '\x1bF'):   # Alt+F / Alt+Shift+F
                        cursor_pos = word_right(cursor_pos)
                    
                    # ── Bracket paste start ────────────────────
                    elif seq == '\x1b[200~':
                        in_paste = True
                        paste_buffer = ""
                    
                    # ── Other escape: ignore ────────────────────
                    # (e.g., mouse events, focus events, etc.)
                
                # ── Ctrl+L: Model selector ──────────────────────────
                elif ch == '\x0c':
                    self._clear_input_line()
                    
                    from .selector import select_model
                    models = self._session._provider_registry.list_models()
                    models = sorted(models, key=lambda m: (m.provider, m.model_id))
                    
                    loop = asyncio.get_event_loop()
                    self._disable_raw_mode()
                    
                    chosen = await loop.run_in_executor(None, select_model, models)
                    
                    self._enable_raw_mode()
                    
                    if chosen:
                        await self._session.set_model(chosen)
                        sys.stdout.write(f'\n  Switched to {chosen.display_name or chosen.model_id}\n')
                        self._set_terminal_title()
                    
                    self._reset_prompt(buffer, cursor_pos)
                
                # ── Ctrl+A: Home ────────────────────────────────────
                elif ch == '\x01':
                    cursor_pos = 0
                
                # ── Ctrl+E: End ─────────────────────────────────────
                elif ch == '\x05':
                    cursor_pos = len(buffer)
                
                # ── Ctrl+K: Kill to end of line ─────────────────────
                elif ch == '\x0b':
                    buffer = buffer[:cursor_pos]
                
                # ── Ctrl+W: Kill word backward ──────────────────────
                elif ch == '\x17':
                    old_pos = cursor_pos
                    cursor_pos = word_left(cursor_pos)
                    buffer = buffer[:cursor_pos] + buffer[old_pos:]
                
                # ── Ctrl+U: Kill to start of line ───────────────────
                elif ch == '\x15':
                    buffer = buffer[cursor_pos:]
                    cursor_pos = 0
                
                # ── Ctrl+D: Delete at cursor (or EOF if empty) ──────
                elif ch == '\x04':
                    if buffer:
                        if cursor_pos < len(buffer):
                            buffer = buffer[:cursor_pos] + buffer[cursor_pos + 1:]
                    elif not lines:
                        # EOF on empty line: exit
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        raise KeyboardInterrupt
                
                # ── @ file picker ────────────────────────────────────
                elif ch == '@':
                    import os as _os
                    self._disable_raw_mode()

                    from .selector import select_file
                    loop = asyncio.get_event_loop()
                    chosen = await loop.run_in_executor(None, select_file, _os.getcwd())

                    self._enable_raw_mode()

                    if chosen:
                        insert_text(chosen)

                    self._reset_prompt(buffer, cursor_pos)
                
                # ── Printable characters ────────────────────────────
                elif ch.isprintable():
                    insert_text(ch)
                    if self._suggestions_visible:
                        self._suggestion_selected = 0
                        self._suggestion_navigated = False  # Reset on new input
        
        finally:
            self._disable_raw_mode()
        
        return ""
    
    def _redraw_input_line(self, buffer: str, cursor: int, extra_lines: int):
        """Redraw the input line with proper cursor positioning.
        Clears everything from the prompt line downward."""
        import sys
        
        prompt = "> "
        display = prompt + buffer
        
        # Clear current line and everything below it, then draw prompt
        sys.stdout.write('\r\033[J')
        sys.stdout.write(display)
        
        if cursor < len(buffer):
            sys.stdout.write(f'\033[{len(buffer) - cursor}D')
        sys.stdout.flush()
    
    def _render_suggestions(self, buffer: str):
        """Render command suggestions below the input line when typing /.
        Supports arrow key navigation and Enter to select.
        Note: _redraw_input_line clears everything with \\033[J, so we just draw."""
        import sys
        
        # Only show suggestions when buffer starts with / and no space yet
        if not buffer.startswith('/') or ' ' in buffer:
            self._suggestions_visible = False
            self._suggestion_selected = 0
            self._suggestion_matches = []
            return
        
        # Get matching commands (prefix match on command name)
        cmds = self._get_commands()
        prefix = buffer.lower()
        self._suggestion_matches = [(name, desc) for name, desc in cmds if name.lower().startswith(prefix)]
        
        if not self._suggestion_matches:
            self._suggestions_visible = False
            self._suggestion_selected = 0
            self._suggestion_navigated = False
            return
        
        # Clamp selection
        if self._suggestion_selected >= len(self._suggestion_matches):
            self._suggestion_selected = 0
        
        # Limit to ~8 suggestions
        display_matches = self._suggestion_matches[:8]
        max_name_len = max(len(name) for name, _ in display_matches)
        
        # Draw suggestions starting from next line
        sys.stdout.write('\n')
        
        lines_rendered = 0
        for i, (name, desc) in enumerate(display_matches):
            if i == self._suggestion_selected:
                line = f"  \033[7m {name:<{max_name_len+1}} {desc[:50]}\033[0m"
            else:
                line = f"  \033[2m{name:<{max_name_len+2}}\033[0m {desc[:50]}"
            sys.stdout.write(line + '\n')
            lines_rendered += 1
        
        if len(self._suggestion_matches) > 8:
            sys.stdout.write(f"  \033[2m... and {len(self._suggestion_matches) - 8} more\033[0m\n")
            lines_rendered += 1
        
        # Move cursor back to prompt line
        sys.stdout.write(f'\033[{lines_rendered + 1}A\r')
        # Reposition cursor to end of prompt
        sys.stdout.write(f'\033[{len(buffer) + 2}C')
        sys.stdout.flush()
        
        self._suggestions_visible = True
        self._suggestions_lines = lines_rendered
    
    def _clear_suggestions(self):
        """Clear suggestion state (visual clearing handled by _redraw_input_line)."""
        self._suggestions_visible = False
        self._suggestions_lines = 0
        self._suggestion_selected = 0
        self._suggestion_matches = []
        self._suggestion_navigated = False
    
    def _clear_input_line(self):
        """Clear the current input line."""
        import sys
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
    
    def _reset_prompt(self, buffer: str, cursor_pos: int) -> None:
        """Reset terminal to a clean prompt line after external UI (selectors, pickers)."""
        import sys
        # Clear any suggestion artifacts
        if self._suggestions_visible:
            sys.stdout.write('\033[J')
            self._suggestions_visible = False
            self._suggestions_lines = 0
        # Print prompt on a fresh line
        sys.stdout.write('\n> ')
        sys.stdout.write(buffer)
        sys.stdout.flush()


# ── Simpler console-based UI ────────────────────────────────────────────────

class ConsoleUI:
    """Simpler console-based UI for non-interactive display."""
    
    def __init__(self, session: AgentSession):
        self._session = session
        self._console = Console()
    
    async def run_once(self, message: str) -> None:
        """Run a single prompt and display results."""
        self._session.subscribe(self._on_event)
        
        self._console.print(f"[dim]Model: {self._session.model.model_id}[/]")
        
        await self._session.prompt(message)
    
    def _on_event(self, event: AgentEvent) -> None:
        """Handle events for display."""
        if event.type == EventType.MESSAGE_UPDATE:
            data = event.data
            if "text" in data:
                self._console.print(data["text"], end="", highlight=False)
        
        elif event.type == EventType.TOOL_EXECUTION_START:
            self._console.print(
                f"\n[dim]Running tool: {event.data.get('tool_name')}...[/]"
            )
        
        elif event.type == EventType.TOOL_EXECUTION_END:
            result = event.data.get("result")
            if result and result.content:
                text = result.content[0].get("text", "")
                if len(text) > 300:
                    text = text[:300] + "..."
                self._console.print(f"[dim]{text}[/]")
        
        elif event.type == EventType.AGENT_END:
            self._console.print()
