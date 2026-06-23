"""
Interactive terminal UI components for Sugiri.

Provides:
- TerminalSelector: Interactive list selector with arrow keys, typing filter, Enter/Esc
- Used by /model, session picker, and other interactive commands
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# termios/tty only available on Unix (Linux/macOS)
try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

# Windows: gunakan msvcrt untuk raw input
if sys.platform == 'win32':
    import msvcrt
else:
    msvcrt = None


class TerminalSelector:
    """Interactive terminal list selector.

    Renders a list and lets the user navigate with arrow keys,
    filter by typing, and select with Enter. Cancel with Esc.

    Usage:
        selector = TerminalSelector(
            items=["Item 1", "Item 2", "Item 3"],
            display=lambda item: item,
        )
        result = selector.run()
        # result is the selected item, or None if cancelled
    """

    def __init__(
        self,
        items: List[Any],
        *,
        title: str = "Select",
        display: Callable[[Any], str] = str,
        search_key: Optional[Callable[[Any], str]] = None,
        max_height: int = 15,
        prompt: str = "> ",
        empty_message: str = "No items found",
    ):
        self._items = items
        self._title = title
        self._display = display
        self._search_key = search_key or display
        self._max_height = max_height
        self._prompt = prompt
        self._empty_message = empty_message

        self._filtered: List[int] = list(range(len(items)))
        self._selected_idx: int = 0
        self._filter_text: str = ""
        self._scroll_offset: int = 0
        self._last_draw_lines: int = 0  # Track previous render height

    def _reset(self):
        self._filtered = list(range(len(self._items)))
        self._selected_idx = 0
        self._filter_text = ""
        self._scroll_offset = 0
        self._last_draw_lines = 0

    def _apply_filter(self):
        """Filter items based on typed text."""
        if not self._filter_text:
            self._filtered = list(range(len(self._items)))
        else:
            lower = self._filter_text.lower()
            self._filtered = [
                i for i in range(len(self._items))
                if lower in self._search_key(self._items[i]).lower()
            ]

        self._selected_idx = 0
        self._scroll_offset = 0

    def _visible_count(self) -> int:
        """How many items fit on screen."""
        header_height = 4  # title + border + filter prompt + border
        footer_height = 2  # border + hint
        available = self._max_height - header_height - footer_height
        return max(1, min(available, len(self._filtered)))

    def _render(self) -> str:
        """Render the selector UI as a string."""
        lines = []
        width = 60

        # Top border
        lines.append("┌" + "─" * (width - 2) + "┐")
        lines.append("│ " + self._title.center(width - 4) + " │")
        lines.append("├" + "─" * (width - 2) + "┤")

        # Filter input line
        filter_line = f"│ {self._prompt}{self._filter_text}"
        filter_line += " " * (width - len(filter_line) - 1)
        filter_line += "│"
        lines.append(filter_line)
        lines.append("├" + "─" * (width - 2) + "┤")

        # Item list
        visible = self._visible_count()

        if not self._filtered:
            lines.append("│ " + self._empty_message.center(width - 4) + " │")
            for _ in range(visible):
                lines.append("│" + " " * (width - 2) + "│")
        else:
            # Adjust scroll
            if self._selected_idx < self._scroll_offset:
                self._scroll_offset = self._selected_idx
            if self._selected_idx >= self._scroll_offset + visible:
                self._scroll_offset = self._selected_idx - visible + 1

            for i in range(visible):
                item_idx = self._scroll_offset + i
                if item_idx >= len(self._filtered):
                    lines.append("│" + " " * (width - 2) + "│")
                    continue

                real_idx = self._filtered[item_idx]
                display_text = self._display(self._items[real_idx])

                if item_idx == self._selected_idx:
                    # Highlighted
                    line = f"│ ▸ {display_text}"
                    line += " " * (width - len(line) - 1)
                    line += "│"
                    # Use ANSI reverse video for highlight
                    line = f"\033[7m{line}\033[0m"
                else:
                    line = f"│   {display_text}"
                    line += " " * (width - len(line) - 1)
                    line += "│"

                lines.append(line)

        # Bottom
        lines.append("├" + "─" * (width - 2) + "┤")
        lines.append("│ " + "↑↓/jk Navigate  Type to filter  Enter=Select  Esc/q=Cancel".center(width - 4) + " │")
        lines.append("└" + "─" * (width - 2) + "┘")

        return "\n".join(lines)

    def run(self) -> Optional[Any]:
        """Run the selector interactively. Returns selected item or None."""
        self._reset()
        self._draw()

        while True:
            ch = self._read_char()

            if ch is None:
                continue

            if ch == '\x1b':  # Escape sequence
                seq = self._read_escape()

                if seq == '':  # Plain Esc
                    self._clear()
                    return None
                elif seq.endswith('A'):  # Up arrow
                    if self._filtered:
                        self._selected_idx = max(0, self._selected_idx - 1)
                elif seq.endswith('B'):  # Down arrow
                    if self._filtered:
                        self._selected_idx = min(
                            len(self._filtered) - 1,
                            self._selected_idx + 1
                        )
                elif seq in ('[5~', '[H'):  # Page Up / Home
                    self._selected_idx = 0
                elif seq in ('[6~', '[F'):  # Page Down / End
                    self._selected_idx = len(self._filtered) - 1 if self._filtered else 0
                elif seq.startswith('[') and not (seq[-1].isalpha() or seq[-1] == '~'):
                    pass  # Partial CSI - ignore

                self._draw()

            elif ch in ('\r', '\n'):  # Enter
                self._clear()
                if self._filtered:
                    return self._items[self._filtered[self._selected_idx]]
                return None

            elif ch in ('\x7f', '\x08'):  # Backspace
                if self._filter_text:
                    self._filter_text = self._filter_text[:-1]
                    self._apply_filter()
                    self._draw()

            elif ch == '\x03':  # Ctrl+C
                self._clear()
                sys.stdout.write('\n')
                raise KeyboardInterrupt

            elif not self._filter_text and ch in ('j', 'J'):  # Vim: down
                if self._filtered:
                    self._selected_idx = min(len(self._filtered) - 1, self._selected_idx + 1)
                self._draw()

            elif not self._filter_text and ch in ('k', 'K'):  # Vim: up
                if self._filtered:
                    self._selected_idx = max(0, self._selected_idx - 1)
                self._draw()

            elif not self._filter_text and ch in ('q', 'Q'):  # Quit
                self._clear()
                return None

            elif ch.isprintable():
                self._filter_text += ch
                self._apply_filter()
                self._draw()

    def _read_char(self) -> Optional[str]:
        """Read a single character from stdin. Cross-platform."""
        try:
            if msvcrt is not None:  # Windows
                ch = msvcrt.getwch()
                if ch == '\x00' or ch == '\xe0':  # Extended key prefix
                    return None  # Handled via escape sequence
                return ch
            else:
                ch = sys.stdin.read(1)
                return ch if ch else None
        except Exception:
            return None

    def _read_escape(self) -> str:
        """Read escape sequence. Returns '' for plain Esc."""
        # Try to read follow-up bytes with short timeout
        import os as _os
        
        if msvcrt is not None:  # Windows
            import msvcrt as _msvcrt
            result = ''
            # On Windows, arrow keys send sequences like \xe0H, \xe0P, etc.
            # In modern terminals, standard ANSI sequences are used.
            # Read available bytes without blocking
            while _msvcrt.kbhit():
                try:
                    ch = _msvcrt.getwch()
                    result += ch
                    if len(result) > 0 and result[-1].isalpha() or result[-1] == '~':
                        break
                    if len(result) >= 10:
                        break
                except Exception:
                    break
            # Normalize: if result starts with [ or O, return as-is
            if result and result[0] in ('[', 'O'):
                return result
            return ''
        
        # Unix: non-blocking I/O
        fd = sys.stdin.fileno()
        old_flags = None
        try:
            import fcntl
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | _os.O_NONBLOCK)
        except (ImportError, OSError, AttributeError):
            pass
        
        result = ''
        try:
            for _ in range(12):
                try:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    result += ch
                    if result and result[0] in ('[', 'O'):
                        if len(result) > 1 and (ch.isalpha() or ch == '~'):
                            break
                    else:
                        break
                except (BlockingIOError, TypeError, OSError):
                    break
        finally:
            if old_flags is not None:
                try:
                    import fcntl
                    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
                except Exception:
                    pass
        
        if not result:
            return ''
        if result[0] not in ('[', 'O'):
            return ''
        return result

    def _draw(self):
        """Clear and redraw the selector at the same position.
        After drawing, positions cursor at the filter input line."""
        new_lines = self._render().count('\n') + 1

        # If we've drawn before, move up to overwrite the old render
        if self._last_draw_lines > 0:
            # Move cursor up to start of previous render
            sys.stdout.write(f'\033[{self._last_draw_lines}A')

        # Clear from cursor to end of screen
        sys.stdout.write('\033[J')

        # Draw the new content
        sys.stdout.write(self._render())
        sys.stdout.write('\n')

        # Move cursor up to the filter input line
        # Filter line is at row 4 (after title + 2 borders)
        # Cursor is now 1 line below render, so go up: new_lines - 4 + 1
        filter_row_from_top = 4
        lines_to_go_up = new_lines - filter_row_from_top + 1
        sys.stdout.write(f'\033[{lines_to_go_up}A')

        # Move right past the filter prompt prefix + typed text
        prefix_len = len(f"│ {self._prompt}")
        cursor_col = prefix_len + len(self._filter_text)
        sys.stdout.write(f'\033[{cursor_col}C')

        sys.stdout.flush()

        # Remember line count for next redraw
        self._last_draw_lines = new_lines

    def _clear(self):
        """Clear the selector from screen."""
        # Cursor is at the filter input line (row 4 from top of render)
        # Move up to the very top, go to column 0, then clear all below
        if self._last_draw_lines > 0:
            filter_row_from_top = 4
            sys.stdout.write(f'\033[{filter_row_from_top - 1}A')  # Go to top
            sys.stdout.write('\r')   # Go to column 0
            sys.stdout.write('\033[J')  # Clear from top to end of screen
        sys.stdout.flush()
        self._last_draw_lines = 0


# ── Convenience functions ──────────────────────────────────────────────────

def select_model(models: List[Any]) -> Optional[Any]:
    """Show a model selector and return the chosen model.

    Args:
        models: List of Model objects
    """
    if not models:
        print("No models available.")
        return None

    selector = TerminalSelector(
        items=models,
        title="Select Model",
        display=lambda m: f"{m.display_name or m.model_id:<30} {m.provider}",
        search_key=lambda m: f"{m.model_id} {m.display_name or ''} {m.provider}",
        prompt="Filter: ",
    )

    return selector.run()


def select_session(sessions: List[Dict]) -> Optional[Dict]:
    """Show a session picker and return the chosen session.

    Args:
        sessions: List of session info dicts (from SessionManager.list_sessions)
    """
    if not sessions:
        print("No sessions found.")
        return None

    selector = TerminalSelector(
        items=sessions,
        title="Select Session",
        display=lambda s: f"{s['id']:<24} {s.get('created', '')[:16]}  msgs:{s.get('message_count', 0)}",
        search_key=lambda s: s['id'],
        prompt="Filter: ",
    )

    return selector.run()


def select_thinking(current: str, levels: List[str]) -> Optional[str]:
    """Show a thinking level selector and return the chosen level.

    Args:
        current: Current thinking level value (e.g. "off", "high")
        levels: List of available thinking level strings
    """
    if not levels:
        print("Thinking not available for this model.")
        return None

    # Build items with current marker
    items = []
    for lvl in levels:
        marker = " ◀ current" if lvl == current else ""
        items.append({"value": lvl, "marker": marker})

    selector = TerminalSelector(
        items=items,
        title="Select Thinking Level",
        display=lambda item: f"{item['value']:<12}{item['marker']}",
        search_key=lambda item: item['value'],
        prompt="Level: ",
    )

    chosen = selector.run()
    return chosen["value"] if chosen else None


def select_file(cwd: str) -> Optional[str]:
    """Show a file/directory picker for @ mentions.

    Args:
        cwd: Current working directory to scan

    Returns:
        Relative path of selected file, or None if cancelled
    """
    import os
    from pathlib import Path

    base = Path(cwd).resolve()

    def scan_dir(directory: Path) -> List[Dict]:
        """Scan a directory and return items sorted: dirs first, then files."""
        items = []
        try:
            for entry in sorted(directory.iterdir()):
                if entry.name.startswith('.'):
                    continue  # skip hidden
                is_dir = entry.is_dir()
                items.append({
                    "name": entry.name,
                    "path": str(entry.relative_to(base)),
                    "is_dir": is_dir,
                })
        except PermissionError:
            pass
        # Dirs first, then files, both alphabetical
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return items

    items = scan_dir(base)
    if not items:
        print("No files found.")
        return None

    selector = TerminalSelector(
        items=items,
        title=f"Select File - {base.name or '/'}",
        display=lambda item: f"{'📁' if item['is_dir'] else '📄'} {item['name']}",
        search_key=lambda item: item["name"],
        prompt="File: ",
        max_height=18,
    )

    chosen = selector.run()
    return chosen["path"] if chosen else None
