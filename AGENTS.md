# Sugiri

Sugiri is an AI coding agent created by Ilham Sugiri.

## Memory

> Update terakhir: 22 Juni 2026

- Versi: v1.2.2
- 21 model LLM (Anthropic, OpenAI, Google, DeepSeek)
- Baru: Permission gate (/permission), arrow suggestions (/ + ↑↓), export session (/export)
- Bug fixed: kursor, spinner, escape sequence, provider auth error, session save O(n²)
- Thinking mode: adaptive Opus 4.6+, V4 effort (high/max), GPT-5.x max
- Installer: standalone (.sh .bat .py), 7 linux package manager
- File penting: ringkasan-percakapan.md (riwayat diskusi lengkap)

## Project Structure

- `src/core/` - Core types, agent loop, session management
- `src/providers/` - LLM provider implementations (Anthropic, OpenAI, Google)
- `src/tools/` - Built-in tools (read, write, edit, bash)
- `src/sessions/` - JSONL tree-structured session storage
- `src/extensions/` - Extension system and resource loader
- `src/config/` - Settings and configuration management
- `src/ui/` - Terminal UI components
- `src/modes/` - Run modes (interactive, print, JSON, RPC)
- `cli.py` - Main CLI entry point

## Development Commands

```bash
# Install
pip install -e .

# Run interactively
python cli.py

# Run in print mode
python cli.py -p "What files are here?"

# With specific model
python cli.py --model anthropic/claude-sonnet-4-5 "Help me refactor"

# Load an extension
python cli.py -e examples/extensions/hello.py
```

## Conventions

- Python 3.10+ with type hints
- Async/await for all I/O operations
- Follow existing patterns in the codebase
- Tools return ToolCallResult with content and details
- Sessions use JSONL format with tree structure (id, parentId)
