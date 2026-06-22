#!/usr/bin/env python3
"""
Sugiri — AI Coding Agent, created by Ilham Sugiri.
Copyright (c) 2025 Ilham Sugiri. MIT License.

Usage:
    sugiri                           # Interactive chat
    sugiri -p "Summarize this code"  # Print mode (single-shot)
    sugiri -v                        # Show version
    sugiri install ./ext.py          # Install package
    sugiri list                      # List packages
    sugiri remove pkg-name           # Remove package
    sugiri config                    # Show settings
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click

from core.types import Model, ThinkingLevel
from providers import (
    get_auth_storage,
    get_provider_registry,
    ProviderRegistry,
)
from sessions import SessionManager
from extensions import ResourceLoader
from modes import InteractiveMode, PrintMode, JSONMode, RPCMode


# ── Helpers ──────────────────────────────────────────────────────────────

# Deprecated models with suggested replacements
DEPRECATED_MODELS = {}

def resolve_model(registry: ProviderRegistry, model_spec: str) -> Optional[Model]:
    if "/" in model_spec:
        provider_name, model_id = model_spec.split("/", 1)
        return registry.find_model(provider_name, model_id)
    # Exact match first
    for provider in ["anthropic", "openai", "google", "deepseek"]:
        for m in registry.list_models(provider):
            if m.model_id == model_spec:
                return m
    # Fallback: substring match (e.g. "sonnet" matches "claude-sonnet-4-5")
    for provider in ["anthropic", "openai", "google", "deepseek"]:
        for m in registry.list_models(provider):
            if model_spec in m.model_id:
                return m
    return None


def _validate_model(resolved_model: Model, model_spec: str, registry: ProviderRegistry) -> None:
    """Validate model and print warnings for deprecated or ambiguous models."""
    # Check deprecation
    if model_spec in DEPRECATED_MODELS:
        alternatives = DEPRECATED_MODELS[model_spec]
        alt_text = " or ".join(alternatives)
        print(f"\n⚠️  Model '{model_spec}' is deprecated (will be removed July 2026).")
        print(f"   Consider switching to: {alt_text}")
        # Show cost comparison
        from config.pricing import PRICING
        if model_spec in PRICING.get(resolved_model.provider, {}):
            old_price = PRICING[resolved_model.provider][model_spec]
            print(f"   Current price: ${old_price['input']}/M in, ${old_price['output']}/M out")
        print()
    
    # Warn if model was matched via substring (ambiguous)
    if resolved_model.model_id != model_spec:
        if "/" not in model_spec:
            print(f"ℹ️  '{model_spec}' matched to '{resolved_model.model_id}' ({resolved_model.provider})")


def _warn_fallback(resolved_model: Optional[Model], model_spec: str, 
                   registry: ProviderRegistry, provider: Optional[str] = None) -> None:
    """Warn when falling back to a default model."""
    if resolved_model is None:
        if provider:
            available = [m.model_id for m in registry.list_models(provider)]
            print(f"\n❌ No model matching '{model_spec}' found for provider '{provider}'.")
            if available:
                print(f"   Available: {', '.join(available)}")
        else:
            all_models = registry.list_models()
            available = [f"{m.provider}/{m.model_id}" for m in all_models[:10]]
            print(f"\n❌ Model '{model_spec}' not found.")
            if available:
                print(f"   Try: {', '.join(available)}")
            if len(all_models) > 10:
                print(f"   ... and {len(all_models) - 10} more")


def get_agent_dir() -> str:
    return os.environ.get("AGENT_DIR", os.path.expanduser("~/.agent"))


async def _run(
    initial_message: Optional[str],
    mode: str,
    provider: Optional[str],
    model_spec: Optional[str],
    api_key: Optional[str],
    thinking_level: Optional[str],
    continue_session: bool,
    resume: bool,
    session_path: Optional[str],
    fork_session: Optional[str],
    no_session: bool,
    session_name: Optional[str],
    tools_spec: Optional[str],
    extension_paths: List[str],
    no_extensions: bool,
    no_skills: bool,
    no_context_files: bool,
    system_prompt: Optional[str],
    theme: Optional[str],
    verbose: bool = False,
) -> None:
    from core.session import AgentSession

    agent_dir = get_agent_dir()
    cwd = os.getcwd()

    auth_storage = get_auth_storage(agent_dir)
    if api_key:
        auth_storage.set_runtime_api_key(provider or "anthropic", api_key)

    registry = get_provider_registry(auth_storage)

    resolved_model: Optional[Model] = None
    if model_spec:
        resolved_model = resolve_model(registry, model_spec)
        if resolved_model is None:
            _warn_fallback(resolved_model, model_spec, registry, provider)
            # Fall back to default
            from config import load_last_model
            last = load_last_model(agent_dir)
            resolved_model = registry.find_model(last.get("provider", ""), last.get("model_id", "")) if last else None
            if resolved_model is None:
                resolved_model = registry.list_models()[0] if registry.list_models() else None
            if resolved_model:
                print(f"   Using: {resolved_model.display_name or resolved_model.model_id}\n")
            else:
                print(f"   No models available. Please check your configuration.\n")
                sys.exit(1)
        else:
            _validate_model(resolved_model, model_spec, registry)
    elif provider:
        models = registry.list_models(provider)
        if models:
            resolved_model = models[0]
    else:
        from config import load_last_model
        last = load_last_model(agent_dir)
        if last:
            resolved_model = registry.find_model(last.get("provider", ""), last.get("model_id", ""))
            if resolved_model:
                if verbose:
                    print(f"Restored last model: {resolved_model.display_name or resolved_model.model_id}")
                # Validate even restored models
                last_model_id = last.get("model_id", "")
                if last_model_id in DEPRECATED_MODELS:
                    alternatives = DEPRECATED_MODELS[last_model_id]
                    print(f"\n⚠️  Your saved model '{last_model_id}' is deprecated.")
                    print(f"   Please switch with /model to: {' or '.join(alternatives)}\n")

    level = ThinkingLevel.OFF
    if thinking_level:
        try:
            level = ThinkingLevel(thinking_level)
        except ValueError:
            pass

    tools_list = None
    if tools_spec:
        tools_list = [t.strip() for t in tools_spec.split(",")]

    session_mgr: Optional[SessionManager] = None
    if no_session:
        session_mgr = SessionManager.in_memory()
    elif session_path:
        session_mgr = SessionManager.open(session_path)
    elif continue_session:
        session_dir = os.path.join(agent_dir, "sessions")
        session_mgr = SessionManager.continue_recent(cwd, session_dir)
        if session_mgr is None:
            print("No previous session found. Starting new session.")
            session_mgr = SessionManager(os.path.join(agent_dir, "sessions"), cwd)
            session_mgr.create_new(session_name)
    elif resume:
        session_dir = os.path.join(agent_dir, "sessions")
        sessions = SessionManager.list_sessions(session_dir, cwd)
        if not sessions:
            print("No sessions found.")
            sys.exit(1)
        print("Available sessions:")
        for i, s in enumerate(sessions):
            print(f"  [{i}] {s['id']} - {s['created']} ({s['message_count']} msgs)")
        choice = input("Select session (number): ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(sessions):
                session_mgr = SessionManager.open(sessions[idx]["file"])
            else:
                print("Invalid selection."); sys.exit(1)
        except ValueError:
            print("Invalid input."); sys.exit(1)

    if fork_session:
        source_mgr = SessionManager.open(fork_session)
        session_mgr = source_mgr.fork(source_mgr._active_path_ids[-1] if source_mgr._active_path_ids else None)

    loader = ResourceLoader(cwd, agent_dir)
    loader.reload()

    session = await AgentSession.create(
        cwd=cwd, agent_dir=agent_dir, model=resolved_model, thinking_level=level,
        tools=tools_list, session_manager=session_mgr, resource_loader=loader,
        disable_extensions=no_extensions, disable_skills=no_skills,
        disable_context_files=no_context_files, system_prompt_override=system_prompt,
        no_session=no_session, session_name=session_name,
    )

    if extension_paths:
        for ext_path in extension_paths:
            session._extension_runtime.load_extension_from_path(ext_path)

    try:
        if mode == "interactive":
            await InteractiveMode(session, initial_message).run()
        elif mode == "print":
            if not initial_message:
                print("Error: No prompt provided. Use -p 'message' or pipe via stdin.")
                sys.exit(1)
            await PrintMode(session, initial_message).run()
        elif mode == "json":
            if not initial_message:
                print("Error: No prompt provided for JSON mode.")
                sys.exit(1)
            await JSONMode(session, initial_message).run()
        elif mode == "rpc":
            await RPCMode(session).run()
    finally:
        await session.dispose()


# ── Main CLI Group ───────────────────────────────────────────────────────

@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("-m", "--message", "prompt_text", default=None, help="Prompt message (for -p/print mode)")
@click.option("-p", "--print", "print_mode", is_flag=True, help="Print response and exit (non-interactive)")
@click.option("--mode", type=click.Choice(["interactive", "print", "json", "rpc"]), default="interactive")
@click.option("--provider", default=None, help="Provider (anthropic, openai, google, deepseek)")
@click.option("--model", default=None, help="Model ID (supports provider/model format)")
@click.option("--api-key", default=None, help="API key")
@click.option("--thinking", type=click.Choice(["off", "minimal", "low", "medium", "high", "xhigh"]), default=None)
@click.option("-c", "--continue", "continue_session", is_flag=True, help="Continue most recent session")
@click.option("-r", "--resume", is_flag=True, help="Browse and select session to resume")
@click.option("--session", default=None, help="Use specific session file or ID")
@click.option("--fork", "fork_session", default=None, help="Fork specific session")
@click.option("--no-session", is_flag=True, help="Ephemeral mode")
@click.option("-n", "--name", default=None, help="Session display name")
@click.option("-t", "--tools", default=None, help="Comma-separated tool allowlist")
@click.option("-e", "--extension", multiple=True, help="Load extension from path")
@click.option("--no-extensions", is_flag=True)
@click.option("--skill", multiple=True)
@click.option("--no-skills", is_flag=True)
@click.option("--prompt-template", multiple=True)
@click.option("--no-context-files", "-nc", is_flag=True)
@click.option("--system-prompt", default=None)
@click.option("--theme", default=None)
@click.option("-v", "--version", is_flag=True, help="Show version")
@click.option("--verbose", is_flag=True)
@click.option("--offline", is_flag=True)
@click.option("-a", "--approve", is_flag=True)
@click.pass_context
def cli(
    ctx: click.Context,
    prompt_text: Optional[str],
    print_mode: bool,
    mode: str,
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    thinking: Optional[str],
    continue_session: bool,
    resume: bool,
    session: Optional[str],
    fork_session: Optional[str],
    no_session: bool,
    name: Optional[str],
    tools: Optional[str],
    extension: tuple,
    no_extensions: bool,
    skill: tuple,
    no_skills: bool,
    prompt_template: tuple,
    no_context_files: bool,
    system_prompt: Optional[str],
    theme: Optional[str],
    version: bool,
    verbose: bool,
    offline: bool,
    approve: bool,
):
    """Sugiri — AI Coding Agent, created by Ilham Sugiri.

    Run without subcommand to start interactive chat.
    Use -p with -m "message" for single-shot print mode.
    Pipe stdin: echo "prompt" | sugiri -p

    \b
    Package commands:
      sugiri install/remove/list/update/config
    """
    if ctx.invoked_subcommand is not None:
        return

    if version:
        print("Sugiri v1.2.2")
        return

    # Determine final mode
    final_mode = mode
    if print_mode:
        final_mode = "print"

    # Collect initial message: --message flag > ctx.args (positional) > stdin pipe
    initial_message = prompt_text or (" ".join(ctx.args) if ctx.args else None)
    if final_mode in ("print", "json") and not initial_message:
        if not sys.stdin.isatty():
            initial_message = sys.stdin.read().strip() or None

    asyncio.run(_run(
        initial_message=initial_message,
        mode=final_mode,
        provider=provider, model_spec=model, api_key=api_key,
        thinking_level=thinking,
        continue_session=continue_session, resume=resume,
        session_path=session, fork_session=fork_session,
        no_session=no_session, session_name=name,
        tools_spec=tools, extension_paths=list(extension),
        no_extensions=no_extensions, no_skills=no_skills,
        no_context_files=no_context_files, system_prompt=system_prompt,
        theme=theme, verbose=verbose,
    ))


# ── Package Commands ─────────────────────────────────────────────────────

@cli.command()
@click.argument("source")
@click.option("-l", "--local", is_flag=True, help="Install to project .agent/")
def install(source: str, local: bool):
    """Install a package from path, git, or npm."""
    from packages import get_package_manager
    pm = get_package_manager(get_agent_dir(), os.getcwd())
    if not pm.install(source, local=local):
        sys.exit(1)


@cli.command()
@click.argument("name")
def remove(name: str):
    """Remove an installed package by name."""
    from packages import get_package_manager
    pm = get_package_manager(get_agent_dir(), os.getcwd())
    if not pm.remove(name):
        sys.exit(1)


@cli.command(name="list")
def list_packages():
    """List installed packages."""
    from packages import get_package_manager
    pm = get_package_manager(get_agent_dir(), os.getcwd())
    packages = pm.list()
    if packages:
        print(f"\n{'Package':<30} {'Type':<12} {'Location':<10} {'Source'}")
        print("-" * 80)
        for pkg in packages:
            print(f"{pkg['name']:<30} {pkg['type']:<12} {pkg['location']:<10} {pkg.get('source', '')[:30]}")
        print(f"\n{len(packages)} package(s) installed.")
    else:
        print("No packages installed.")


@cli.command()
@click.argument("name", required=False)
@click.option("--all", "update_all", is_flag=True, help="Update all packages")
def update(name: Optional[str], update_all: bool):
    """Update installed packages."""
    from packages import get_package_manager
    pm = get_package_manager(get_agent_dir(), os.getcwd())
    if not pm.update(name=name, update_all=update_all):
        sys.exit(1)


@cli.command()
def config():
    """Show current agent settings."""
    agent_dir = get_agent_dir()
    settings_file = os.path.join(agent_dir, "settings.json")
    if os.path.exists(settings_file):
        with open(settings_file) as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print("No settings file found. Default settings are in effect.")


if __name__ == "__main__":
    cli()
