"""
Test suite for Sugiri.
"""

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, '.')

from src.core.types import (
    AgentMessage,
    EventBus,
    Model,
    Settings,
    ThinkingLevel,
)
from src.core.agent import Agent
from src.providers import ProviderRegistry, AuthStorage
from src.sessions import SessionManager
from src.config import SettingsManager
from src.tools import (
    ReadTool,
    WriteTool,
    EditTool,
    BashTool,
    create_default_tools,
    create_readonly_tools,
)
from src.extensions import ExtensionRuntime, ResourceLoader


# ── Tool Tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bash_tool():
    """Test BashTool execution."""
    tool = BashTool(".")
    result = await tool.execute("test-1", {"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content[0]["text"]


@pytest.mark.asyncio
async def test_read_tool():
    """Test ReadTool file reading."""
    tool = ReadTool(".")
    result = await tool.execute("test-2", {"path": "README.md", "limit": 1})
    assert not result.is_error
    assert "Sugiri" in result.content[0]["text"]


@pytest.mark.asyncio
async def test_write_tool():
    """Test WriteTool file creation."""
    tool = WriteTool(tempfile.gettempdir())
    test_path = "agent_test_write.txt"
    result = await tool.execute("test-3", {"path": test_path, "content": "test content"})
    assert not result.is_error
    assert "Created" in result.content[0]["text"] or "Updated" in result.content[0]["text"]
    
    # Cleanup
    full_path = Path(tempfile.gettempdir()) / test_path
    if full_path.exists():
        full_path.unlink()


@pytest.mark.asyncio
async def test_edit_tool():
    """Test EditTool text replacement."""
    tmpdir = tempfile.gettempdir()
    test_file = Path(tmpdir) / "agent_test_edit.txt"
    test_file.write_text("original text here")
    
    tool = EditTool(tmpdir)
    result = await tool.execute("test-4", {
        "path": "agent_test_edit.txt",
        "edits": [{"oldText": "original text", "newText": "modified text"}],
    })
    assert not result.is_error
    
    # Verify
    content = test_file.read_text()
    assert "modified text" in content
    assert "original text" not in content
    
    test_file.unlink()


def test_tool_factory():
    """Test tool factory functions."""
    tools = create_default_tools()
    assert len(tools) == 4
    assert {t.name for t in tools} == {"read", "write", "edit", "bash"}
    
    readonly = create_readonly_tools()
    assert len(readonly) == 1
    assert readonly[0].name == "read"
    
    return "✅ Tool factories"


# ── Session Tests ────────────────────────────────────────────────────────────

def test_session_create():
    """Test session creation."""
    sm = SessionManager.in_memory()
    msg = AgentMessage(role="user", content="Hello")
    entry_id = sm.append_message(msg)
    assert entry_id is not None
    assert len(sm.get_all_entries()) == 1
    return "✅ Session creation"


def test_session_tree():
    """Test session tree structure."""
    sm = SessionManager.in_memory()
    
    msg1 = AgentMessage(role="user", content="Q1")
    id1 = sm.append_message(msg1)
    
    msg2 = AgentMessage(role="assistant", content="A1")
    id2 = sm.append_message(msg2, id1)
    
    msg3 = AgentMessage(role="user", content="Q2")
    id3 = sm.append_message(msg3, id2)
    
    assert len(sm.get_active_messages()) == 3
    assert sm._active_path_ids == [id1, id2, id3]
    
    # Branch
    sm.branch(id2)
    assert sm._active_path_ids == [id1, id2]
    
    return "✅ Session tree"


def test_session_branching():
    """Test session branching creates alternate paths."""
    sm = SessionManager.in_memory()
    
    msg1 = AgentMessage(role="user", content="Q1")
    id1 = sm.append_message(msg1)
    
    msg2 = AgentMessage(role="assistant", content="A1")
    id2 = sm.append_message(msg2, id1)
    
    # Branch back and add new message
    sm.branch(id1)
    msg3 = AgentMessage(role="user", content="Q2-alt")
    id3 = sm.append_message(msg3, id1)
    
    # Check tree
    entry1 = sm.get_entry(id1)
    assert len(entry1.children_ids) == 2  # Both id2 and id3
    
    return "✅ Session branching"


def test_session_fork():
    """Test session forking."""
    sm = SessionManager.in_memory()
    
    msg1 = AgentMessage(role="user", content="Q1")
    id1 = sm.append_message(msg1)
    
    msg2 = AgentMessage(role="assistant", content="A1")
    id2 = sm.append_message(msg2, id1)
    
    # Fork
    new_sm = sm.fork(id2, "fork-test")
    assert len(new_sm.get_all_entries()) == 2
    
    return "✅ Session fork"


def test_session_labels():
    """Test session labels."""
    sm = SessionManager.in_memory()
    
    msg1 = AgentMessage(role="user", content="Q1")
    id1 = sm.append_message(msg1)
    
    sm.set_label(id1, "start")
    assert sm.get_label(id1) == "start"
    
    return "✅ Session labels"


# ── Config Tests ─────────────────────────────────────────────────────────────

def test_settings_defaults():
    """Test default settings."""
    sm = SettingsManager.in_memory()
    settings = sm.get_settings()
    assert settings.default_provider == "anthropic"
    assert settings.theme == "dark"
    assert settings.compaction_enabled is True
    return "✅ Settings defaults"


def test_settings_override():
    """Test settings overrides."""
    sm = SettingsManager.in_memory({"theme": "light", "default_thinking": "high"})
    assert sm.get("theme") == "light"
    assert sm.get("default_thinking") == "high"
    
    settings = sm.get_settings()
    assert settings.default_thinking == ThinkingLevel.HIGH
    return "✅ Settings override"


# ── Provider Tests ───────────────────────────────────────────────────────────

def test_provider_registry():
    """Test provider registry model listing."""
    registry = ProviderRegistry()
    
    anthropic_models = registry.list_models("anthropic")
    assert len(anthropic_models) == 5
    assert anthropic_models[0].model_id == "claude-sonnet-4-5"
    
    openai_models = registry.list_models("openai")
    assert len(openai_models) > 0
    
    all_models = registry.list_models()
    assert len(all_models) >= 10  # 5 anthropic + 6 openai + 6 google + 4 deepseek
    
    # Find specific model
    model = registry.find_model("openai", "gpt-4o")
    assert model is not None
    assert model.provider == "openai"
    
    return "✅ Provider registry"


# ── Extension Tests ──────────────────────────────────────────────────────────

def test_extension_loading():
    """Test extension loading."""
    event_bus = EventBus()
    tools_registry = {}
    ext_runtime = ExtensionRuntime(event_bus, tools_registry)
    
    result = ext_runtime.load_extension_from_path("examples/extensions/hello.py")
    assert result is not None
    
    return "✅ Extension loading"


def test_extension_custom_tool():
    """Test extension custom tool registration."""
    event_bus = EventBus()
    tools_registry = {}
    ext_runtime = ExtensionRuntime(event_bus, tools_registry)
    
    result = ext_runtime.load_extension_from_path("examples/extensions/custom_tool.py")
    assert result is not None
    assert "weather" in tools_registry
    
    return "✅ Extension custom tools"


# ── Agent Tests ──────────────────────────────────────────────────────────────

def test_agent_state():
    """Test agent state management."""
    agent = Agent(
        system_prompt="You are a test assistant.",
        tools=create_default_tools(),
    )
    
    agent.state.model = Model("anthropic", "claude-sonnet-4-5")
    agent.state.thinking_level = ThinkingLevel.MEDIUM
    
    # Add messages
    agent.add_to_history(AgentMessage(role="user", content="Hello"))
    assert len(agent.state.messages) == 1
    
    # Steering/follow-up
    agent.steer("Do this")
    agent.follow_up("Also do that")
    
    assert len(agent._steering_queue) == 1
    assert len(agent._follow_up_queue) == 1
    
    return "✅ Agent state"


# ── Event Bus Tests ──────────────────────────────────────────────────────────

def test_event_bus():
    """Test event bus pub/sub."""
    bus = EventBus()
    events = []
    
    bus.on("test.event", lambda data: events.append(data))
    bus.emit("test.event", {"message": "hello"})
    
    assert len(events) == 1
    assert events[0]["message"] == "hello"
    
    return "✅ Event bus"


# ── Run All Tests ────────────────────────────────────────────────────────────

async def run_all():
    """Run all tests and report results."""
    tests = [
        # Async tests
        test_bash_tool,
        test_read_tool,
        test_write_tool,
        test_edit_tool,
        
        # Sync functions returned as results
    ]
    
    sync_results = [
        test_tool_factory,
        test_session_create,
        test_session_tree,
        test_session_branching,
        test_session_fork,
        test_session_labels,
        test_settings_defaults,
        test_settings_override,
        test_provider_registry,
        test_extension_loading,
        test_extension_custom_tool,
        test_agent_state,
        test_event_bus,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            results.append(f"❌ {test.__name__}: {e}")
    
    for test in sync_results:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            results.append(f"❌ {test.__name__}: {e}")
    
    for r in results:
        print(r)
    
    passed = sum(1 for r in results if r.startswith("✅"))
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all())
