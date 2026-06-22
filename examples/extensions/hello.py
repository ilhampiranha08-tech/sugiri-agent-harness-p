"""
Example: Hello World extension.

Place in ~/.agent/extensions/ to auto-discover.
"""

from src.core.types import ExtensionAPI


def default(api: ExtensionAPI):
    """Simple extension that prints hello on session start."""
    
    def on_session_start(event, ctx):
        print(f"\n[Hello Extension] Agent session started!")
    
    def on_agent_start(event, ctx):
        print(f"\n[Hello Extension] Agent is starting...")
    
    api.on("session_start", on_session_start)
    api.on("agent_start", on_agent_start)
    
    print("[Hello Extension] Loaded!")
