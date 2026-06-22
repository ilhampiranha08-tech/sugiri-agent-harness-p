"""
Example: Permission gate extension.

Warns and confirms before executing dangerous bash commands.
Mirrors pi's confirm-destructive.ts example.
"""

from src.core.types import ExtensionAPI


DANGEROUS_PATTERNS = [
    "rm -rf",
    "rm -r",
    "sudo rm",
    "> /dev/",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "chmod 777",
    "chown -R",
]


def default(api: ExtensionAPI):
    """Permission gate that warns about dangerous commands."""
    
    def on_tool_call(event, ctx):
        if event.get("toolName") != "bash":
            return
        
        command = event.get("input", {}).get("command", "")
        
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command.lower():
                print(f"\n⚠️  [Permission Gate] Dangerous command detected!")
                print(f"   Command: {command}")
                print(f"   Pattern matched: {pattern}")
                
                # In interactive mode, this would prompt the user
                # For now, just warn
                return
        
        return None  # Allow the command
    
    api.on("tool_call", on_tool_call)
    print("[Permission Gate] Loaded!")
