"""
Provider system - abstraction over multiple LLM backends.

Supports: Anthropic, OpenAI, Google Gemini, and custom providers.
Mirrors pi's provider architecture.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from core.types import (
    AgentMessage,
    AgentTool,
    Model,
    Provider as ProviderInterface,
    ThinkingLevel,
)


class ProviderAuthError(Exception):
    """Raised when an API key is not configured for a provider."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"API key not set for '{provider}'. Run /login to configure.")


# ── Provider Registry ────────────────────────────────────────────────────────

BUILTIN_MODELS: Dict[str, List[Model]] = {
    "anthropic": [
        Model("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5", 200000, 8192, True, True, True),
        Model("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6", 200000, 8192, True, True, True),
        Model("anthropic", "claude-opus-4-5", "Claude Opus 4.5", 200000, 8192, True, True, True),
        Model("anthropic", "claude-opus-4-8", "Claude Opus 4.8", 200000, 8192, True, True, True),
        Model("anthropic", "claude-haiku-4-5", "Claude Haiku 4.5", 200000, 8192, True, True, False),
    ],
    "openai": [
        Model("openai", "gpt-4o", "GPT-4o", 128000, 16384, True, True, False),
        Model("openai", "gpt-4o-mini", "GPT-4o Mini", 128000, 16384, True, True, False),
        Model("openai", "gpt-5.4", "GPT 5.4", 128000, 16384, True, True, True),
        Model("openai", "gpt-5.4-mini", "GPT 5.4 Mini", 128000, 16384, True, True, True),
        Model("openai", "gpt-5.5", "GPT 5.5", 128000, 16384, True, True, True),
        Model("openai", "o4-mini", "o4 Mini", 200000, 100000, True, True, True),
    ],
    "google": [
        Model("google", "gemini-2.5-pro", "Gemini 2.5 Pro", 200000, 8192, True, True, True),
        Model("google", "gemini-2.5-flash", "Gemini 2.5 Flash", 1048576, 8192, True, True, False),
        Model("google", "gemini-3-flash", "Gemini 3 Flash", 1048576, 8192, True, True, True),
        Model("google", "gemini-3-pro-preview", "Gemini 3 Pro Preview", 200000, 8192, True, True, True),
        Model("google", "gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", 200000, 8192, True, True, True),
    ],
    "deepseek": [
        Model("deepseek", "deepseek-v4-flash", "DeepSeek V4 Flash", 131072, 8192, True, False, True),
        Model("deepseek", "deepseek-v4-pro", "DeepSeek V4 Pro", 131072, 8192, True, False, True),
    ],
}

BUILTIN_PROVIDERS: Dict[str, type] = {}


class ProviderRegistry:
    """Registry of available providers and models."""
    
    def __init__(self, auth_storage: Optional["AuthStorage"] = None):
        self._providers: Dict[str, ProviderInterface] = {}
        # Deep-copy to prevent shared mutation across instances
        self._models: Dict[str, List[Model]] = {
            k: [Model(m.provider, m.model_id, m.display_name, m.context_window, m.max_output_tokens, m.supports_tools, m.supports_images, m.supports_thinking) for m in v]
            for k, v in BUILTIN_MODELS.items()
        }
        self._custom_providers: Dict[str, type] = {}
        self._auth_storage = auth_storage
        # Shared HTTP client for connection reuse (lazy init)
        self._shared_http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_shared_client(self) -> httpx.AsyncClient:
        """Get or create a shared HTTP client for connection pooling."""
        if self._shared_http_client is None:
            self._shared_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 min for long-thinking models
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._shared_http_client
    
    def register_provider(self, name: str, provider_cls: type) -> None:
        """Register a custom provider class."""
        self._custom_providers[name] = provider_cls
    
    async def get_provider(self, name: str, api_key: Optional[str] = None) -> ProviderInterface:
        """Get or create a provider instance."""
        if name in self._providers:
            return self._providers[name]
        
        # Resolve API key: explicit > auth storage > env var
        if api_key is None and self._auth_storage:
            api_key = self._auth_storage.get_api_key(name)
        
        if name == "anthropic":
            from .anthropic import AnthropicProvider
            provider = AnthropicProvider(api_key=api_key, shared_client=await self._get_shared_client())
        elif name == "openai":
            from .openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=api_key, shared_client=await self._get_shared_client())
        elif name == "google":
            from .google_provider import GoogleProvider
            provider = GoogleProvider(api_key=api_key, shared_client=await self._get_shared_client())
        elif name == "deepseek":
            from .deepseek_provider import DeepSeekProvider
            provider = DeepSeekProvider(api_key=api_key, shared_client=await self._get_shared_client())
        elif name in self._custom_providers:
            provider = self._custom_providers[name](api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {name}")
        
        self._providers[name] = provider
        return provider
    
    def find_model(self, provider: str, model_id: str) -> Optional[Model]:
        """Find a model by provider and ID."""
        models = self._models.get(provider, [])
        for m in models:
            if m.model_id == model_id:
                return m
        return None
    
    def list_models(self, provider: Optional[str] = None) -> List[Model]:
        """List available models, optionally filtered by provider."""
        if provider:
            return self._models.get(provider, [])
        return [m for models in self._models.values() for m in models]
    
    def add_custom_models(self, provider: str, models: List[Model]) -> None:
        """Add custom models for a provider."""
        if provider not in self._models:
            self._models[provider] = []
        self._models[provider].extend(models)
    
    async def close_all(self) -> None:
        """Close all provider clients and release resources."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        self._providers.clear()
        if self._shared_http_client is not None:
            await self._shared_http_client.aclose()
            self._shared_http_client = None
    
    def clear_cached_provider(self, name: str) -> None:
        """Remove a cached provider so it will be recreated with fresh auth.
        Also closes the underlying HTTP client to prevent resource leaks."""
        provider = self._providers.pop(name, None)
        if provider is not None and hasattr(provider, 'close'):
            try:
                import asyncio
                asyncio.ensure_future(provider.close())
            except Exception:
                pass


# ── Auth Storage ─────────────────────────────────────────────────────────────

class AuthStorage:
    """Manages API keys from env vars and auth file."""
    
    def __init__(self, agent_dir: str = "~/.agent"):
        self.agent_dir = os.path.expanduser(agent_dir)
        self._runtime_keys: Dict[str, str] = {}
        self._env_keys: Dict[str, str] = {
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "google": os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""),
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
        }
        self._auth_file = os.path.join(self.agent_dir, "auth.json")
        self._load_auth_file()
    
    def _load_auth_file(self) -> None:
        """Load auth keys from auth.json."""
        self._file_keys: Dict[str, str] = {}
        if os.path.exists(self._auth_file):
            try:
                with open(self._auth_file) as f:
                    self._file_keys = json.load(f)
            except Exception:
                pass
    
    def set_runtime_api_key(self, provider: str, key: str) -> None:
        """Set an API key at runtime (not persisted)."""
        self._runtime_keys[provider] = key
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key respecting priority: runtime > file > env."""
        if provider in self._runtime_keys:
            return self._runtime_keys[provider]
        
        if provider in self._file_keys:
            return self._file_keys[provider]
        
        return self._env_keys.get(provider) or None
    
    def save_auth(self, provider: str, key: str) -> None:
        """Save an API key to auth file."""
        os.makedirs(self.agent_dir, exist_ok=True)
        self._file_keys[provider] = key
        with open(self._auth_file, "w") as f:
            json.dump(self._file_keys, f, indent=2)


# ── Global Registry ──────────────────────────────────────────────────────────

_provider_registry: Optional[ProviderRegistry] = None
_auth_storage: Optional[AuthStorage] = None


def get_provider_registry(auth_storage: Optional[AuthStorage] = None) -> ProviderRegistry:
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry(auth_storage or get_auth_storage())
    return _provider_registry


def get_auth_storage(agent_dir: str = "~/.agent") -> AuthStorage:
    global _auth_storage
    if _auth_storage is None:
        _auth_storage = AuthStorage(agent_dir)
    return _auth_storage
