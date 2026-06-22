"""
Model pricing per 1 million tokens (USD).
Prices approximate as of 2026. Update as providers change.
"""

# Per-model pricing: provider -> model_id -> {input, output, cache_read}
# cache_read: Anthropic discount for cached prompts (optional)
PRICING = {
    "anthropic": {
        "claude-sonnet-4-5":   {"input": 3.00,  "output": 15.00, "cache_read": 0.30},
        "claude-sonnet-4-6":   {"input": 3.00,  "output": 15.00, "cache_read": 0.30},
        "claude-opus-4-5":     {"input": 15.00, "output": 75.00},
        "claude-opus-4-8":     {"input": 15.00, "output": 75.00},
        "claude-haiku-4-5":    {"input": 0.80,  "output": 4.00},
    },
    "openai": {
        "gpt-4o":              {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":         {"input": 0.15,  "output": 0.60},
        "gpt-5.4":             {"input": 2.50,  "output": 10.00},
        "gpt-5.4-mini":        {"input": 0.30,  "output": 1.20},
        "gpt-5.5":             {"input": 3.75,  "output": 15.00},
        "o4-mini":             {"input": 1.10,  "output": 4.40},
    },
    "google": {
        "gemini-2.5-pro":      {"input": 1.25,  "output": 10.00},
        "gemini-2.5-flash":    {"input": 0.15,  "output": 0.60},
        "gemini-3-flash":      {"input": 0.15,  "output": 0.60},
        "gemini-3-pro-preview":{"input": 1.25,  "output": 10.00},
        "gemini-3.1-pro-preview":{"input": 1.25, "output": 10.00},
    },
    "deepseek": {
        "deepseek-v4-flash":   {"input": 0.20,  "output": 0.80},
        "deepseek-v4-pro":     {"input": 0.50,  "output": 2.00},
    },
}


def calculate_cost(provider: str, model_id: str, input_tokens: int = 0,
                   output_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Calculate cost in USD for token usage.
    
    Args:
        provider: Provider name (anthropic, openai, google, deepseek)
        model_id: Model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_read_tokens: Cached prompt tokens (Anthropic cache)
    
    Returns:
        Cost in USD
    """
    model_pricing = PRICING.get(provider, {}).get(model_id)
    if not model_pricing:
        # Fallback: try partial match
        for pid, models in PRICING.items():
            for mid, price in models.items():
                if mid in model_id or model_id in mid:
                    model_pricing = price
                    break
            if model_pricing:
                break
    
    if not model_pricing:
        return 0.0
    
    input_price = model_pricing.get("input", 0)
    output_price = model_pricing.get("output", 0)
    cache_price = model_pricing.get("cache_read", input_price)
    
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    
    # Cache discount for Anthropic: cache_read tokens are 90% cheaper
    if cache_read_tokens > 0 and "cache_read" in model_pricing:
        discount = (input_price - cache_price) * cache_read_tokens / 1_000_000
        cost -= discount
    
    return round(cost, 6)


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.2f}"
    else:
        return f"${cost:.2f}"
