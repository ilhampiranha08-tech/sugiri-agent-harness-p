"""
Example: Custom tool extension.

Registers a custom 'weather' tool.
"""

from src.core.types import ExtensionAPI, ToolCallResult, AgentTool


class WeatherTool(AgentTool):
    name = "weather"
    label = "Weather"
    description = "Get the current weather for a city. Returns temperature and conditions."
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "country": {"type": "string", "description": "Country code (optional)"},
        },
        "required": ["city"],
    }
    
    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        city = params["city"]
        country = params.get("country", "")
        
        # Simulated weather data
        weather_data = {
            "London": "🌧️ 12°C, Rainy",
            "Tokyo": "🌤️ 22°C, Partly Cloudy",
            "New York": "☀️ 28°C, Sunny",
            "Singapore": "🌩️ 30°C, Thunderstorms",
            "Sydney": "☀️ 25°C, Clear",
        }
        
        key = city if city in weather_data else None
        if key is None:
            # Try with country
            for k in weather_data:
                if city.lower() in k.lower():
                    key = k
                    break
        
        weather = weather_data.get(key, f"🌤️ 20°C, Unknown (data not available for {city})")
        
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name="weather",
            params=params,
            content=[{
                "type": "text",
                "text": f"Weather for {city}: {weather}",
            }],
            details={"city": city, "weather": weather},
        )


def default(api: ExtensionAPI):
    """Register custom weather tool."""
    api.register_tool(WeatherTool())
    print("[Weather Tool] Loaded! 'weather' tool registered.")
