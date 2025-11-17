from typing import Any
from src.ai_agent.providers.openai import OpenAIProvider
from src.ai_agent.providers.claude import ClaudeProvider
from src.ai_agent.providers.gemini import GeminiProvider
from src.ai_agent.providers.openrouter import OpenRouterProvider


def ProviderFactory(provider_name: str, model: str) -> Any:
    p = provider_name.lower()
    if p == "openai":
        return OpenAIProvider(model)
    if p == "claude":
        return ClaudeProvider(model)
    if p == "gemini":
        return GeminiProvider(model)
    if p == "openrouter":
        return OpenRouterProvider(model)

    raise ValueError(f"Unknown provider: {provider_name}")
