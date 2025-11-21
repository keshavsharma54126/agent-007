# providers/openrouter_provider.py
import time
from typing import Any, Dict, Generator, List, Optional
import os

# OpenRouter commonly exposes an OpenAI-compatible client (openai package configured)
import openai as _openai_client  # some users set openai to call openrouter
from src.ai_agent.providers.llm_provider import LLMProvider


class OpenRouterProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        # user must set OPENROUTER_API_KEY
        _openai_client.api_key = os.getenv("OPENROUTER_API_KEY")
        # OPENROUTER base URL sometimes required:
        base_url = os.getenv("OPENROUTER_BASE_URL")
        if base_url:
            _openai_client.base_url = base_url
        self.client = _openai_client

    def chat(self, messages: List[Dict[str, str]], tools: Optional[Any] = None,
             tool_choice: str = "auto", stream: bool = False):
        start = time.time()

        if stream:
            try:
                stream_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True
                )
                for chunk in stream_obj:
                    choice = chunk.choices[0]
                    yield {
                        "delta": getattr(choice, "delta", {}).get("content") if hasattr(choice, "delta") else None,
                        "tool_calls": getattr(choice, "delta", {}).get("tool_calls") if hasattr(choice, "delta") else None,
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "raw": chunk
                    }
                return
            except Exception:
                # fallback to non-stream
                pass

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False
        )
        choice = response.choices[0]
        msg = getattr(choice, "message", choice)

        return {
            "content": getattr(msg, "content", None),
            "tool_calls": getattr(msg, "tool_calls", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": getattr(response, "usage", None),
            "response_time": time.time() - start,
            "raw": response
        }
