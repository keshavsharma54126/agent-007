# providers/claude_provider.py
import time
from typing import Any, Dict, Generator, List, Optional
import os

# anthropic SDK assumed (pip install anthropic)
from anthropic import Anthropic
from src.ai_agent.providers.llm_provider import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def chat(self, messages: List[Dict[str, str]], tools: Optional[Any] = None,
             tool_choice: str = "auto", stream: bool = False):

        start = time.time()

        # Anthropics / Claude SDKs often implement a messages.create(...) call returning full response.
        # Some versions support streaming; if not, we fallback to non-stream.
        if stream:
            # If Claude SDK supports streaming, use it. If not, fall back to non-stream below.
            try:
                stream_obj = self.client.streaming.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                for chunk in stream_obj:
                    # chunk normalization depends on SDK shape
                    yield {
                        "delta": getattr(chunk, "delta", None) or getattr(chunk, "text", None),
                        "tool_calls": getattr(chunk, "tool_calls", None),
                        "finish_reason": getattr(chunk, "finish_reason", None),
                        "raw": chunk
                    }
                return
            except Exception:
                # streaming unsupported -> fall through to non-stream implementation
                pass

        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )

        # response.content might be a list with text
        content = None
        if getattr(response, "content", None):
            try:
                # some SDKs put text under content[0].text
                content = response.content[0].text
            except Exception:
                content = str(response.content)

        tool_calls = getattr(response, "tool_calls", None)

        # Claude/Anthropic usually do not reveal usage tokens in all SDKs; attempt best-effort
        usage = None

        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": getattr(response, "finish_reason", None),
            "usage": usage,
            "response_time": time.time() - start,
            "raw": response
        }
