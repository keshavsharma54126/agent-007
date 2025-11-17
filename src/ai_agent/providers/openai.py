# providers/openai_provider.py
import time
from typing import Any, Dict, Generator, List, Optional
import os

from openai import OpenAI  # official OpenAI Python SDK (v1+ openai-python)
from src.ai_agent.providers.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, messages: List[Dict[str, str]], tools: Optional[Any] = None,
             tool_choice: str = "auto", stream: bool = False):

        start = time.time()

        if stream:
            # SDK returns an iterator of chunks
            stream_obj = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
            )
            for chunk in stream_obj:
                # Normalize chunk
                choice = chunk.choices[0]
                yield {
                    "delta": getattr(choice, "delta", {}).get("content") if hasattr(choice, "delta") else None,
                    "tool_calls": getattr(choice, "delta", {}).get("tool_calls") if hasattr(choice, "delta") else None,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "raw": chunk
                }
            return

        # non-streaming
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )
        choice = response.choices[0]
        msg = choice.message

        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return {
            "content": getattr(msg, "content", None),
            "tool_calls": getattr(msg, "tool_calls", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage_dict,
            "response_time": time.time() - start,
            "raw": response
        }
