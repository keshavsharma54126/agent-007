# providers/base.py
from typing import Any, Dict, List, Generator, Optional


class LLMProvider:
    """
    Standard unified interface for all LLM providers.

    Methods:
        chat(...) -> Dict[str, Any]  # non-stream
        chat(..., stream=True) -> Generator[Dict[str, Any], None, None]  # stream
    Normalized response (non-stream):
    {
        "content": str,
        "tool_calls": list | None,
        "finish_reason": str | None,
        "usage": {
            "prompt_tokens": int | None,
            "completion_tokens": int | None,
            "total_tokens": int | None
        },
        "response_time": float,
        "raw": Any
    }

    Stream chunk:
    {
        "delta": str | None,
        "tool_calls": list | None,
        "finish_reason": str | None,
        "raw": Any
    }
    """

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[Any] = None,
        tool_choice: str = "auto",
        stream: bool = False,
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        raise NotImplementedError("chat must be implemented by provider adapters")
