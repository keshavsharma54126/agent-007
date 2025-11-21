# providers/gemini_provider.py
import time
from typing import Any, Dict, Generator, List, Optional
import os

# google generative ai sdk
import google.generativeai as genai
from src.ai_agent.providers.llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # gemini SDK wraps calls differently; we won't instantiate a per-model client object in all SDK versions

    def chat(self, messages: List[Dict[str, str]], tools: Optional[Any] = None,
             tool_choice: str = "auto", stream: bool = False):
        """
        Gemini often expects a final user prompt or structured messages depending on SDK version.
        Many public SDKs may not support the exact tool-calling schema; we normalize best-effort.
        """

        start = time.time()

        # Many Gemini SDKs do not support "tool" injection in the same way; use a simple generate call.
        # Use the last user message as prompt (common pattern).
        user_msg = None
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content")
                break
        prompt = user_msg or (messages[-1]["content"] if messages else "")

        # streaming: if SDK supports streaming, implement; otherwise fall back.
        if stream:
            try:
                stream_gen = genai.generate_stream(
                    model=self.model,
                    prompt=prompt,
                )
                for event in stream_gen:
                    # event object shape varies; try common fields
                    yield {
                        "delta": getattr(event, "text", None) or getattr(event, "delta", None),
                        "tool_calls": getattr(event, "tool_calls", None),
                        "finish_reason": getattr(event, "finish_reason", None),
                        "raw": event
                    }
                return
            except Exception:
                # fallback to non-streaming
                pass

        # non-streaming
        resp = genai.generate(model=self.model, prompt=prompt)
        # gemini generate response shape depends on SDK; often resp.text or resp.candidates[0].output
        content = None
        if hasattr(resp, "text"):
            content = resp.text
        else:
            try:
                content = resp.candidates[0].output
            except Exception:
                content = str(resp)

        return {
            "content": content,
            "tool_calls": None,  # gemini tool-call normalization is advanced; set None unless you implement
            "finish_reason": None,
            "usage": None,
            "response_time": time.time() - start,
            "raw": resp
        }
