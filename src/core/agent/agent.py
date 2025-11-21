from typing import List, Dict, Optional, Any
from openai import OpenAI
import json
from ai_agent.providers.providerfactory import ProviderFactory
from ai_agent import query_database, search_wikipedia


class Agent:
    def __init__(self, system_prompt: Optional[str] = None,
                 provider: str = "openai", model: str = "gpt-4o"):
        self.llm = ProviderFactory(provider, model)
        self.messages: List[Dict[str, str]] = []
        default_prompt = "You are a helpful assistant. Use tools when necessary."
        self.messages.append(
            {"role": "system", "content": system_prompt or default_prompt})

    def execute_tool(self, tool_call: Any) -> str:
        """
        tool_call is expected to have: function.name, function.arguments (JSON)
        Normalize errors kindly.
        """
        try:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            return json.dumps({"error": f"Invalid tool call object: {e}"})

        try:
            if fn == "query_database":
                return query_database(args.get("query", ""))
            elif fn == "search_wikipedia":
                return search_wikipedia(args.get("query", ""))
            else:
                return json.dumps({"error": f"Unknown tool: {fn}"})
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def process_query(self, user_input: str, max_iterations: int = 5, stream: bool = False) -> str:
        """
        If stream=True and provider supports streaming, yields chunks (generator).
        Otherwise returns final content string.
        """
        self.messages.append({"role": "user", "content": user_input})

        # STREAMING MODE: if provider supports streaming, return a generator
        if stream:
            # The provider.chat(...) may be a generator of chunks
            gen = self.llm.chat(messages=self.messages, tools=None,
                                tool_choice="auto", stream=True)
            # We don't directly support tool-calls-in-stream here for simplicity (tool-calls typically arrive in control message)
            # The caller should iterate over this generator and render chunks.
            return gen  # caller must iterate
        # NON-STREAM mode
        last_response_content = ""
        for _ in range(max_iterations):
            normalized = self.llm.chat(
                messages=self.messages, tools=None, tool_choice="auto", stream=False)
            content = normalized.get("content")
            tool_calls = normalized.get("tool_calls")
            finish_reason = normalized.get("finish_reason")

            # If no tool calls, finish
            if not tool_calls:
                # append assistant message and return
                self.messages.append({"role": "assistant", "content": content or ""})
                last_response_content = content or ""
                return last_response_content

            # If there are tool calls, execute sequentially
            # tool_calls expected to be list of tool call objects with .function name/arguments and .id
            for tc in tool_calls:
                result = self.execute_tool(tc)
                # append tool output to the messages so the LLM can see results
                self.messages.append(
                    {"role": "tool", "tool_call_id": getattr(tc, "id", None), "content": result})

        # If exceeded iterations:
        final_msg = "Max tool-call iterations reached. Partial answer: " + \
            (last_response_content or "")
        self.messages.append({"role": "assistant", "content": final_msg})
        return final_msg

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.messages
