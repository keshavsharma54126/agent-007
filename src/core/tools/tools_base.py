# tools.py
"""
Unified Tool Definitions & Normalization Layer
----------------------------------------------

This module defines a fully provider-agnostic tool schema that is compatible with:
- OpenAI (function calling)
- Anthropic / Claude (tool_use)
- Google Gemini (function calling)
- Groq (OpenAI compatible)
- OpenRouter (OpenAI compatible)

Use this file without modification across all LLM adapters.

"""

from typing import Any, Dict, List, Optional, Callable, TypedDict
import inspect


# -------------------------------------------------------------------
# 1. TOOL SPEC TYPES
# -------------------------------------------------------------------

class ToolParameter(TypedDict, total=False):
    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    items: Optional[Any]
    properties: Optional[Any]
    required: Optional[List[str]]


class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]


# -------------------------------------------------------------------
# 2. PUBLIC API FOR REGISTERING TOOLS
# -------------------------------------------------------------------

REGISTERED_TOOLS: Dict[str, Callable] = {}


def tool(fn: Callable) -> Callable:
    """
    Decorator that registers a function as a tool.
    The function signature becomes its schema automatically.
    """
    REGISTERED_TOOLS[fn.__name__] = fn
    fn.__tool_schema__ = build_tool_schema(fn)
    return fn


def build_tool_schema(fn: Callable) -> ToolDefinition:
    """
    Auto-build JSON schema for a Python function signature.
    """
    sig = inspect.signature(fn)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        annotation = param.annotation

        if annotation == int:
            ptype = "integer"
        elif annotation == float:
            ptype = "number"
        elif annotation == bool:
            ptype = "boolean"
        elif annotation == list:
            ptype = "array"
        elif annotation == dict:
            ptype = "object"
        else:
            ptype = "string"  # fallback

        properties[name] = {
            "type": ptype,
            "description": f"Parameter `{name}`"
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": fn.__doc__ or fn.__name__,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


# -------------------------------------------------------------------
# 3. EXPORT SCHEMAS FOR PROVIDERS
# -------------------------------------------------------------------

def openai_tools_format() -> List[Dict[str, Any]]:
    """
    Returns tools in native OpenAI format.
    """
    result = []
    for fn in REGISTERED_TOOLS.values():
        result.append({
            "type": "function",
            "function": fn.__tool_schema__
        })
    return result


def anthropic_tools_format() -> List[Dict[str, Any]]:
    """
    Returns tools in Claude / Anthropic native format.
    Claude requires:
    [
      {
        "name": "...",
        "description": "...",
        "input_schema": {...}
      }
    ]
    """
    out = []
    for fn in REGISTERED_TOOLS.values():
        schema = fn.__tool_schema__
        out.append({
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"]
        })
    return out


def gemini_tools_format() -> List[Dict[str, Any]]:
    """
    Gemini expects OpenAI-style:
    tools = [{
        "function_declarations": [...]
    }]
    """
    funcs = []
    for fn in REGISTERED_TOOLS.values():
        schema = fn.__tool_schema__
        funcs.append({
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"]
        })

    return funcs


# Groq + OpenRouter = OpenAI format
groq_tools_format = openai_tools_format
openrouter_tools_format = openai_tools_format


# -------------------------------------------------------------------
# 4. NORMALIZING PROVIDER TOOL-CALL RESPONSES
# -------------------------------------------------------------------

def normalize_tool_calls(provider: str, raw: Any) -> List[Dict[str, Any]]:
    """
    Converts provider-specific tool-call responses into unified format:
    [
      {
        "tool_name": "...",
        "arguments": {...},
        "raw": original_payload
      }
    ]
    """

    if raw is None:
        return []

    normalized = []

    # ---------------------------
    # OpenAI / Groq / OpenRouter
    # ---------------------------
    if provider in ["openai", "groq", "openrouter"]:
        calls = raw.get("tool_calls", [])
        for tc in calls:
            normalized.append({
                "tool_name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments", {}),
                "raw": tc
            })
        return normalized

    # ---------------------------
    # Anthropic / Claude
    # ---------------------------
    if provider == "anthropic":
        msgs = raw.get("content", [])
        for item in msgs:
            if item.get("type") == "tool_use":
                normalized.append({
                    "tool_name": item["name"],
                    "arguments": item.get("input", {}),
                    "raw": item
                })
        return normalized

    # ---------------------------
    # Gemini
    # ---------------------------
    if provider == "gemini":
        parts = raw.candidates[0].content.parts
        for p in parts:
            if hasattr(p, "function_call"):
                normalized.append({
                    "tool_name": p.function_call.name,
                    "arguments": p.function_call.args,
                    "raw": p
                })
        return normalized

    return []


# -------------------------------------------------------------------
# 5. RUN A TOOL BY NAME
# -------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Run a registered tool by name.
    """
    fn = REGISTERED_TOOLS.get(tool_name)
    if not fn:
        raise ValueError(f"Tool `{tool_name}` not registered")

    return fn(**arguments)


# -------------------------------------------------------------------
# 6. GET ALL TOOL SCHEMAS (useful for debugging)
# -------------------------------------------------------------------

def get_all_tool_schemas():
    """
    Returns dict of all registered tools and schemas.
    """
    return {name: fn.__tool_schema__ for name, fn in REGISTERED_TOOLS.items()}


# END OF FILE
