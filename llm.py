"""
LLM endpoint used by check_eligibility, shared by the MCP server and the Flask API.

Any OpenAI-compatible service (OpenAI, vLLM, Ollama, LM Studio, LiteLLM, ...)
configured in .env with the same variables chatbotgbm uses:

  LLM_BASE_URL          e.g. https://llm.example.org/v1  (empty: OpenAI)
  LLM_API_KEY           falls back to OPENAI_API_KEY
  LLM_MODEL             default: o3-mini
  LLM_REASONING_EFFORT  sent only when set; default "medium" on OpenAI

With none of them set, behaviour is unchanged: OpenAI o3-mini.
Settings are read when called, so they follow load_dotenv() in the servers.
"""

import json
import os
import re

from openai import OpenAI

DEFAULT_MODEL = "o3-mini"


def base_url() -> str | None:
    return os.getenv("LLM_BASE_URL") or None


def model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def make_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and base_url():
        api_key = "not-needed"  # local servers such as Ollama accept any key
    return OpenAI(api_key=api_key, base_url=base_url())


def request_params() -> dict:
    """Model and options for chat.completions.create."""
    params = {"model": model(), "response_format": {"type": "json_object"}}
    effort = os.getenv("LLM_REASONING_EFFORT", "" if base_url() else "medium")
    if effort:
        params["reasoning_effort"] = effort
    return params


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json_answer(text: str) -> dict:
    """
    JSON object from a model answer.  Tolerates what open models add around
    it: <think> blocks, ```json fences, text before or after the object.
    Raises ValueError if there is no JSON object.
    """
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text or "").strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object in the model answer")
        return json.loads(cleaned[start:end + 1])
