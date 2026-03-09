"""Shared utilities for slm_executor pipeline (parsing, nested keys, etc.)."""
import json
import re
from typing import Any, Dict


def set_nested_key(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set d[key1][key2]...[keyN] = value where path is 'key1.key2....keyN'."""
    keys = path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def parse_model_json(r: Any) -> Dict[str, Any]:
    """Parse JSON from an OpenAI-style response. Handles fenced blocks and extra text."""
    try:
        content = getattr(r.choices[0].message, "content", None)
    except Exception:
        content = None
    if not content or not str(content).strip():
        raise ValueError("Model returned empty content; cannot json.loads().")
    s = str(content).strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in model content. Content was: {content!r}")
        return json.loads(m.group(1))
