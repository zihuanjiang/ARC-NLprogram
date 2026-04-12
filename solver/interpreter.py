"""
Interpreter component — translates a single instruction line into an
atomic action string.

NOTE: The interpreter prompt is closed-source.
"""
from typing import Any, Dict, Optional
import re

INTERPRETER_PROMPT = "[REDACTED — closed-source prompt]"


def format_interpreter_user_input(
    instruction: str,
    state: Dict[str, Any],
    memory: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user-message payload for the interpreter LLM call."""
    locals_dict = state.get("local_variables", {})
    parts = [f"CUR_LINE:\n{instruction}"]
    if locals_dict:
        parts.append(f"LOCALS:\n{locals_dict}")
    if memory:
        mem_keys = {k: type(v).__name__ for k, v in memory.items()}
        parts.append(f"MEMORY_KEYS:\n{mem_keys}")
    return "\n\n".join(parts)


def interpret(
    instruction: str,
    state: Dict[str, Any],
    memory: Optional[Dict[str, Any]] = None,
    rebase: bool = False,
    *,
    provider,
    model_config,
) -> Dict[str, Any]:
    """
    Ask the LLM to choose the next atomic interpreter action for the current line.

    Returns dict with keys ``action`` (str) and ``raw_model_output`` (str).
    """
    user_content = format_interpreter_user_input(instruction, state, memory=memory)
    r = provider.chat(
        config=model_config,
        messages=[
            {"role": "user", "content": INTERPRETER_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = r.choices[0].message.content or ""

    if rebase:
        print(raw)

    action_match = re.search(
        r"(?:^|\n)\s*action:\s*\n?(.*?)(?:\n\s*$|\Z)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    action_str = action_match.group(1).strip() if action_match else raw.strip()

    thought_match = re.search(
        r"(?:^|\n)\s*thought:\s*\n?(.*?)(?=\n\s*action:|\Z)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    return {"action": action_str, "thought": thought, "raw_model_output": raw}
