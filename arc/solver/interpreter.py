"""
Interpreter component — translates a single instruction line into an
atomic action string via an LLM call.
"""
import logging
import re
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

INTERPRETER_PROMPT = """
You are the INTERPRETER for a tiny, standardized instruction language.

Given:
(1) CUR_LINE: the CURRENT instruction line only (no other program lines). It is the exact line to process.
(2) LOCALS: current local variable snapshot (may be empty).
(3) GLOBAL MEMORY KEYS: names of available global objects (e.g., grid). Do not mutate globals except via grid_set(...) or write_global(...).

Your job:
Choose the NEXT single atomic action that advances execution of the CURRENT line only.

Core syntax conventions:
- <...>  : an external/value query. Preserve the query text (without brackets) in eval_query / eval_condition.
- '...'  : a direct string literal (no evaluation).
- (...)  : a boolean condition expression.
- {...}  : a block marker (structural).
- grid[x][y] : the ONLY syntax allowed to mutate the grid global.

EXPLICIT VARIABLE MARKER ($-RULE) — CRITICAL:
- Any token starting with "$" (e.g., "$x", "$y") is an explicit reference to a stored variable in LOCALS or GLOBAL MEMORY.
- "$" is a required part of the reference syntax between interpreter and executor.
- Therefore: NEVER remove, strip, normalize, or rewrite "$" when you emit action arguments.
- If CUR_LINE contains "$name", then the emitted action argument MUST contain "$name" exactly (including the "$").

How to treat "$name":
- "$name" means "use the value of variable name".
- "$name" is NOT an English query. It is a direct variable reference.
- If a required "$name" is missing from LOCALS and also not plausibly in GLOBAL MEMORY KEYS, emit error(missing=[...], ...).

General rules:
- Generate exactly ONE atomic action per call.
- Only generate an action that completes CUR. Do NOT act on any other lines.
- Do NOT execute the action. Do NOT update state.
- Never emit a no-op: if CUR is purely structural (Else/{/}), use structural(...) to advance.

How to handle each kind of CURRENT line:
A) Assignment to a string literal
   Form: name = 'text'
   -> set_local(name='name', value='text') # value WITHOUT quotes

B) Assignment to a direct variable reference (must keep $)
   Form: name = $var
   -> eval_query(dst='name', ref='$var') # KEEP the '$' in ref exactly

C) Assignment to an external/value query
   Form: name = <query>
   -> eval_query(dst='name', query='query') # query WITHOUT angle brackets, KEEP the '$' in ref exactly

D) Assignment to a local expression (may include $refs; keep $)
   Forms: name = (expr)  OR  name = expr
   - expr uses LOCALS variables, $refs, and literals. It must NOT include <...>.
   - Allowed operators: + - * / // % , parentheses, and/or/not, comparisons == != < <= > >=
   - If expr contains "$name", KEEP "$name" exactly in the emitted expr string.
   -> eval_expr(dst='name', expr='expr')

E) If / While headers (you ONLY evaluate the condition)
   Forms:
     If (condition) {
     While (condition) {
   - You do not choose branches or loop steps here; you only evaluate the condition into __cond.
   - condition MAY contain <query> segments (external checks). Preserve them literally.
   - condition may use: True/False, LOCALS variables, numeric/string literals, comparisons, and/or/not,
     plus the phrase "is true" / "is false".
   -> eval_condition(dst_bool='__cond', condition='condition') # condition WITHOUT outer parentheses

F) Else line (structural)
   Form: Else {
   -> structural(kind='ELSE')

G) Braces / block markers (structural)
   Lines such as "{", "}", or empty/whitespace
   -> structural(kind='LBRACE'|'RBRACE'|'EMPTY')

H) break statement
   Form: break
   -> control_break()

I) Grid mutation (the only allowed global mutation for grid)
   Form: grid[x][y] = rhs
   Constraints:
   - x and y must be either integer literals or names of existing LOCALS variables.
   - rhs must be either:
       * a LOCALS variable name, OR
       * a numeric
   - Do NOT use write_global to modify 'grid'. Only grid_set is allowed.
   -> grid_set(x='x', y='y', rhs_type='var|num', rhs='...')

J) Return statement
   Form: return
   -> return()

Global memory rule:
- write_global(...) is allowed for non-grid globals only.
- Never write_global(key='grid', ...). Use grid_set(...) for grid updates.

Output format (STRICT):
thought: <one short paragraph: what CUR requires and what you will compute/emit next>
action: <exactly one action call as required>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_yamlable(x: Any) -> Any:
    """Convert arbitrary objects into YAML-friendly primitives."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _coerce_yamlable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_coerce_yamlable(v) for v in x]
    return repr(x)


def _dump_yaml(data: Any) -> str:
    coerced = _coerce_yamlable(data)
    s = yaml.safe_dump(
        coerced,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )
    return s.rstrip()


def _format_nested_memory_keys(memory: Dict[str, Any], indent: str = "") -> str:
    """Format memory as nested keys (key structure only, no values)."""
    lines = []
    for k in sorted(memory.keys()):
        v = memory[k]
        if isinstance(v, dict) and v and all(isinstance(kk, str) for kk in v.keys()):
            lines.append(f"{indent}{k}:")
            lines.append(_format_nested_memory_keys(v, indent=indent + "  "))
        else:
            lines.append(f"{indent}{k}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User-input formatting
# ---------------------------------------------------------------------------

def format_interpreter_user_input(
    instruction: str,
    state: Dict[str, Any],
    memory: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user-message payload for the interpreter LLM call."""
    local_vars = state.get("local_variables") or {}
    parts = [
        f"CUR_LINE:\n{instruction.rstrip()}\n\n",
        "LOCALS:\n" + _dump_yaml(local_vars),
    ]
    if memory is not None:
        parts.append("\n\nGLOBAL MEMORY KEYS:\n")
        parts.append(_format_nested_memory_keys(memory))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_action_from_response(raw: str) -> str:
    """
    Extract the action string from the interpreter's raw LLM output.
    Expects ``thought: ...\\naction: <action>``.
    """
    if not raw or not raw.strip():
        return ""
    s = raw.strip()
    idx = s.find("\naction:")
    if idx != -1:
        line = s[idx + 1:]
        if line.lower().startswith("action:"):
            return line[7:].strip()
    if s.lower().startswith("action:"):
        return s[7:].strip()
    return s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    Ask the LLM to choose the next atomic interpreter action for *instruction*.

    Returns dict with keys ``action`` (str), ``thought`` (str),
    and ``raw_model_output`` (str).
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
    action = parse_action_from_response(raw)

    if rebase:
        logger.debug("INTERPRETER raw:\n%s", raw)

    thought_match = re.search(
        r"(?:^|\n)\s*thought:\s*\n?(.*?)(?=\n\s*action:|\Z)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    return {"action": action, "thought": thought, "raw_model_output": raw}
