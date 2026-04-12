"""
Executor component — carries out atomic interpreter actions.

Actions are routed to either a deterministic Python executor (set_local,
eval_expr, grid_set, structural, control_break, return, write_global) or an
LLM-backed executor (eval_query, eval_condition).
"""
import ast
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXECUTOR_LOCAL_UPDATE_PROMPT = """
You are the EXECUTOR for ONE atomic interpreter action.

Given:
(1) ACTION: the single actomic action to execute.
(2) LOCALS: current local variable snapshot (may be empty).
(3) GLOBAL MEMORY: full global memory snapshot.

Your job:
- Execute ACTION exactly once.
- You may update locals ONLY as required by the action.
- You must always produce a short message summarizing what you did and the result (pc_message).

Rules:
- $-variables: the interpreter may write variable references like "$x".
  - "$" is ONLY a marker. The stored variable name is "x".
  - When reading any "$name", look up "name" (without "$") in LOCALS first, then GLOBAL MEMORY.
- If something is impossible (missing variable, invalid index, unclear query), do not guess:
  - return no local changes and set pc_message to "ERROR: ...".
- Do NOT mutate GLOBAL_MEMORY in this prompt (no grid edits, no global writes).
- If ACTION is set_local / eval_expr / eval_query / eval_condition: only set the destination variable.
- If ACTION is structural(...) or control_break(): do not change locals.
- STRICTLY follow the 1-indexed coordinate system: the first row is 1, the first column is 1.
- Color is represented by an integer between 0 and 9, each color is unique.
- Only color 0 is "black".

Output format (STRICT):
thought: <one short paragraph>
payload:
```json
{
  "local_variables": { "only_changed_or_new": "values" },
  "pc_message": "one short message"
}
```
"""

EXECUTOR_GLOBAL_UPDATE_PROMPT = """
You are the EXECUTOR for ONE atomic interpreter action that modifies GLOBAL MEMORY.

Given:
(1) ACTION: the single atomic action to execute.
(2) LOCALS: current local variable snapshot (may be empty).
(3) GLOBAL MEMORY: full global memory snapshot.

Your job:
- Execute ACTION exactly once.
- You may update both locals and global memory as required.
- You must always produce a short message summarizing what you did.

Output format (STRICT):
thought: <one short paragraph>
payload:
```json
{
  "local_variables": { "only_changed_or_new": "values" },
  "global_updates": { "key": "value" },
  "pc_message": "one short message"
}
```
"""

EXECUTOR_VERIFY_PROMPT = """
You are the VERIFIER for an atomic interpreter action execution.

Given:
(1) ACTION: the single atomic action that was executed.
(2) LOCALS: the local variable snapshot BEFORE execution.
(3) GLOBAL MEMORY: full global memory snapshot.
(4) EXECUTION RESULT: the result produced by the executor (local variable changes and message).

Your job:
- Analyze whether the EXECUTION RESULT correctly carries out the ACTION.
- Check that variable values, computations, and messages are accurate.
- Consider edge cases: off-by-one errors, wrong variable lookups, incorrect logic.

Rules:
- $-variables: the interpreter may write variable references like "$x".
  - "$" is ONLY a marker. The stored variable name is "x".
  - When reading any "$name", look up "name" (without "$") in LOCALS first, then GLOBAL MEMORY.
- ALL indexing in the pseudocode is 1-indexed: the first row is row 1, the first column is column 1.
  - The grid in GLOBAL MEMORY is stored as a 0-indexed array, so row 1 corresponds to grid[0] and column 1 corresponds to grid[0][0].
  - When verifying a result that references "row $row" or "column $col", always apply this 1-to-0 offset before looking up values in the grid array.
- Color is represented by an integer between 0 and 9, each color is unique.
- Only color 0 is "black".

Output format (STRICT):
thought: <one short paragraph analyzing whether the result is correct or not, and why>
payload:
```json
{
  "decision": "correct" or "incorrect",
  "reason": "brief explanation"
}
```
"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"^\s*([A-Za-z]\w*)\s*\(\s*(.*?)\s*\)\s*$", re.DOTALL)


def parse_action(action: str) -> Tuple[str, Dict[str, Any]]:
    """Parse ``action_name(key=value, ...)`` into ``(name, kwargs)``."""
    action = (action or "").strip()
    m = _ACTION_RE.match(action)
    if not m:
        return action, {}

    fn = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return fn, {}

    try:
        node = ast.parse(f"f({args_str})", mode="eval").body
    except SyntaxError:
        return fn, {}

    if not isinstance(node, ast.Call):
        return fn, {}

    kwargs: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            kwargs[kw.arg] = ast.unparse(kw.value)

    return fn, kwargs


def classify_executor_action(action: str) -> str:
    """
    Classify into ``'python_local'``, ``'python_global'``, or ``'llm_local'``.
    """
    a_type, _ = parse_action(action)
    a_type_l = (a_type or "").strip().lower()

    if a_type_l in {"set_local", "eval_expr", "structural", "control_break", "return"}:
        return "python_local"
    if a_type_l in {"grid_set", "write_global"}:
        return "python_global"
    if a_type_l in {"eval_query", "eval_condition"}:
        return "llm_local"
    return "llm_local"


# ---------------------------------------------------------------------------
# Expression evaluation helpers
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Name, ast.Constant, ast.Load,
)

_DOLLAR_IDENT_RE = re.compile(r"\$([A-Za-z_]\w*)")


def _strip_dollar(name: Any) -> Any:
    if isinstance(name, str) and name.startswith("$"):
        return name[1:]
    return name


def _resolve_local_key(locals_dict: Dict[str, Any], name: Any) -> Optional[str]:
    """Resolve a local variable key allowing optional leading ``$``."""
    if not isinstance(name, str):
        return None
    base = name[1:] if name.startswith("$") else name
    if base in locals_dict:
        return base
    if name in locals_dict:
        return name
    alt = "$" + base
    if alt in locals_dict:
        return alt
    return None


def _get_local_value(locals_dict: Dict[str, Any], name: Any) -> Any:
    k = _resolve_local_key(locals_dict, name)
    if k is None:
        raise KeyError(f"Local var '{name}' not found (checked with/without '$').")
    return locals_dict[k]


def _check_expr_ast(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(
            n,
            _ALLOWED_NODES + _ALLOWED_BINOPS + _ALLOWED_UNARYOPS
            + _ALLOWED_BOOLOPS + _ALLOWED_CMPOPS
        ):
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")
        if isinstance(n, ast.Call):
            raise ValueError("Function calls not allowed")
        if isinstance(n, ast.Attribute):
            raise ValueError("Attributes not allowed")
        if isinstance(n, ast.Subscript):
            raise ValueError("Subscripting not allowed in eval_expr")
        if isinstance(n, ast.BinOp) and not isinstance(n.op, _ALLOWED_BINOPS):
            raise ValueError(f"Disallowed binary op: {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp) and not isinstance(n.op, _ALLOWED_UNARYOPS):
            raise ValueError(f"Disallowed unary op: {type(n.op).__name__}")
        if isinstance(n, ast.BoolOp) and not isinstance(n.op, _ALLOWED_BOOLOPS):
            raise ValueError(f"Disallowed boolean op: {type(n.op).__name__}")
        if isinstance(n, ast.Compare):
            for op in n.ops:
                if not isinstance(op, _ALLOWED_CMPOPS):
                    raise ValueError(f"Disallowed compare op: {type(op).__name__}")


_INTLIKE_RE = re.compile(r"-?\d+")


def _coerce_for_eval(v: Any) -> Any:
    """Coerce string representations of numbers/booleans to native types.

    LLMs occasionally emit ``set_local(name='row', value='1')`` (string)
    instead of ``value=1`` (int).  When such values later appear in an
    arithmetic expression like ``$row + 1``, Python raises a ``TypeError``.
    This helper transparently converts those strings so that ``safe_eval_expr``
    works correctly.
    """
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    if _INTLIKE_RE.fullmatch(s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return v


def safe_eval_expr(expr: str, locals_dict: Dict[str, Any]) -> Any:
    """Evaluate a safe arithmetic / boolean expression with ``$``-variable substitution."""
    expr2 = _DOLLAR_IDENT_RE.sub(r"\1", expr)
    tree = ast.parse(expr2, mode="eval")
    _check_expr_ast(tree)
    env: Dict[str, Any] = {"True": True, "False": False}
    for k, v in (locals_dict or {}).items():
        coerced = _coerce_for_eval(v)
        env[k] = coerced
        if isinstance(k, str) and k.startswith("$"):
            env.setdefault(k[1:], coerced)
    return eval(
        compile(tree, "<eval_expr>", "eval"),
        {"__builtins__": {}},
        env,
    )


# ---------------------------------------------------------------------------
# Dot-path helpers for global memory
# ---------------------------------------------------------------------------

def get_by_dot_path(mem: Dict[str, Any], path: str) -> Any:
    cur: Any = mem
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Missing global path segment: {part} in {path}")
    return cur


def set_by_dot_path(mem: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = mem
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"Non-dict segment while traversing: {part} in {path}")
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    last = parts[-1]
    if not isinstance(cur, dict):
        raise KeyError(f"Cannot set on non-dict at final container for {path}")
    cur[last] = value


def _as_int_index(token: Any, locals_dict: Dict[str, Any]) -> int:
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        s = token.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        v = _get_local_value(locals_dict or {}, s)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
            return int(v.strip())
        raise ValueError(f"Local index var {s} is not an int: {v}")
    raise ValueError(f"Cannot convert index token to int: {token}")


# ---------------------------------------------------------------------------
# LLM executor helpers
# ---------------------------------------------------------------------------

def _is_grid_matrix(value: Any) -> bool:
    """Heuristic check for a 2-D grid (list of equal-length lists of ints)."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    row_len = len(value[0])
    if row_len == 0:
        return False
    for row in value:
        if len(row) != row_len:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _grid_to_ascii(grid: list) -> str:
    return "\n".join("".join(str(int(x)) for x in row) for row in grid)


def dump_global_memory_custom(memory: Dict[str, Any]) -> str:
    serialized: Dict[str, Any] = {}
    for key, value in memory.items():
        serialized[key] = value
    return json.dumps(serialized, indent=2)


def format_executor_user_prompt(
    action: str,
    state: Dict[str, Any],
    memory: Dict[str, Any],
) -> str:
    local_vars = state.get("local_variables") or {}
    return (
        f"ACTION:\n{action}\n\n"
        f"LOCALS:\n{json.dumps(local_vars, indent=2)}\n\n"
        f"GLOBAL_MEMORY:\n{dump_global_memory_custom(memory)}\n"
    )


def extract_payload_json(text: str) -> Dict[str, Any]:
    fenced_re = re.compile(r"\s*({.*?})\s*```", re.DOTALL)
    m = fenced_re.search(text or "")
    if not m:
        legacy_re = re.compile(r"json\s*({.*})", re.DOTALL)
        m = legacy_re.search(text or "")
    if not m:
        raise ValueError("No json { ... } payload found")

    raw_json = m.group(1)
    safe_json = raw_json.replace("\\'", "'")

    try:
        return json.loads(safe_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse executor payload JSON: {e}") from e


# ---------------------------------------------------------------------------
# Python executor (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _python_execute(
    action: str,
    memory: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute interpreter actions that need no LLM: set_local, eval_expr,
    structural, control_break, return, grid_set, write_global."""
    locals_dict = state.setdefault("local_variables", {}) or {}
    a_type, kw = parse_action(action)
    t = (a_type or "").strip().lower()

    local_delta: Dict[str, Any] = {}
    global_updates: Dict[str, Any] = {}

    def done(msg: str) -> Dict[str, Any]:
        if local_delta:
            locals_dict.update(local_delta)
        state["local_variables"] = locals_dict
        state["last_executor_message"] = msg
        return {
            "local_delta": local_delta,
            "global_updates": global_updates,
            "pc_message": msg,
        }

    try:
        if t == "set_local":
            name = _strip_dollar(kw["name"])
            value = kw["value"]
            local_delta[name] = value
            return done(f"Stored local {name}.")

        if t == "eval_expr":
            dst = _strip_dollar(kw["dst"])
            expr = kw["expr"]
            val = safe_eval_expr(expr, locals_dict)
            local_delta[dst] = val
            return done(f"Computed {dst} from expr.")

        if t == "structural":
            kind = kw.get("kind", "")
            return done(f"Structural step: {kind}.")

        if t == "control_break":
            return done("Break encountered.")

        if t == "return":
            state["terminated"] = True
            res = done("Return encountered.")
            res["terminated"] = True
            return res

        if t == "grid_set":
            if "grid" not in memory:
                return done("ERROR: GLOBAL_MEMORY has no 'grid'.")
            grid = memory["grid"]

            x_token = kw["x"]
            y_token = kw["y"]
            rhs_type = kw["rhs_type"]
            rhs = kw["rhs"]

            x = _as_int_index(x_token, locals_dict)
            y = _as_int_index(y_token, locals_dict)

            if rhs_type == "num":
                value = int(rhs) if isinstance(rhs, str) and re.fullmatch(r"-?\d+", rhs.strip()) else rhs
            elif rhs_type == "var":
                value = _get_local_value(locals_dict, rhs)
            else:
                return done(f"ERROR: invalid rhs_type {rhs_type}.")

            grid[x - 1][y - 1] = value
            global_updates["grid_set"] = {"x": x, "y": y, "value": value}
            return done(f"Updated grid[{x}][{y}].")

        if t == "write_global":
            key = kw["key"]
            src_var = kw["src_var"]

            if key == "grid" or key.startswith("grid."):
                return done("ERROR: write_global to grid is forbidden (use grid_set).")

            value = _get_local_value(locals_dict, src_var)
            set_by_dot_path(memory, key, value)
            global_updates["write_global"] = {"key": key, "value": value}
            return done(f"Wrote global {key} from {src_var}.")

        return done(f"ERROR: Unknown action type '{a_type}'.")
    except Exception as e:
        return done(f"ERROR: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def format_verify_user_prompt(
    action: str,
    pre_locals: Dict[str, Any],
    memory: Dict[str, Any],
    execution_result: Dict[str, Any],
) -> str:
    return (
        f"ACTION:\n{action}\n\n"
        f"LOCALS (before execution):\n{json.dumps(pre_locals, indent=2)}\n\n"
        f"GLOBAL_MEMORY:\n{dump_global_memory_custom(memory)}\n\n"
        f"EXECUTION RESULT:\n{json.dumps(execution_result, indent=2)}\n"
    )


def _verify_execution(
    action: str,
    pre_locals: Dict[str, Any],
    memory: Dict[str, Any],
    execution_result: Dict[str, Any],
    *,
    provider,
    model_config,
) -> Dict[str, Any]:
    """Call the LLM to verify an execution result.

    Returns ``{"decision": "correct"|"incorrect", "reason": str, "raw": str}``.
    """
    user_content = format_verify_user_prompt(
        action, pre_locals, memory, execution_result,
    )
    r = provider.chat(
        config=model_config,
        messages=[
            {"role": "user", "content": EXECUTOR_VERIFY_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = r.choices[0].message.content or ""
    try:
        payload = extract_payload_json(raw)
        decision = payload.get("decision", "correct").lower().strip()
        reason = payload.get("reason", "")
    except ValueError:
        logger.debug("VERIFY parse failed; accepting result by default.")
        decision = "correct"
        reason = "Verification parse failed; accepting result."
    logger.debug("VERIFY decision=%s reason=%s", decision, reason)
    return {"decision": decision, "reason": reason, "raw": raw}


def _format_retry_feedback(
    reason: str,
    execution_result: Dict[str, Any],
) -> str:
    result_str = json.dumps(execution_result, indent=2)
    return (
        "VERIFICATION FAILED — Your previous execution result was judged INCORRECT.\n\n"
        f"Verifier's analysis: {reason}\n\n"
        f"Your previous result was:\n{result_str}\n\n"
        "Please re-execute the ACTION and produce a corrected result. "
        "Follow the same output format as before."
    )


# ---------------------------------------------------------------------------
# LLM executor
# ---------------------------------------------------------------------------

def _llm_execute(
    action: str,
    memory: Dict[str, Any],
    state: Dict[str, Any],
    kind: Optional[str] = None,
    rebase: bool = False,
    *,
    provider,
    model_config,
    max_verify_retries: int = 1,
) -> Dict[str, Any]:
    """Execute an interpreter action via LLM (eval_query / eval_condition).

    After obtaining the initial result, an independent verifier LLM call checks
    correctness.  If the verifier judges the result incorrect, the wrong answer
    and feedback are appended to the conversation and the executor is asked to
    retry (up to *max_verify_retries* times).
    """
    if kind is None:
        kind = classify_executor_action(action)
    system_prompt = (
        EXECUTOR_GLOBAL_UPDATE_PROMPT if kind == "llm_global"
        else EXECUTOR_LOCAL_UPDATE_PROMPT
    )
    user_content = format_executor_user_prompt(action, state, memory)

    pre_locals = dict((state.get("local_variables") or {}).items())

    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    local_delta: Dict[str, Any] = {}
    global_updates: Dict[str, Any] = {}
    pc_message = ""

    for attempt in range(max_verify_retries + 1):
        r = provider.chat(config=model_config, messages=messages)
        raw = r.choices[0].message.content or ""

        if rebase:
            logger.debug("EXECUTOR raw (attempt %d):\n%s", attempt + 1, raw)

        payload = extract_payload_json(raw)
        local_delta = payload.get("local_variables") or {}
        global_updates = payload.get("global_updates") or {}
        pc_message = payload.get("pc_message") or "ERROR: missing pc_message"

        if attempt < max_verify_retries:
            execution_result = {
                "local_variables": local_delta,
                "pc_message": pc_message,
            }
            if global_updates:
                execution_result["global_updates"] = global_updates

            verification = _verify_execution(
                action, pre_locals, memory, execution_result,
                provider=provider, model_config=model_config,
            )

            if verification["decision"] != "incorrect":
                break

            logger.info(
                "EXECUTOR verify FAILED (attempt %d/%d): %s",
                attempt + 1, max_verify_retries + 1, verification["reason"],
            )
            messages.append({"role": "user", "content": raw})
            messages.append({
                "role": "user",
                "content": _format_retry_feedback(
                    verification["reason"], execution_result,
                ),
            })

    locals_dict = state.setdefault("local_variables", {}) or {}
    if local_delta:
        for name, value in local_delta.items():
            if name.startswith("_") and name not in locals_dict:
                continue
            locals_dict[name] = value
    state["local_variables"] = locals_dict
    state["last_executor_message"] = pc_message

    return {
        "local_delta": local_delta,
        "global_updates": global_updates,
        "pc_message": pc_message,
        "raw_model_output": raw,
    }


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def execute(
    action: str,
    memory: Dict[str, Any],
    state: Dict[str, Any],
    rebase: bool = False,
    *,
    provider,
    model_config,
    max_verify_retries: int = 1,
) -> Dict[str, Any]:
    """
    Top-level dispatcher: route to Python or LLM executor.

    Returns a dict with execution results and ``pc_message``.
    """
    route = classify_executor_action(action)

    if route in {"python_local", "python_global"}:
        return _python_execute(action, memory, state)

    return _llm_execute(
        action, memory, state, route, rebase,
        provider=provider, model_config=model_config,
        max_verify_retries=max_verify_retries,
    )
