"""
Program-counter updater — decides which instruction line to execute next.

Uses an LLM to classify the control-flow decision, then deterministically
computes the actual next line index from the program structure.
"""
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class PCAction(str, Enum):
    ADVANCE_1 = "ADVANCE_1"
    JUMP_TO_ELSE = "JUMP_TO_ELSE"
    JUMP_OUT_WHILE = "JUMP_OUT_WHILE"
    JUMP_BACK_WHILE = "JUMP_BACK_WHILE"
    STAY = "STAY"


@dataclass
class ProgramStructure:
    lines: List[str]
    starred_idx: int
    open_to_close: Dict[int, int]
    close_to_open: Dict[int, int]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PC_UPDATER_PROMPT = """
You are PC_UPDATER. You are a control-flow decider.

You must follow the decision rules exactly and in priority order.
Do not "reason about what should happen" beyond these rules.
Do not invent extra control flow (there is no "else if" in this language).

Given:
- RAW_PROGRAM_WITH_STAR: full program text with exactly one line prefixed by "* " (the CURRENT line).
- CUR_LINE: the current line only (same text as the starred line, but without "* ").
- INTERPRETER_ACTION: the atomic action that was executed for CUR_LINE.
- EXECUTOR: a JSON object with fields like:
    - local_delta (object; may include "__cond")
    - pc_message (string; may start with "ERROR:")
    - global_updates (object; usually empty here)

You must output:
thought: one short paragraph
decision: EXACTLY ONE token from:
  ADVANCE_1
  JUMP_TO_ELSE
  JUMP_OUT_WHILE
  JUMP_BACK_WHILE

DECISION RULES (apply in this exact priority order):

(1) If INTERPRETER_ACTION starts with "control_break(" -> decision MUST be JUMP_OUT_WHILE.

(2) If INTERPRETER_ACTION starts with "eval_condition(" then:
  (2a) If __cond is True -> decision MUST be ADVANCE_1.
  (2b) If __cond is False -> decision MUST be JUMP_TO_ELSE.

(3) If CUR_LINE is "}" then decide what block it closes:
  - Find the nearest unmatched opening "{" above it.
  - Look at the header line that introduced that block:
      * If it is a While block -> decision MUST be JUMP_BACK_WHILE.
      * Otherwise (If/Else/other) -> decision MUST be ADVANCE_1.

(4) ALL OTHER LINES -> decision MUST be ADVANCE_1.

Output format (STRICT):
thought: <one short paragraph>
decision: <ONE token from the allowed list>
"""


# ---------------------------------------------------------------------------
# Line helpers
# ---------------------------------------------------------------------------

def _strip_star(line: str) -> str:
    return line[2:] if line.startswith("* ") else line


def _is_if_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("If ") and "{" in s


def _is_while_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("While ") and "{" in s


def _is_else_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("Else") and "{" in s


def _is_break_line(line: str) -> bool:
    return line.strip() == "break"


def _is_closing_brace_only(line: str) -> bool:
    return line.strip() == "}"


def _count_braces(line: str) -> Tuple[int, int]:
    return line.count("{"), line.count("}")


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def parse_program_structure(raw_program_with_star: str) -> ProgramStructure:
    """Parse program text with one starred line into lines, starred index,
    and opening/closing brace maps."""
    raw_lines = raw_program_with_star.splitlines()
    starred = [i for i, ln in enumerate(raw_lines) if ln.startswith("* ")]
    if len(starred) != 1:
        raise ValueError(f"Expected exactly one '* ' line, found {len(starred)}")
    starred_idx = starred[0]

    lines = [_strip_star(ln) for ln in raw_lines]

    open_to_close: Dict[int, int] = {}
    close_to_open: Dict[int, int] = {}
    stack: List[int] = []

    for i, ln in enumerate(lines):
        opens, closes = _count_braces(ln)
        for _ in range(opens):
            stack.append(i)
        for _ in range(closes):
            if not stack:
                continue
            open_i = stack.pop()
            open_to_close[open_i] = i
            close_to_open[i] = open_i

    return ProgramStructure(
        lines=lines,
        starred_idx=starred_idx,
        open_to_close=open_to_close,
        close_to_open=close_to_open,
    )


# ---------------------------------------------------------------------------
# Structural queries
# ---------------------------------------------------------------------------

def _find_next_nonempty_idx(lines: List[str], start: int) -> Optional[int]:
    for i in range(start, len(lines)):
        if lines[i].strip() != "":
            return i
    return None


def _find_matching_else_header(
    struct: ProgramStructure,
    if_header_idx: int,
) -> Optional[int]:
    if if_header_idx not in struct.open_to_close:
        return None
    if_close = struct.open_to_close[if_header_idx]
    j = _find_next_nonempty_idx(struct.lines, if_close + 1)
    if j is not None and _is_else_header(struct.lines[j]):
        return j
    return None


def _find_enclosing_while_header(
    struct: ProgramStructure,
    idx: int,
) -> Optional[int]:
    candidates = []
    for i, ln in enumerate(struct.lines):
        if _is_while_header(ln) and i in struct.open_to_close:
            close_i = struct.open_to_close[i]
            if i < idx <= close_i:
                candidates.append(i)
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Next-index computation (deterministic)
# ---------------------------------------------------------------------------

def compute_next_idx(
    struct: ProgramStructure,
    pc_action: PCAction,
    interpreter_action: str,
    executor: Dict[str, Any],
) -> int:
    """Compute the next program line index after executing current line."""
    lines = struct.lines
    i = struct.starred_idx
    cur_line = lines[i]

    pc_message = (executor or {}).get("pc_message", "") or ""
    if pc_message.startswith("ERROR:"):
        return i  # STAY on error

    # --- closing brace ---
    if _is_closing_brace_only(cur_line):
        open_i = struct.close_to_open.get(i)
        if open_i is not None:
            opener = lines[open_i]
            if _is_while_header(opener):
                return open_i
            if _is_if_header(opener):
                else_i = _find_matching_else_header(struct, open_i)
                if else_i is not None and else_i in struct.open_to_close:
                    else_close = struct.open_to_close[else_i]
                    nxt = _find_next_nonempty_idx(lines, else_close + 1)
                    return nxt if nxt is not None else i
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    # --- break ---
    if interpreter_action.strip().startswith("control_break(") or _is_break_line(cur_line):
        wh_i = _find_enclosing_while_header(struct, i)
        if wh_i is not None and wh_i in struct.open_to_close:
            close_i = struct.open_to_close[wh_i]
            nxt = _find_next_nonempty_idx(lines, close_i + 1)
            return nxt if nxt is not None else i
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    # --- explicit PC actions ---
    if pc_action == PCAction.STAY:
        return i

    if pc_action == PCAction.ADVANCE_1:
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    if pc_action == PCAction.JUMP_TO_ELSE:
        if _is_if_header(cur_line):
            else_i = _find_matching_else_header(struct, i)
            if else_i is not None:
                nxt = _find_next_nonempty_idx(lines, else_i + 1)
                return nxt if nxt is not None else i
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    if pc_action == PCAction.JUMP_OUT_WHILE:
        if _is_while_header(cur_line) and i in struct.open_to_close:
            close_i = struct.open_to_close[i]
            nxt = _find_next_nonempty_idx(lines, close_i + 1)
            return nxt if nxt is not None else i
        wh_i = _find_enclosing_while_header(struct, i)
        if wh_i is not None and wh_i in struct.open_to_close:
            close_i = struct.open_to_close[wh_i]
            nxt = _find_next_nonempty_idx(lines, close_i + 1)
            return nxt if nxt is not None else i
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    if pc_action == PCAction.JUMP_BACK_WHILE:
        wh_i = _find_enclosing_while_header(struct, i)
        if wh_i is not None:
            return wh_i
        nxt = _find_next_nonempty_idx(lines, i + 1)
        return nxt if nxt is not None else i

    nxt = _find_next_nonempty_idx(lines, i + 1)
    return nxt if nxt is not None else i


# ---------------------------------------------------------------------------
# Star manipulation
# ---------------------------------------------------------------------------

def move_star(raw_program_with_star: str, next_idx: int) -> str:
    raw_lines = raw_program_with_star.splitlines()
    stripped = [_strip_star(ln) for ln in raw_lines]
    if 0 <= next_idx < len(stripped):
        stripped[next_idx] = "* " + stripped[next_idx]
    return "\n".join(stripped)


def _extract_starred_cur_line(raw_program_with_star: str) -> str:
    for ln in raw_program_with_star.splitlines():
        if ln.startswith("* "):
            return ln[2:]
    return ""


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

def _parse_pc_updater_output(text: str) -> Tuple[str, str]:
    """Returns ``(decision_token, thought_text)``."""
    thought = ""
    decision = ""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("thought:"):
            thought = s[len("thought:"):].strip()
        elif s.lower().startswith("decision:"):
            decision = s[len("decision:"):].strip()
            break
    if not decision:
        for ln in (text or "").splitlines():
            s = ln.strip()
            if s in {"ADVANCE_1", "JUMP_TO_ELSE", "JUMP_OUT_WHILE", "JUMP_BACK_WHILE", "STAY"}:
                decision = s
                break
    return decision, thought


def decide_pc_action_llm(
    provider,
    model_config,
    raw_program_with_star: str,
    interpreter_action: str,
    executor: Dict[str, Any],
) -> Tuple[PCAction, str, str]:
    """Ask the LLM for the next PC decision given program and last step."""
    cur_line = _extract_starred_cur_line(raw_program_with_star)
    user_payload = {
        "RAW_PROGRAM_WITH_STAR": raw_program_with_star,
        "CUR_LINE": cur_line,
        "INTERPRETER_ACTION": interpreter_action,
        "EXECUTOR": {k: v for k, v in executor.items() if k != "raw_model_output"},
    }
    r = provider.chat(
        config=model_config,
        messages=[
            {"role": "user", "content": PC_UPDATER_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    pc_raw = (r.choices[0].message.content or "").strip()
    token, thought = _parse_pc_updater_output(pc_raw)
    try:
        action = PCAction(token)
    except Exception:
        action = PCAction.ADVANCE_1
    return action, pc_raw, thought


# ---------------------------------------------------------------------------
# Top-level PC update
# ---------------------------------------------------------------------------

def pc_update(
    provider,
    model_config,
    program_with_star: str,
    interpreter_action: str,
    executor: Dict[str, Any],
    rebase: bool = False,
) -> Tuple[str, str, str, str, str]:
    """Compute next program counter after one step.

    Returns ``(updated_program, next_line, pc_action_value, thought, raw)``.
    """
    struct = parse_program_structure(program_with_star)
    pc_action, pc_raw, pc_thought = decide_pc_action_llm(
        provider=provider,
        model_config=model_config,
        raw_program_with_star=program_with_star,
        interpreter_action=interpreter_action,
        executor=executor,
    )
    if rebase:
        logger.debug("PC_UPDATER raw:\n%s", pc_raw)
    next_idx = compute_next_idx(
        struct=struct,
        pc_action=pc_action,
        interpreter_action=interpreter_action,
        executor=executor,
    )
    updated = move_star(program_with_star, next_idx)
    next_line = _strip_star(updated.splitlines()[next_idx])
    return updated, next_line, pc_action.value, pc_thought, pc_raw
