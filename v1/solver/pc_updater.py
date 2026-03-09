"""
Program-counter updater — decides which instruction line to execute next.

Uses a deterministic set of rules (ADVANCE_1, JUMP_TO_ELSE, etc.) applied
to the program structure, with an LLM fallback for ambiguous cases.

NOTE: The PC updater prompt is closed-source.
"""
import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


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


PC_UPDATER_PROMPT = "[REDACTED — closed-source prompt]"


# ---------- Line helpers ----------

def _strip_star(line: str) -> str:
    return line[2:] if line.startswith("* ") else line


def _is_if_header(line: str) -> bool:
    return bool(re.match(r"\s*If\s*\(", _strip_star(line)))


def _is_else_header(line: str) -> bool:
    return bool(re.match(r"\s*Else\s*\{", _strip_star(line)))


def _is_while_header(line: str) -> bool:
    return bool(re.match(r"\s*While\s*\(", _strip_star(line)))


def _is_opening_brace(line: str) -> bool:
    return _strip_star(line).rstrip().endswith("{")


def _is_closing_brace(line: str) -> bool:
    return _strip_star(line).strip() == "}"


def _is_return(line: str) -> bool:
    return _strip_star(line).strip().lower() == "return"


# ---------- Structure parsing ----------

def parse_structure(lines: List[str], starred_idx: int) -> ProgramStructure:
    open_to_close: Dict[int, int] = {}
    close_to_open: Dict[int, int] = {}
    stack: List[int] = []
    for i, line in enumerate(lines):
        clean = _strip_star(line)
        if clean.rstrip().endswith("{"):
            stack.append(i)
        elif clean.strip() == "}":
            if stack:
                opener = stack.pop()
                open_to_close[opener] = i
                close_to_open[i] = opener
    return ProgramStructure(lines=lines, starred_idx=starred_idx,
                            open_to_close=open_to_close, close_to_open=close_to_open)


def _find_else_for_if(ps: ProgramStructure, if_idx: int) -> Optional[int]:
    close_idx = ps.open_to_close.get(if_idx)
    if close_idx is None:
        return None
    candidate = close_idx + 1
    if candidate < len(ps.lines) and _is_else_header(ps.lines[candidate]):
        return candidate
    return None


def _find_while_header_above(ps: ProgramStructure, close_idx: int) -> Optional[int]:
    opener = ps.close_to_open.get(close_idx)
    if opener is not None and _is_while_header(ps.lines[opener]):
        return opener
    for i in range(close_idx - 1, -1, -1):
        if _is_while_header(ps.lines[i]):
            return i
    return None


# ---------- PC update logic ----------

def _move_star(lines: List[str], old_idx: int, new_idx: int) -> List[str]:
    out = list(lines)
    if 0 <= old_idx < len(out) and out[old_idx].startswith("* "):
        out[old_idx] = out[old_idx][2:]
    if 0 <= new_idx < len(out):
        if not out[new_idx].startswith("* "):
            out[new_idx] = "* " + out[new_idx]
    return out


def apply_pc_action(
    action: PCAction,
    ps: ProgramStructure,
) -> Tuple[List[str], int]:
    idx = ps.starred_idx

    if action == PCAction.ADVANCE_1:
        new_idx = min(idx + 1, len(ps.lines) - 1)

    elif action == PCAction.JUMP_TO_ELSE:
        else_idx = _find_else_for_if(ps, idx)
        new_idx = (else_idx + 1) if else_idx is not None else min(idx + 1, len(ps.lines) - 1)

    elif action == PCAction.JUMP_OUT_WHILE:
        close_idx = ps.open_to_close.get(idx)
        new_idx = (close_idx + 1) if close_idx is not None else min(idx + 1, len(ps.lines) - 1)

    elif action == PCAction.JUMP_BACK_WHILE:
        while_idx = _find_while_header_above(ps, idx)
        new_idx = while_idx if while_idx is not None else idx

    elif action == PCAction.STAY:
        new_idx = idx

    else:
        new_idx = min(idx + 1, len(ps.lines) - 1)

    new_lines = _move_star(ps.lines, idx, new_idx)
    return new_lines, new_idx


def pc_update(
    *,
    provider,
    model_config,
    program_with_star: str,
    interpreter_action: str,
    executor: Dict[str, Any],
    rebase: bool = False,
) -> Tuple[str, Optional[str], str, str, str]:
    """
    Determine the next PC action and apply it.

    Returns ``(updated_program, next_instruction, pc_action, thought, raw)``.
    """
    lines = program_with_star.splitlines()
    starred_idx = next((i for i, l in enumerate(lines) if l.startswith("* ")), 0)
    ps = parse_structure(lines, starred_idx)

    user_content = (
        f"PROGRAM (starred line = current):\n{program_with_star}\n\n"
        f"INTERPRETER_ACTION: {interpreter_action}\n\n"
        f"EXECUTOR: {json.dumps(executor, default=str)}"
    )

    r = provider.chat(
        config=model_config,
        messages=[
            {"role": "system", "content": PC_UPDATER_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = r.choices[0].message.content or ""
    if rebase:
        print("PC_UPDATER raw:\n", raw)

    thought_match = re.search(r"thought:\s*(.*?)(?=\ndecision:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else ""

    decision_match = re.search(r"decision:\s*(\w+)", raw, re.IGNORECASE)
    decision_str = decision_match.group(1).strip().upper() if decision_match else "ADVANCE_1"

    try:
        pc_action = PCAction(decision_str)
    except ValueError:
        pc_action = PCAction.ADVANCE_1

    new_lines, new_idx = apply_pc_action(pc_action, ps)
    updated_program = "\n".join(new_lines)

    next_instr = _strip_star(new_lines[new_idx]) if new_idx < len(new_lines) else None
    if next_instr and _is_return(next_instr):
        next_instr = None

    return updated_program, next_instr, pc_action.value, thought, raw
