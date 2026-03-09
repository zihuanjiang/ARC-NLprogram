"""
Orchestrator that drives the interpreter-executor-pc_updater loop.
"""
import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from .interpreter import interpret
from .executor import execute
from .pc_updater import pc_update


def run_one_step(
    current_instruction: str,
    state: Dict[str, Any],
    memory: Dict[str, Any],
    pc_instruction: str,
    rebase: bool,
    time_sleep: int = 15,
    *,
    provider,
    model_config,
) -> Tuple[str, Optional[str], Dict, Dict, Tuple[str, str, str]]:
    """
    Run interpreter -> executor -> pc_update once.  Mutates *state* and
    *memory* in place.
    """
    for retry in range(3):
        try:
            interpreter_result = interpret(
                current_instruction, state, memory=memory, rebase=rebase,
                provider=provider, model_config=model_config,
            )
            break
        except Exception as e:
            print(f"interpreter error, retrying {retry + 1}: {e}")
            time.sleep(time_sleep)
    else:
        raise RuntimeError("interpreter failed after 3 retries")

    time.sleep(time_sleep)

    for retry in range(3):
        try:
            executor_result = execute(
                interpreter_result["action"], memory, state, rebase=rebase,
                provider=provider, model_config=model_config,
            )
            break
        except Exception as e:
            print(f"executor error, retrying {retry + 1}: {e}")
            time.sleep(time_sleep)
    else:
        raise RuntimeError("executor failed after 3 retries")

    if executor_result.get("terminated") or state.get("terminated"):
        return pc_instruction, None, interpreter_result, executor_result, ("RETURN", "", "RETURN")

    time.sleep(time_sleep)

    for retry in range(3):
        try:
            updated_program, next_instr, pc_action, pc_thought, pc_raw = pc_update(
                provider=provider,
                model_config=model_config,
                program_with_star=pc_instruction,
                interpreter_action=interpreter_result["action"],
                executor=executor_result,
                rebase=rebase,
            )
            break
        except Exception as e:
            print(f"pc_update error, retrying {retry + 1}: {e}")
            time.sleep(time_sleep)
    else:
        raise RuntimeError("pc_update failed after 3 retries")

    return updated_program, next_instr, interpreter_result, executor_result, (pc_action, pc_thought, pc_raw)


def solve(
    instruction: str,
    input_grid: List[List[int]],
    *,
    provider,
    model_config,
    max_steps: int = 500,
    time_sleep: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the interpreter-executor pipeline on a single grid.

    Args:
        instruction: The natural-language program to execute.
        input_grid: The ARC input grid (list of lists of ints).
        provider: An ``LLMProvider`` instance.
        model_config: A ``ModelConfig`` for the chat call.
        max_steps: Safety limit on execution steps.
        time_sleep: Seconds to sleep between LLM calls.
        verbose: Print step-by-step logs.

    Returns:
        Dict with ``output_grid``, ``steps``, ``terminated``, etc.
    """
    memory: Dict[str, Any] = {
        "grid": copy.deepcopy(input_grid),
        "grid_width": len(input_grid[0]) if input_grid else 0,
        "grid_height": len(input_grid),
    }
    state: Dict[str, Any] = {"local_variables": {}}

    lines = instruction.splitlines()
    current_instruction = lines[0]
    lines[0] = "* " + lines[0]
    pc_instruction = "\n".join(lines)

    step_logs: List[Dict[str, Any]] = []

    for step in range(max_steps):
        updated_program, next_instr, ir, er, (pc_action, pc_thought, pc_raw) = run_one_step(
            current_instruction, state, memory, pc_instruction, rebase=verbose,
            time_sleep=time_sleep, provider=provider, model_config=model_config,
        )

        step_logs.append({
            "step": step,
            "instruction": current_instruction,
            "interpreter": ir,
            "executor": er,
            "pc_action": pc_action,
        })

        if er.get("terminated") or state.get("terminated") or next_instr is None:
            if verbose:
                print(f"Terminated at step {step}")
            break

        pc_instruction = updated_program
        current_instruction = next_instr

    return {
        "output_grid": memory.get("grid"),
        "steps": step_logs,
        "terminated": state.get("terminated", False),
        "local_variables": state.get("local_variables", {}),
    }
