"""
Orchestrator that drives the interpreter-executor-pc_updater loop.

Supports optional structured logging and resume-from-checkpoint.
"""
import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .interpreter import interpret
from .executor import execute
from .pc_updater import pc_update

logger = logging.getLogger(__name__)


def run_one_step(
    current_instruction: str,
    state: Dict[str, Any],
    memory: Dict[str, Any],
    pc_instruction: str,
    rebase: bool,
    time_sleep: int = 15,
    max_verify_retries: int = 1,
    *,
    provider,
    model_config,
) -> Tuple[str, Optional[str], Dict, Dict, Tuple[str, str, str]]:
    """
    Run interpreter -> executor -> pc_update once.  Mutates *state* and
    *memory* in place.

    Returns ``(updated_pc_instruction, next_instr, interpreter_result,
    executor_result, (pc_action, pc_thought, pc_raw))``.
    """
    for retry in range(3):
        try:
            interpreter_result = interpret(
                current_instruction, state, memory=memory, rebase=rebase,
                provider=provider, model_config=model_config,
            )
            break
        except Exception as e:
            logger.warning("interpreter error, retrying %d: %s", retry + 1, e)
            time.sleep(time_sleep)
    else:
        raise RuntimeError("interpreter failed after 3 retries")

    time.sleep(time_sleep)

    for retry in range(3):
        try:
            executor_result = execute(
                interpreter_result["action"], memory, state, rebase=rebase,
                provider=provider, model_config=model_config,
                max_verify_retries=max_verify_retries,
            )
            break
        except Exception as e:
            logger.warning("executor error, retrying %d: %s", retry + 1, e)
            time.sleep(time_sleep)
    else:
        raise RuntimeError("executor failed after 3 retries")

    if executor_result.get("terminated") or state.get("terminated"):
        return (
            pc_instruction, None,
            interpreter_result, executor_result,
            ("RETURN", "", "RETURN"),
        )

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
            logger.warning("pc_update error, retrying %d: %s", retry + 1, e)
            time.sleep(time_sleep)
    else:
        raise RuntimeError("pc_update failed after 3 retries")

    return (
        updated_program, next_instr,
        interpreter_result, executor_result,
        (pc_action, pc_thought, pc_raw),
    )


def solve(
    instruction: str,
    input_grid: List[List[int]],
    *,
    provider,
    model_config,
    max_steps: int = 500,
    time_sleep: int = 1,
    verbose: bool = True,
    execution_log=None,
    log_save_path: Optional[str] = None,
    log_save_every: int = -1,
    max_verify_retries: int = 1,
) -> Dict[str, Any]:
    """
    Run the interpreter-executor pipeline on a single grid.

    Parameters
    ----------
    instruction : str
        The natural-language program to execute.
    input_grid : list[list[int]]
        The ARC input grid.
    provider : LLMProvider
        LLM provider instance.
    model_config : ModelConfig
        Model configuration for the chat call.
    max_steps : int
        Safety limit on execution steps.
    time_sleep : int
        Seconds to sleep between LLM calls.
    verbose : bool
        Print step-by-step logs.
    execution_log : ExecutionLog | None
        Optional logger.  When provided, each step is recorded.  If the
        log already contains steps (resume mode), the solver fast-forwards
        to the checkpoint and continues from there.
    log_save_path : str | None
        File path to write the log to on periodic saves.
    log_save_every : int
        Save the log every N steps.  ``-1`` (default) means only at the end.
    max_verify_retries : int
        Maximum number of times to retry the executor verification.

    Returns
    -------
    dict
        ``output_grid``, ``steps``, ``terminated``, ``local_variables``.
    """
    # ------------------------------------------------------------------
    # Resume from checkpoint if the log already has recorded steps
    # ------------------------------------------------------------------
    start_step = 0
    if execution_log is not None and len(execution_log.steps) > 0:
        last = execution_log.steps[-1]
        memory = copy.deepcopy(last.post_memory)
        state = copy.deepcopy(last.post_state)
        pc_instruction = last.post_pc_instruction
        current_instruction = last.next_instruction
        start_step = last.step_number + 1
        if current_instruction is None:
            return {
                "output_grid": memory.get("grid"),
                "steps": [s.to_dict() for s in execution_log.steps],
                "terminated": True,
                "local_variables": state.get("local_variables", {}),
            }
        if verbose:
            logger.info("Resuming from step %d", start_step)
    else:
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

    for step in range(start_step, max_steps):
        pre_memory = copy.deepcopy(memory)
        pre_state = copy.deepcopy(state)

        updated_program, next_instr, ir, er, (pc_action, pc_thought, pc_raw) = run_one_step(
            current_instruction, state, memory, pc_instruction, rebase=verbose,
            time_sleep=time_sleep, provider=provider, model_config=model_config,
            max_verify_retries=max_verify_retries,
        )

        step_dict = {
            "step": step,
            "instruction": current_instruction,
            "interpreter": ir,
            "executor": er,
            "pc_action": pc_action,
        }
        step_logs.append(step_dict)

        if execution_log is not None:
            from arc.log.step_logger import StepRecord
            record = StepRecord(
                step_number=step,
                current_instruction=current_instruction,
                pc_instruction=pc_instruction,
                pre_memory=pre_memory,
                pre_state=pre_state,
                interpreter_action=ir.get("action", ""),
                interpreter_thought=ir.get("thought", ""),
                executor_result={k: v for k, v in er.items() if k != "raw_model_output"},
                pc_action=pc_action,
                pc_thought=pc_thought,
                post_memory=copy.deepcopy(memory),
                post_state=copy.deepcopy(state),
                post_pc_instruction=updated_program,
                next_instruction=next_instr,
            )
            execution_log.append(record)

            if (log_save_path is not None
                    and log_save_every > 0
                    and len(execution_log.steps) % log_save_every == 0):
                execution_log.save(log_save_path)
                if verbose:
                    logger.info("  [log saved at step %d]", step)

        if er.get("terminated") or state.get("terminated") or next_instr is None:
            if verbose:
                logger.info("Terminated at step %d", step)
            break

        pc_instruction = updated_program
        current_instruction = next_instr

    if execution_log is not None and log_save_path is not None:
        execution_log.save(log_save_path)
        if verbose:
            logger.info("  [log saved — %d steps total]", len(execution_log.steps))

    return {
        "output_grid": memory.get("grid"),
        "steps": step_logs,
        "terminated": state.get("terminated", False),
        "local_variables": state.get("local_variables", {}),
    }
