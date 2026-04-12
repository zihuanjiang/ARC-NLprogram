"""
CLI entrypoint for the interpreter-executor ARC solver.

Usage:
    python -m arc --task_id <ID> --data_folder <PATH> --instruction <FILE> [OPTIONS]
"""
import argparse
import json
import logging
import os
import sys

from arc.data.ARCTask import ARCTask
from arc.llm.provider import LLMProvider
from arc.llm.config import MODEL_CONFIGURATIONS
from arc.solver.runner import solve
from arc.log.step_logger import ExecutionLog

logger = logging.getLogger("arc")


def _setup_logging(log_file: str) -> None:
    """Configure the ``arc`` logger hierarchy to write to *log_file*."""
    root = logging.getLogger("arc")
    root.setLevel(logging.DEBUG)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the interpreter-executor ARC solver.",
    )
    parser.add_argument("--task_id", required=True, help="ARC task ID to solve")
    parser.add_argument("--data_folder", required=True, help="Path to ARC data folder")
    parser.add_argument("--dataset_set", default="training", help="Dataset split (default: training)")
    parser.add_argument("--model_key", default=None, help="Model key from MODEL_CONFIGURATIONS")
    parser.add_argument("--instruction", required=True, help="Path to a .txt file with the NL program")
    parser.add_argument("--max_steps", type=int, default=500, help="Max execution steps")
    parser.add_argument("--max_verify_retries", type=int, default=1, help="Maximum number of times to retry the executor verification (default: 1)")
    parser.add_argument("--time_sleep", type=int, default=15, help="Sleep between LLM calls (seconds)")
    parser.add_argument("--output", default=None, help="Path to write the output grid JSON")
    parser.add_argument("--log_dir", default=None, help="Directory to save execution log")
    parser.add_argument("--resume", default=None, help="Path to a log JSON file to resume from")
    parser.add_argument("--log_every", type=int, default=-1,
                        help="Save the log every N steps (-1 = only at the end, default: -1)")
    parser.add_argument("--report", default=None, help="Path to save PDF visualization report")
    args = parser.parse_args()

    # --- Determine log directory (from --log_dir, --resume, or fallback) ---
    log_dir = args.log_dir
    if log_dir is None and args.resume:
        log_dir = os.path.dirname(args.resume)
    if log_dir is None:
        log_dir = "."

    os.makedirs(log_dir, exist_ok=True)
    text_log_path = os.path.join(log_dir, f"{args.task_id}.log")
    _setup_logging(text_log_path)

    if not MODEL_CONFIGURATIONS:
        logger.error("No model configurations defined in arc/llm/config.py")
        sys.exit(1)

    model_key = args.model_key or next(iter(MODEL_CONFIGURATIONS))
    if model_key not in MODEL_CONFIGURATIONS:
        logger.error("Model key '%s' not found. Available: %s",
                      model_key, list(MODEL_CONFIGURATIONS.keys()))
        sys.exit(1)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    task = ARCTask(folder=args.data_folder, set=args.dataset_set).load(args.task_id)
    logger.info("Loaded task %s: %d train, %d test",
                task.task_id, len(task.trainingExamples), len(task.testExamples))

    with open(args.instruction, "r") as f:
        instruction = f.read().strip()

    provider = LLMProvider(api_key=api_key)
    model_config = MODEL_CONFIGURATIONS[model_key]

    test_input = (
        task.testExamples[0]["input"]
        if task.testExamples
        else task.trainingExamples[0]["input"]
    )
    test_output = (
        task.testExamples[0]["output"]
        if task.testExamples
        else task.trainingExamples[0]["output"]
    )

    # --- Logging / resume setup ---
    execution_log = None
    if args.resume:
        execution_log = ExecutionLog.load(args.resume)
        logger.info("Loaded checkpoint with %d steps", len(execution_log.steps))
    elif args.log_dir:
        execution_log = ExecutionLog(task_id=args.task_id, instruction=instruction)

    log_path = None
    if args.log_dir:
        log_path = os.path.join(args.log_dir, f"{args.task_id}_log.json")

    logger.info("Running solver (model=%s, max_steps=%d)", model_key, args.max_steps)
    logger.info("Test input: %s", test_input)
    logger.info("Test output: %s", test_output)
    result = solve(
        instruction=instruction,
        input_grid=test_input,
        provider=provider,
        model_config=model_config,
        max_steps=args.max_steps,
        time_sleep=args.time_sleep,
        verbose=True,
        execution_log=execution_log,
        log_save_path=log_path,
        log_save_every=args.log_every,
        max_verify_retries=args.max_verify_retries,
    )

    logger.info("Terminated: %s", result["terminated"])
    logger.info("Steps executed: %d", len(result["steps"]))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result["output_grid"], f, indent=2)
        logger.info("Output written to %s", args.output)

    if execution_log is not None and log_path:
        execution_log.save(log_path)
        logger.info("Execution log saved to %s", log_path)

    if args.report:
        from arc.vis.report import generate_pdf_report
        if execution_log is not None:
            generate_pdf_report(execution_log, args.report, instruction=instruction, task=task)
            logger.info("PDF report saved to %s", args.report)
        else:
            logger.warning("--report requires --log_dir or --resume to generate a report")


if __name__ == "__main__":
    main()
