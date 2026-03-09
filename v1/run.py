"""
Entrypoint for the v1 interpreter-executor ARC solver.

Usage:
    python run.py --task_id <TASK_ID> --data_folder <PATH> [--model_key <KEY>] [--max_steps 500]

The solver requires:
  1. ARC task data files (challenges + solutions JSON) in --data_folder.
  2. An OPENROUTER_API_KEY environment variable (or modify LLMProvider).
  3. At least one entry in MODEL_CONFIGURATIONS (see arc/components/llm/config.py).
"""
import argparse
import os
import sys
import json
import copy

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from solver.runner import solve


def main():
    parser = argparse.ArgumentParser(description="Run the v1 interpreter-executor ARC solver.")
    parser.add_argument("--task_id", required=True, help="ARC task ID to solve")
    parser.add_argument("--data_folder", required=True, help="Path to ARC data folder")
    parser.add_argument("--dataset_set", default="training", help="Dataset split (default: training)")
    parser.add_argument("--model_key", default=None, help="Model key from MODEL_CONFIGURATIONS")
    parser.add_argument("--instruction", default=None, help="Path to a text file with the NL program")
    parser.add_argument("--max_steps", type=int, default=500, help="Max execution steps (default: 500)")
    parser.add_argument("--time_sleep", type=int, default=1, help="Sleep between LLM calls in seconds")
    parser.add_argument("--output", default=None, help="Path to write the output grid JSON")
    args = parser.parse_args()

    if not MODEL_CONFIGURATIONS:
        print("ERROR: No model configurations defined.")
        print("Add your model entries to arc/components/llm/config.py")
        sys.exit(1)

    model_key = args.model_key or next(iter(MODEL_CONFIGURATIONS))
    if model_key not in MODEL_CONFIGURATIONS:
        print(f"ERROR: Model key '{model_key}' not found. Available: {list(MODEL_CONFIGURATIONS.keys())}")
        sys.exit(1)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    task = ARCTask(folder=args.data_folder, set=args.dataset_set).load(args.task_id)
    print(f"Loaded task {task.task_id}: {len(task.trainingExamples)} train, {len(task.testExamples)} test")

    if args.instruction:
        with open(args.instruction, "r") as f:
            instruction = f.read().strip()
    else:
        print("No --instruction file provided. Supply a text file with the NL program.")
        sys.exit(1)

    provider = LLMProvider(api_key=api_key)
    model_config = MODEL_CONFIGURATIONS[model_key]

    test_input = task.testExamples[0]["input"] if task.testExamples else task.trainingExamples[0]["input"]

    print(f"\nRunning solver (model={model_key}, max_steps={args.max_steps})...")
    result = solve(
        instruction=instruction,
        input_grid=test_input,
        provider=provider,
        model_config=model_config,
        max_steps=args.max_steps,
        time_sleep=args.time_sleep,
        verbose=True,
    )

    print(f"\nTerminated: {result['terminated']}")
    print(f"Steps executed: {len(result['steps'])}")
    print(f"Output grid: {result['output_grid']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result["output_grid"], f, indent=2)
        print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
