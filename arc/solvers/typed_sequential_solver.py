import json
import os
from typing import Optional

from arc.data.ARCTask import ARCTask
from arc.components.program_generator.sequential_generator import SequentialProgramGenerator
from arc.components.program_executor.sequential_executor import SequentialProgramExecutor
from arc.components.program_executor.typed_sequential_executor import TypedSequentialProgramExecutor
from arc.evaluator.pointwise_acc import Pointwise_acc
from arc.evaluator.success_rate import Success_rate
from arc.evaluator.weighted_f1 import Weighted_f1
from arc.evaluator.arga_weighted_error import ARGAWeightedError
from arc.utils.solver_utils import load_config_with_updates

from arc.solvers.base import Solver


class TypedSequentialSolver(Solver):
    """
    A solver that implements a sequential pipeline:
    task -> generate NL program -> execute the program.
    """
    def __init__(self, base_config_path="config/typed_sequential_solver.yaml", config_updates=None):
        """
        Initialize the TypedSequentialSolver.
        
        Args:
            base_config_path: Path to the base config YAML file (relative to solvers directory).
                             If None and config_updates is provided, uses config_updates directly.
            config_updates: Dictionary of config updates to apply to the base config.
                           If base_config_path is None, this is used as the full config.
        """
        # Load and merge config
        if base_config_path is None and config_updates is not None:
            # Use config_updates directly as the config
            config = config_updates
        else:
            # Resolve config path relative to this solver's directory
            solver_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(solver_dir, base_config_path)
            config = load_config_with_updates(config_path, config_updates)
        
        # Extract solver config sections
        program_gen_config = config.get('program_gen_config', {})
        executor_config = config.get('executor_config', {})
        # Flag to control whether to use the typed executor or the original sequential executor
        self.use_typed_executor = config.get('use_typed_executor', False)
        # When True, instructions are read from previous runs (instructions_dir/task_id/solution_log/generation_output.json)
        # instead of being generated. Default False to keep current behavior.
        self.read_instructions_from_previous_runs = config.get('read_instructions_from_previous_runs', False)
        self.instructions_dir = config.get('instructions_dir', None)
        
        # Create program generator and executor
        # Note: Abstraction stage is skipped, so we don't create an abstractor
        # Use SequentialProgramGenerator as the generator backend
        self.program_generator = SequentialProgramGenerator(**program_gen_config)
        
        # Choose executor based on flag:
        # - If use_typed_executor is True, use TypedSequentialProgramExecutor
        # - Otherwise, fall back to the original SequentialProgramExecutor
        if self.use_typed_executor:
            print("Using TypedSequentialProgramExecutor for execution.")
            self.program_executor = TypedSequentialProgramExecutor(**executor_config)
        else:
            print("Using SequentialProgramExecutor for execution.")
            self.program_executor = SequentialProgramExecutor(**executor_config)
        
        self.metrics = [Pointwise_acc(), Success_rate(), Weighted_f1(), ARGAWeightedError()]

    def _load_instructions_from_previous_run(self, task_id: str) -> Optional[str]:
        """
        Load instructions from a previous run: {instructions_dir}/{task_id}/solution_log/generation_output.json.
        Returns the 'instructions' value if found and non-empty, else None.
        """
        path = os.path.join(self.instructions_dir, task_id, 'solution_log', 'generation_output.json')
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            instr = data.get('instructions')
            if isinstance(instr, str) and instr.strip():
                return instr
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def solve(
        self, 
        task: ARCTask, 
        experiment_name: str = "default_experiment",
        instructions: Optional[str] = None
    ) -> dict:
        """
        Solve an ARC task using the sequential pipeline.
        
        Args:
            task: The ARC task to solve
            experiment_name: Name of the experiment (for logging)
            instructions: Optional pre-generated instructions for mid-point entry.
                         If provided, skips generation and goes directly to execution.
        
        Returns:
            Dictionary containing the solution and evaluation results
        """
        print(f"\n--- Solving Task {task.task_id} ---")
        
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        def aggregate_usage(usage_list):
            if not isinstance(usage_list, list):
                usage_list = [usage_list]
            for u in usage_list:
                if u:
                    total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += u.get("completion_tokens", 0)
                    total_usage["total_tokens"] += u.get("total_tokens", 0)

        # Skip abstraction stage (as per requirements)
        abstractions = None
        print("Skipping abstraction generation (not used in this solver)")

        # Optionally load instructions from a previous run (experiments/sequ/.../solution_log/generation_output.json)
        instructions_source = None
        if self.read_instructions_from_previous_runs:
            loaded = self._load_instructions_from_previous_run(task.task_id)
            if loaded is not None:
                instructions = loaded
                instructions_source = "read_from_previous_run"
                print("Using instructions read from previous run (skipping generation)")
            else:
                print(f"No previous instructions found for {task.task_id}; falling back to generation.")

        # Generate or use provided instructions
        if instructions is not None:
            generation_output = {
                "instructions": instructions,
                "usage": None,
                "status": instructions_source
            }
        else:
            # Generate the natural language program
            generation_output = self.program_generator.generate(
                task=task,
                abstractions=abstractions
            )
            aggregate_usage(generation_output.get("usage"))

        instructions = generation_output.get('instructions', None)

        if not instructions:
            print("Generation stage failed to produce instructions. Aborting solve.")
            return {
                "status": "failed_at_generation",
                "generation_output": generation_output,
                "predicted_grid": None,
                "total_usage": total_usage
            }
        
        print(f"Using Instructions:\n{instructions}\n")

        # Execute the program
        execution_output = self.program_executor.execute(
            task=task,
            instructions=instructions,
        )
        aggregate_usage(execution_output.get("total_usage"))

        # Evaluate the result
        predicted_grids = [execution_output['predicted_grid']] if execution_output.get('predicted_grid') else []
        
        # Initialize list of dicts for each test example
        evaluation_results = [{} for _ in range(len(task.testExamples))]
        
        # Evaluate test predictions
        if predicted_grids:
            for metric in self.metrics:
                scores = metric.run(task, predicted_grids)
                for i, score in enumerate(scores):
                    evaluation_results[i][metric.name] = score
        
        # Evaluate train predictions if test_train_accuracy is enabled
        train_evaluation_results = None
        if execution_output.get("predicted_train_grids") is not None:
            predicted_train_grids = execution_output["predicted_train_grids"]
            print(f"Evaluating {len(predicted_train_grids)} train predictions...")
            
            # Create a temporary task structure for train evaluation
            train_evaluation_results = []
            for train_idx, predicted_grid in enumerate(predicted_train_grids):
                train_ex = task.trainingExamples[train_idx]
                ground_truth = [train_ex["output"]]
                predicted = [predicted_grid] if predicted_grid is not None else [None]
                
                # Create a temporary task-like structure for evaluation
                temp_task = type('TempTask', (), {
                    'testExamples': [{'output': ground_truth[0]}]
                })()
                
                train_eval = {}
                for metric in self.metrics:
                    scores = metric.run(temp_task, predicted)
                    train_eval[metric.name] = scores[0] if scores else None
                
                train_evaluation_results.append(train_eval)
        
        task_evaluation = evaluation_results
        print(f"Task evaluation result (test): {task_evaluation}")
        if train_evaluation_results:
            print(f"Task evaluation result (train): {train_evaluation_results}")
        print(f"Total Tokens: {total_usage['total_tokens']}")

        final_log = {
            "status": "success",
            "task_id": task.task_id,
            "grid_abstractions": abstractions,
            "predicted_grid": predicted_grids[0] if predicted_grids else None,
            "generation_output": generation_output,
            "execution_output": execution_output,
            "task_evaluation": task_evaluation,
            "train_evaluation": train_evaluation_results,
            "total_usage": total_usage,
            "debug": {
                "full_generation_output": generation_output,
                "full_execution_output": execution_output,
                "abstractions": abstractions
            }
        }

        print(f"--- Finished Task {task.task_id} ---")
        
        return final_log
