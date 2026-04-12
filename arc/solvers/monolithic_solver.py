import os

from arc.components.abstraction import get_registry
from arc.data.ARCTask import ARCTask
from arc.components.program_generator.llm_generator import LLMProgramGenerator
from arc.components.program_executor.llm_executor_v2 import LLMProgramExecutorV2
from arc.evaluator.pointwise_acc import Pointwise_acc
from arc.evaluator.success_rate import Success_rate
from arc.evaluator.weighted_f1 import Weighted_f1
from arc.evaluator.arga_weighted_error import ARGAWeightedError
from arc.utils.solver_utils import load_config_with_updates

from arc.solvers.base import Solver


class MonolithicSolver(Solver):
    """
    A solver that loads its base configuration from a YAML file
    and allows configuration updates via arguments.
    """
    def __init__(self, base_config_path="config/monolithic_solver.yaml", config_updates=None):
        """
        Initialize the MonolithicSolver.
        
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
        abstraction_config = config.get('abstraction_config', {}).copy()
        program_gen_config = config.get('program_gen_config', {})
        executor_config = config.get('executor_config', {})
        
        # Get abstractor type from config (default to 'llm' for backward compatibility)
        abstractor_type = abstraction_config.pop('abstractor_type', 'llm')
        
        # Create abstractor using registry
        registry = get_registry()
        if not registry.is_registered(abstractor_type):
            raise ValueError(
                f"Abstractor type '{abstractor_type}' is not registered. "
                f"Available types: {registry.list_registered()}"
            )
        
        self.abstractor = registry.create(abstractor_type, abstraction_config)
        self.program_generator = LLMProgramGenerator(**program_gen_config)
        
        # Always use LLMProgramExecutorV2
        self.program_executor = LLMProgramExecutorV2(**executor_config)
        
        self.metrics = [Pointwise_acc(), Success_rate(), Weighted_f1(), ARGAWeightedError()]

    def solve(self, task: ARCTask, experiment_name: str = "default_experiment") -> dict:

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

        # Generate object abstraction only if needed
        abstractions = None
        if self.program_generator.include_abstraction or self.program_executor.include_abstraction:
            abstractions, abstractor_usage = self.abstractor.abstract(task)
            aggregate_usage(abstractor_usage)
        else:
            print("Skipping abstraction generation (not needed for this experiment)")

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
        
        print(f"Generated Instructions:\n{instructions}\n")

        # Execute the program
        executor_result = self.program_executor.execute(
            task=task,
            instructions=instructions,
            abstractions=abstractions
        )
        aggregate_usage(executor_result.get("total_usage", executor_result.get("usage")))

        # Restructure execution output to match standardized format used by all solvers
        # Different executors may return test execution output under different keys (e.g., "execution_output" or "test_execution_output")
        # We standardize to include test_execution_output as a nested key, with train outputs at the top level
        test_execution_output_raw = executor_result.get("test_execution_output") or executor_result.get("execution_output", {})
        
        # Extract train outputs if available
        predicted_train_grids = executor_result.get("predicted_train_grids", [])
        train_execution_outputs = executor_result.get("train_execution_outputs", [])
        
        # Create standardized execution_output structure
        # This ensures consistent output format across all solver types, regardless of underlying executor implementation
        execution_output = {
            "test_execution_output": test_execution_output_raw,
        }
        if predicted_train_grids:
            execution_output["predicted_train_grids"] = predicted_train_grids
        if train_execution_outputs:
            execution_output["train_execution_outputs"] = train_execution_outputs

        # Evaluate the result
        # Extract predicted_grid from test_execution_output first (most reliable source),
        # then fall back to top-level executor_result for backward compatibility
        predicted_grid = test_execution_output_raw.get("predicted_grid") if test_execution_output_raw else None
        if predicted_grid is None:
            predicted_grid = executor_result.get("predicted_grid")
        predicted_grids = [predicted_grid] if predicted_grid else []
        
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
        if predicted_train_grids:
            print(f"Evaluating {len(predicted_train_grids)} train predictions...")
            
            # Create a temporary task structure for train evaluation
            # We need to evaluate each train prediction against its corresponding ground truth
            train_evaluation_results = []
            for train_idx, train_predicted_grid in enumerate(predicted_train_grids):
                train_ex = task.trainingExamples[train_idx]
                ground_truth = [train_ex["output"]]
                predicted = [train_predicted_grid]
                
                # Create a temporary task-like structure for evaluation
                # Note: metrics expect task.testExamples, so we'll create a minimal structure
                temp_task = type('TempTask', (), {
                    'testExamples': [{'output': ground_truth[0]}]
                })()
                
                train_eval = {}
                for metric in self.metrics:
                    scores = metric.run(temp_task, predicted)
                    train_eval[metric.name] = scores[0] if scores else None
                
                if train_evaluation_results is None:
                    train_evaluation_results = []
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
            "predicted_grid": predicted_grid,
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
