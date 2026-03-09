# program_executor/llm_executor_v2.py
import os
from typing import Optional, List

from .base import ProgramExecutor  
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.answer_parsing import parse_ascii_grid
from arc.utils.prompt_builder import build_executor_prompt_v2


class LLMProgramExecutorV2(ProgramExecutor):
    """
    Advanced implementation of ProgramExecutor that uses an LLM to execute natural language instructions.
    Supports generating outputs for both test inputs and training inputs (for train accuracy evaluation).
    """
    def __init__(
        self, 
        model_key: str, 
        include_train_input: bool = True, 
        include_test_input: bool = True, 
        include_nl_program: bool = True, 
        include_abstraction: bool = True, 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None,
        test_train_accuracy: bool = False
    ):
        """
        Initialize the LLMProgramExecutorV2.
        
        Args:
            model_key: Key for the model configuration
            include_train_input: Whether to include training examples in prompt (for reference)
            include_test_input: Whether to generate output for test input
            include_nl_program: Whether to include the natural language program
            include_abstraction: Whether to include abstractions in prompt
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            test_train_accuracy: If True, also generate outputs for training examples (without showing correct answers)
        """
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        
        self.model_key = model_key
        self.include_train_input = include_train_input
        self.include_test_input = include_test_input
        self.include_nl_program = include_nl_program
        self.include_abstraction = include_abstraction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.test_train_accuracy = test_train_accuracy
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def execute(
        self,
        task: ARCTask,
        instructions: str,
        abstractions: Optional[dict] = None,
    ) -> dict:
        """
        Execute the instructions on the task.
        
        If test_train_accuracy is True, generates outputs for all training examples.
        Otherwise, generates output only for test input (standard behavior).
        
        Returns:
            dict containing:
                - predicted_grid: Output grid for test input (if include_test_input is True)
                - predicted_train_grids: List of predicted grids for training examples (if test_train_accuracy is True)
                - train_evaluations: List of evaluation results for each train prediction (if test_train_accuracy is True)
                - Other metadata (reasoning, raw_response, usage, etc.)
        """
        print(f"Generating output grid(s) for task {task.task_id} with model {self.model_key}...")
        print(f"Test train accuracy: {self.test_train_accuracy}")
        
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        def aggregate_usage(usage_stats):
            if usage_stats:
                total_usage["prompt_tokens"] += usage_stats.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage_stats.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage_stats.get("total_tokens", 0)
        
        predicted_train_grids = []
        train_execution_outputs = []
        
        # Generate outputs for training examples if test_train_accuracy is enabled
        if self.test_train_accuracy:
            print(f"Generating outputs for {len(task.trainingExamples)} training examples...")
            for train_idx in range(len(task.trainingExamples)):
                print(f"  Processing training example {train_idx + 1}/{len(task.trainingExamples)}...")
                
                system_prompt, grid_prompt = build_executor_prompt_v2(
                    task,
                    instructions,
                    abstractions if self.include_abstraction else None,
                    include_train_input=self.include_train_input,
                    include_test_input=False,  # Don't include test input when testing train accuracy
                    include_nl_program=self.include_nl_program,
                    test_train_accuracy=True,
                    train_example_index=train_idx
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": grid_prompt}
                ]
                
                model_config = MODEL_CONFIGURATIONS[self.model_key]
                if self.max_tokens: 
                    model_config.max_tokens = self.max_tokens
                if self.temperature: 
                    model_config.temperature = self.temperature
                
                response = self.provider.chat(config=model_config, messages=messages)
                raw_response, reasoning, usage_stats = self.provider.parse_response(response)
                aggregate_usage(usage_stats)
                
                predicted_grid = parse_ascii_grid(raw_response)
                predicted_train_grids.append(predicted_grid)
                
                train_execution_outputs.append({
                    "predicted_grid": predicted_grid,
                    "reasoning": reasoning,
                    "raw_response": raw_response,
                    "train_example_index": train_idx
                })
        
        # Generate output for test input if requested
        predicted_grid = None
        test_execution_output = None
        if self.include_test_input:
            print("Generating output for test input...")
            
            system_prompt, grid_prompt = build_executor_prompt_v2(
                task,
                instructions,
                abstractions if self.include_abstraction else None,
                include_train_input=self.include_train_input,
                include_test_input=True,
                include_nl_program=self.include_nl_program,
                test_train_accuracy=False,
                train_example_index=None
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": grid_prompt}
            ]
            
            model_config = MODEL_CONFIGURATIONS[self.model_key]
            if self.max_tokens: 
                model_config.max_tokens = self.max_tokens
            if self.temperature: 
                model_config.temperature = self.temperature
            
            response = self.provider.chat(config=model_config, messages=messages)
            raw_response, reasoning, usage_stats = self.provider.parse_response(response)
            aggregate_usage(usage_stats)
            
            predicted_grid = parse_ascii_grid(raw_response)
            
            test_execution_output = {
                "predicted_grid": predicted_grid,
                "reasoning": reasoning,
                "raw_response": raw_response,
                "model_used": self.model_key,
                "prompt_trace": {
                    "system": system_prompt,
                    "user": grid_prompt
                },
                "usage": usage_stats
            }
        
        print("Execution complete.")
        
        result = {
            "predicted_grid": predicted_grid,  # For backward compatibility (test input)
            "model_used": self.model_key,
            "total_usage": total_usage
        }
        
        # Add test execution output if available
        if test_execution_output:
            result["execution_output"] = test_execution_output
            result["reasoning"] = test_execution_output["reasoning"]
            result["raw_response"] = test_execution_output["raw_response"]
            result["prompt_trace"] = test_execution_output["prompt_trace"]
        
        # Add train execution outputs if test_train_accuracy is enabled
        if self.test_train_accuracy:
            result["predicted_train_grids"] = predicted_train_grids
            result["train_execution_outputs"] = train_execution_outputs
        
        return result
