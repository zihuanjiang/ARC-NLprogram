# program_executor/llm_executor.py
import os
from typing import Optional

from .base import ProgramExecutor  
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.answer_parsing import parse_ascii_grid
from arc.utils.prompt_builder import build_executor_prompt 
from arc.solvers.registry import ProgramExecutorRegistry

@ProgramExecutorRegistry.register("llm")
class LLMProgramExecutor(ProgramExecutor):
    """
    Implementation of ProgramExecutor that uses an LLM to execute natural language instructions.
    """
    def __init__(self, model_key: str, include_train_input: bool = True, include_test_input: bool = True, include_nl_program: bool = True, include_abstraction: bool = True, max_tokens: Optional[int] = None, temperature: Optional[float] = None, max_attempts: int = 3):

        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        
        self.model_key = model_key
        self.include_train_input = include_train_input
        self.include_test_input = include_test_input
        self.include_nl_program = include_nl_program
        self.include_abstraction = include_abstraction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def execute(
        self,
        task: ARCTask,
        instructions: str,
        abstractions: Optional[dict] = None,
    ) -> dict:

        print(f"Generating output grid for task {task.task_id} with model {self.model_key}...")
 
        system_prompt, grid_prompt = build_executor_prompt(task,
                                                           instructions,
                                                           abstractions if self.include_abstraction else None,
                                                           include_train_input=self.include_train_input,
                                                           include_test_input=self.include_test_input,
                                                           include_nl_program=self.include_nl_program)
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": grid_prompt}
        ]

        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens: model_config.max_tokens = self.max_tokens
        if self.temperature: model_config.temperature = self.temperature
        
        # Retry logic for missing grid/content
        max_attempts = self.max_attempts
        
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"Warning: No grid found in output. Retrying... (Attempt {attempt+1}/{max_attempts})")

            response = self.provider.chat(config=model_config, messages=messages)

            raw_response, reasoning, usage_stats = self.provider.parse_response(response)
            predicted_grid = parse_ascii_grid(raw_response)
            
            if predicted_grid and len(predicted_grid) > 0:
                print("Execution complete.")
                return {
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
        
        print(f"Error: Failed to parse grid from executor output after {max_attempts} attempts.")
        return {
            "predicted_grid": None,
            "reasoning": reasoning if 'reasoning' in locals() else [],
            "raw_response": raw_response if 'raw_response' in locals() else "",
            "model_used": self.model_key,
            "prompt_trace": {
                "system": system_prompt,
                "user": grid_prompt
            },
            "usage": usage_stats if 'usage_stats' in locals() else None,
            "status": "failed_no_grid"
        }