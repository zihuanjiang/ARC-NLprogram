"""
Sequential program generator — produces structured NL programs with
training-pair analysis and universal transformation rules.

NOTE: System prompt contents are closed-source.
"""
import os
from typing import Optional

from .base import ProgramGenerator
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.plotting import base64_from_grid

SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"


class SequentialProgramGenerator(ProgramGenerator):
    """Generates structured NL programs from ARC task examples."""

    def __init__(self, model_key: str, include_train_input: bool = True,
                 include_test_input: bool = False, include_abstraction: bool = True,
                 include_image: bool = False, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, max_attempts: int = 3):
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found.")
        self.model_key = model_key
        self.include_test_input = include_test_input
        self.include_train_input = include_train_input
        self.include_abstraction = include_abstraction
        self.include_image = include_image
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def grid_to_ascii(self, grid: list[list[int]]) -> str:
        return "\n".join("".join(str(int(x)) for x in row) for row in grid)

    def build_user_prompt(self, task: ARCTask) -> str:
        prompt = ""
        for i, ex in enumerate(task.trainingExamples):
            prompt += f"Training pair {i+1}:\n"
            prompt += f"Input Grid:\n{self.grid_to_ascii(ex['input'])}\n"
            prompt += f"Output Grid:\n{self.grid_to_ascii(ex['output'])}\n"
        return prompt

    def generate(self, task: ARCTask, abstractions: Optional[dict] = None) -> dict:
        print(f"Generating instructions for task {task.task_id} with model {self.model_key}...")

        if self.include_image:
            raise NotImplementedError("Image generation not supported for sequential generation.")

        system_prompt = SYSTEM_PROMPT
        user_prompt = self.build_user_prompt(task)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        for attempt in range(self.max_attempts):
            if attempt > 0:
                print(f"Warning: Empty instructions. Retrying ({attempt+1}/{self.max_attempts})")
            response = self.provider.chat(config=MODEL_CONFIGURATIONS[self.model_key], messages=messages)
            content, reasoning, usage_stats = self.provider.parse_response(response)
            if content and content.strip():
                return {"instructions": content, "reasoning": reasoning, "raw_response": content, "model_used": self.model_key, "prompt_trace": {"system": system_prompt, "user": user_prompt}, "usage": usage_stats}

        return {"instructions": "", "reasoning": reasoning if 'reasoning' in dir() else [], "raw_response": content if 'content' in dir() else "", "model_used": self.model_key, "prompt_trace": {"system": system_prompt, "user": user_prompt}, "usage": usage_stats if 'usage_stats' in dir() else None, "status": "failed_empty_response"}
