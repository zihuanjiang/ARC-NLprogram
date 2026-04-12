"""
LLM-based judge for evaluating natural language programs against ARC tasks.

NOTE: Prompt contents are closed-source.
"""
import os
import json
from typing import Optional
from .base import Judge
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.plotting import base64_from_grid

JUDGE_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"

from arc.solvers.registry import JudgeRegistry


@JudgeRegistry.register("llm")
class LLMJudge(Judge):
    """
    Implementation of Judge that uses an LLM to evaluate instructions.
    """

    def __init__(self, model_key: str, include_image: bool = False, max_tokens: Optional[int] = None, temperature: Optional[float] = 0.0):
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        self.model_key = model_key
        self.include_image = include_image
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def judge(self, task: ARCTask, instructions: str) -> dict:
        """Evaluates the instructions against the task's training examples using an LLM."""
        print(f"Judging instructions for task {task.task_id} with model {self.model_key}...")

        user_content = []
        user_content.append({"type": "text", "text": f"Proposed Natural Language Program:\n{instructions}\n\nVerify this program against the following Training Examples:\n"})

        for i, ex in enumerate(task.trainingExamples):
            user_content.append({"type": "text", "text": f"\n--- Training Example {i+1} ---\n"})
            user_content.append({"type": "text", "text": f"Input Grid:\n{ex['input']}\n"})
            if self.include_image:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_from_grid(ex['input'])}"}})
            user_content.append({"type": "text", "text": f"Output Grid:\n{ex['output']}\n"})
            if self.include_image:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_from_grid(ex['output'])}"}})

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens:
            model_config.max_tokens = self.max_tokens
        if self.temperature is not None:
            model_config.temperature = self.temperature
        if model_config.extra_body is None:
            model_config.extra_body = {}
        model_config.extra_body.update({"response_format": {"type": "json_object"}})

        response = self.provider.chat(config=model_config, messages=messages)
        content, _, usage_stats = self.provider.parse_response(response)

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"analysis": "Failed to parse JSON response.", "verdict": "Error", "confidence": 0.0, "raw_response": content}

        result["usage"] = usage_stats
        return result
