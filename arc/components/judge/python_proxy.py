"""
Python-proxy judge: translates NL instructions to Python and executes them.

NOTE: Prompt contents are closed-source.
"""
import os
import json
import traceback
from typing import Optional, List, Any
from .base import Judge
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.solvers.registry import JudgeRegistry

PROXY_JUDGE_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"


@JudgeRegistry.register("python_proxy")
class PythonProxyJudge(Judge):
    """Translates instructions to Python code and executes to verify correctness."""

    def __init__(self, model_key: str, max_tokens: Optional[int] = None, temperature: Optional[float] = 0.0):
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        self.model_key = model_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def judge(self, task: ARCTask, instructions: str) -> dict:
        code = self._generate_code(instructions)
        verdict = "Correct"
        analysis = []

        try:
            local_scope = {}
            exec(code, local_scope)
            solve_fn = local_scope.get('solve')

            if not solve_fn:
                return {"verdict": "Error", "confidence": 0.0, "analysis": "Failed to generate a 'solve' function.", "generated_code": code}

            train_outputs = []
            for i, ex in enumerate(task.trainingExamples):
                try:
                    import copy
                    actual_output = solve_fn(copy.deepcopy(ex['input']))
                    train_outputs.append(actual_output)
                    if actual_output == ex['output']:
                        analysis.append(f"Example {i+1}: Pass")
                    else:
                        verdict = "Incorrect"
                        analysis.append(f"Example {i+1}: Fail.")
                except Exception as e:
                    verdict = "Incorrect"
                    train_outputs.append(None)
                    analysis.append(f"Example {i+1}: Error: {str(e)}")

            test_outputs = []
            for i, ex in enumerate(task.testExamples):
                try:
                    import copy
                    test_outputs.append(solve_fn(copy.deepcopy(ex['input'])))
                except Exception as e:
                    test_outputs.append(None)

        except Exception as e:
            return {"verdict": "Error", "confidence": 0.0, "analysis": f"Code execution failed: {str(e)}", "generated_code": code}

        return {"verdict": verdict, "confidence": 1.0 if verdict == "Correct" else 0.0, "analysis": "\n".join(analysis), "generated_code": code, "train_outputs": train_outputs, "test_outputs": test_outputs}

    def _generate_code(self, instructions: str) -> str:
        messages = [
            {"role": "system", "content": PROXY_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Instructions:\n{instructions}"}
        ]
        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens:
            model_config.max_tokens = self.max_tokens
        if self.temperature is not None:
            model_config.temperature = self.temperature
        if model_config.extra_body and "response_format" in model_config.extra_body:
            del model_config.extra_body["response_format"]

        response = self.provider.chat(config=model_config, messages=messages)
        content, _, _ = self.provider.parse_response(response)

        content = content.strip()
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
