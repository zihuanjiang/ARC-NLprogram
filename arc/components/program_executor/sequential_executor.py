"""
Sequential program executor — executes NL instructions step-by-step via LLM.

NOTE: System prompt contents are closed-source.
"""
import os
import re
from typing import Optional, List, Dict, Tuple

from .base import ProgramExecutor
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.answer_parsing import parse_ascii_grid

EXECUTOR_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"


class SequentialProgramExecutor(ProgramExecutor):
    """
    Executes NL instructions sequentially, feeding each step into the LLM
    and maintaining a multi-turn conversation with the current grid state.
    """

    def __init__(self, model_key: str, include_train_input: bool = True,
                 include_test_input: bool = True, include_nl_program: bool = True,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 test_train_accuracy: bool = False, max_attempts: int = 3):
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        self.model_key = model_key
        self.include_train_input = include_train_input
        self.include_test_input = include_test_input
        self.include_nl_program = include_nl_program
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.test_train_accuracy = test_train_accuracy
        self.max_attempts = max_attempts
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
        self.history = []

    def grid_to_ascii(self, grid: List[List[int]]) -> str:
        return "\n".join("".join(str(int(x)) for x in row) for row in grid)

    def _extract_output_grid_ascii(self, raw_response: str) -> Optional[str]:
        marker = "**Output grid:**"
        idx = raw_response.find(marker)
        if idx == -1:
            return None
        after = raw_response[idx + len(marker):].lstrip("\n\r ")
        lines = []
        for line in after.splitlines():
            if not line.strip():
                break
            lines.append(line.rstrip())
        return "\n".join(lines) if lines else None

    def preprocess_instructions(self, instructions: str) -> Dict[str, List[str]]:
        result = {"training_pairs_analysis": [], "universal_transformation_rule": []}
        training_pattern = r'\*\*Training pairs analysis:\*\*'
        rule_pattern = r'\*\*Universal transformation rule:\*\*'
        training_match = re.search(training_pattern, instructions, re.IGNORECASE)
        rule_match = re.search(rule_pattern, instructions, re.IGNORECASE)

        if not training_match or not rule_match:
            training_pattern = r'Training pairs analysis:'
            rule_pattern = r'Universal transformation rule:'
            training_match = re.search(training_pattern, instructions, re.IGNORECASE)
            rule_match = re.search(rule_pattern, instructions, re.IGNORECASE)

        if not training_match or not rule_match:
            return result

        training_section = instructions[training_match.end():rule_match.start()].strip()
        rule_section = instructions[rule_match.end():].strip()

        def extract_numbered_items(text: str) -> List[str]:
            items: List[str] = []
            pattern = re.compile(r'^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)', re.MULTILINE | re.DOTALL)
            for match in pattern.finditer(text):
                items.append(f"{match.group(1)}. {match.group(2).strip()}")
            return items

        result["training_pairs_analysis"] = extract_numbered_items(training_section)
        result["universal_transformation_rule"] = extract_numbered_items(rule_section)
        return result

    def execute_single(self, instructions: str, input_grid: List[List[int]]) -> dict:
        instructions_dict = self.preprocess_instructions(instructions)
        universal_transformation_rule = instructions_dict["universal_transformation_rule"]

        system_prompt = EXECUTOR_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Initial input grid: {self.grid_to_ascii(input_grid)}"}
        ]

        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens:
            model_config.max_tokens = self.max_tokens
        if self.temperature:
            model_config.temperature = self.temperature

        step_logs: List[Dict] = []
        current_grid = None

        for step_idx, rule_item in enumerate(universal_transformation_rule):
            messages.append({"role": "user", "content": rule_item})
            raw_response = ""
            reasoning = ""
            usage_stats: Dict = {}

            for attempt in range(self.max_attempts):
                if attempt > 0:
                    print(f"Warning: Empty response. Retrying ({attempt+1}/{self.max_attempts})")
                try:
                    response = self.provider.chat(config=model_config, messages=messages)
                    raw_response, reasoning, usage_stats = self.provider.parse_response(response)
                    messages.append({"role": "assistant", "content": raw_response})
                    current_grid = parse_ascii_grid(self._extract_output_grid_ascii(raw_response))
                    break
                except Exception as e:
                    print(f"Error: {e}")

            step_logs.append({"step_index": step_idx, "instruction": rule_item, "raw_response": raw_response, "reasoning": reasoning, "usage": usage_stats, "output_grid": current_grid})

        return {"predicted_grid": current_grid, "model_used": self.model_key, "steps": step_logs, "prompt_trace": messages}

    def execute(self, task: ARCTask, instructions: str) -> dict:
        print(f"Executing instructions for task {task.task_id} with model {self.model_key}...")
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def aggregate_usage(usage_stats):
            if usage_stats:
                total_usage["prompt_tokens"] += usage_stats.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage_stats.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage_stats.get("total_tokens", 0)

        predicted_train_grids: List = []
        train_execution_outputs: List[Dict] = []

        if self.test_train_accuracy:
            for train_idx, train_ex in enumerate(task.trainingExamples):
                train_exec = self.execute_single(instructions=instructions, input_grid=train_ex["input"])
                for step in train_exec.get("steps", []):
                    aggregate_usage(step.get("usage"))
                predicted_train_grids.append(train_exec["predicted_grid"])
                train_exec["train_example_index"] = train_idx
                train_execution_outputs.append(train_exec)

        predicted_grid = None
        test_exec = None
        if self.include_test_input:
            test_exec = self.execute_single(instructions=instructions, input_grid=task.testExamples[0]["input"])
            for step in test_exec.get("steps", []):
                aggregate_usage(step.get("usage"))
            predicted_grid = test_exec.get("predicted_grid")

        result: Dict = {"predicted_grid": predicted_grid, "model_used": self.model_key, "total_usage": total_usage, "test_execution_output": test_exec}
        if self.test_train_accuracy:
            result["predicted_train_grids"] = predicted_train_grids
            result["train_execution_outputs"] = train_execution_outputs
        return result
