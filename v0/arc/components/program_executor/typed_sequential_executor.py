"""
Typed sequential program executor — classifies instructions by type before execution.

NOTE: System prompts and Pydantic model definitions are closed-source.
"""
import os
import json
import copy
import time
from typing import Optional, List, Dict, Any, Literal, Tuple

from pydantic import BaseModel, Field

from .sequential_executor import SequentialProgramExecutor
from arc.data.ARCTask import ARCTask
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.answer_parsing import parse_ascii_grid

# ---------------------------------------------------------------------------
# Prompts  (closed-source — only stub strings are provided)
# ---------------------------------------------------------------------------

CATEGORIZATION_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"
GENERAL_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"
DEFINITION_EXTRACTION_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"
INSTANCE_EXTRACTION_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"
ACTION_INFO_SELECTION_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"
ACTION_EXECUTION_SYSTEM_PROMPT = "[REDACTED — closed-source prompt]"

# ---------------------------------------------------------------------------
# Pydantic models  (closed-source — stubs only)
# ---------------------------------------------------------------------------


class Instance(BaseModel):
    """An instance of a defined object."""
    object_type: Optional[str] = Field(None)

    class Config:
        extra = "allow"


class InstanceList(BaseModel):
    instances: List[Instance] = Field(default_factory=list)


class Definition(BaseModel):
    """A definition of an object or collection."""
    name: str = Field(default="")
    criteria: str = Field(default="")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TypedSequentialProgramExecutor(SequentialProgramExecutor):
    """
    Typed sequential executor that categorises each instruction (DEFINITION /
    ACTION / ITERATION) and applies specialised handling for each type.
    """

    def __init__(self, model_key: str, include_train_input: bool = True,
                 include_test_input: bool = True, include_nl_program: bool = True,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 test_train_accuracy: bool = False, max_attempts: int = 3, **kwargs):
        super().__init__(model_key=model_key, include_train_input=include_train_input,
                         include_test_input=include_test_input, include_nl_program=include_nl_program,
                         max_tokens=max_tokens, temperature=temperature,
                         test_train_accuracy=test_train_accuracy, max_attempts=max_attempts)
        self.definitions: Dict[str, Dict[str, Any]] = {}
        self.objects: Dict[str, Any] = {}
        self.current_grid: Optional[List[List[int]]] = None
        self.initial_grid: Optional[List[List[int]]] = None

    def _categorize_instruction(self, instruction: str, model_config: Any) -> str:
        messages = [{"role": "system", "content": CATEGORIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Categorize this instruction:\n\n{instruction}"}]
        try:
            response = self.provider.chat(config=model_config, messages=messages)
            content, _, _ = self.provider.parse_response(response)
            category = content.strip().upper()
            valid = ["DEFINITION", "ACTION", "ITERATION"]
            if category not in valid:
                for cat in valid:
                    if cat in category:
                        category = cat
                        break
                else:
                    category = "ACTION"
            return category
        except Exception:
            return "ACTION"

    def _extract_object_definition(self, instruction, model_config):
        return {}

    def _extract_object_instances(self, instruction, current_grid, model_config, definition):
        return []

    def _execute_action(self, instruction, current_grid, model_config):
        context_parts = []
        if self.definitions:
            context_parts.append(f"**Definitions:**\n{json.dumps(self.definitions, indent=2)}")
        if self.objects:
            context_parts.append(f"**Object Instances:**\n{json.dumps(self.objects, indent=2, default=str)}")
        if self.initial_grid is not None:
            context_parts.append(f"**Initial Grid:**\n{self.grid_to_ascii(self.initial_grid)}")
        eff = current_grid if current_grid is not None else self.current_grid
        if eff is not None:
            context_parts.append(f"**Current Grid:**\n{self.grid_to_ascii(eff)}")

        user_content = f"**Instruction:**\n{instruction}\n\n" + "\n".join(context_parts)
        messages = [{"role": "system", "content": ACTION_EXECUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}]
        try:
            for attempt in range(self.max_attempts):
                try:
                    response = self.provider.chat(config=model_config, messages=messages)
                    raw_response, _, _ = self.provider.parse_response(response)
                    current_grid = parse_ascii_grid(self._extract_output_grid_ascii(raw_response))
                    break
                except Exception:
                    continue
            self.current_grid = current_grid
            return {"success": True, "updated_grid": current_grid, "raw_response": raw_response}
        except Exception as e:
            return {"success": False, "updated_grid": self.current_grid, "error": str(e)}

    def _execute_instruction(self, instruction, current_grid, category, model_config=None):
        if category == "DEFINITION":
            defn = self._extract_object_definition(instruction, model_config)
            self._extract_object_instances(instruction, current_grid, model_config, defn)
        elif category in ("ACTION", "ITERATION"):
            return self._execute_action(instruction, current_grid, model_config)

    def execute_single(self, instructions: str, input_grid: List[List[int]]) -> dict:
        instructions_dict = self.preprocess_instructions(instructions)
        universal_transformation_rule = instructions_dict["universal_transformation_rule"]

        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens:
            model_config.max_tokens = self.max_tokens
        if self.temperature:
            model_config.temperature = self.temperature

        self.initial_grid = copy.deepcopy(input_grid)
        self.current_grid = copy.deepcopy(input_grid)
        self.definitions = {}
        self.objects = {}

        step_logs: List[Dict] = []
        for step_idx, rule_item in enumerate(universal_transformation_rule):
            category = self._categorize_instruction(rule_item, model_config)
            time.sleep(5)
            execution_result = self._execute_instruction(rule_item, self.current_grid, category, model_config)
            time.sleep(5)
            step_logs.append({"step_index": step_idx, "instruction": rule_item, "category": category, "execution_result": execution_result, "current_grid": self.current_grid})

        return {"predicted_grid": self.current_grid, "model_used": self.model_key, "steps": step_logs}
