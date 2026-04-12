"""
Structured execution logger.

Stores the full execution state at every step so that runs can be
inspected, resumed, or modified after the fact.
"""
import copy
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Snapshot of a single execution step."""

    step_number: int
    current_instruction: str
    pc_instruction: str

    # State *before* this step executed
    pre_memory: Dict[str, Any]
    pre_state: Dict[str, Any]

    # Results produced by this step
    interpreter_action: str = ""
    interpreter_thought: str = ""
    executor_result: Dict[str, Any] = field(default_factory=dict)
    pc_action: str = ""
    pc_thought: str = ""

    # State *after* this step executed
    post_memory: Dict[str, Any] = field(default_factory=dict)
    post_state: Dict[str, Any] = field(default_factory=dict)
    post_pc_instruction: str = ""
    next_instruction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        return cls(**d)


class ExecutionLog:
    """Ordered collection of :class:`StepRecord` objects with
    serialization helpers."""

    def __init__(
        self,
        task_id: str = "",
        instruction: str = "",
        steps: Optional[List[StepRecord]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.instruction = instruction
        self.steps: List[StepRecord] = list(steps or [])
        self.metadata: Dict[str, Any] = metadata or {
            "created": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, record: StepRecord) -> None:
        self.steps.append(record)

    def __len__(self) -> int:
        return len(self.steps)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @property
    def last_step(self) -> Optional[StepRecord]:
        return self.steps[-1] if self.steps else None

    def truncate_to(self, step_number: int) -> None:
        """Remove all steps *after* ``step_number`` (inclusive upper trim)."""
        self.steps = [s for s in self.steps if s.step_number <= step_number]

    # ------------------------------------------------------------------
    # Mutation helpers (for interactive editing)
    # ------------------------------------------------------------------

    def replace_instruction(self, new_instruction: str) -> None:
        """Replace the NL program stored in the log."""
        self.instruction = new_instruction

    def edit_step(
        self,
        step_number: int,
        *,
        pc_instruction: Optional[str] = None,
        local_variables: Optional[Dict[str, Any]] = None,
        memory_grid: Optional[List[List[int]]] = None,
    ) -> None:
        """Patch a step's *post*-state so that a resumed run picks up the
        edits.  All steps after ``step_number`` are removed."""
        idx = None
        for i, s in enumerate(self.steps):
            if s.step_number == step_number:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Step {step_number} not found in log")

        rec = self.steps[idx]

        if pc_instruction is not None:
            rec.post_pc_instruction = pc_instruction
            lines = pc_instruction.splitlines()
            for ln in lines:
                if ln.startswith("* "):
                    rec.next_instruction = ln[2:]
                    break

        if local_variables is not None:
            rec.post_state = copy.deepcopy(rec.post_state)
            rec.post_state["local_variables"] = local_variables

        if memory_grid is not None:
            rec.post_memory = copy.deepcopy(rec.post_memory)
            rec.post_memory["grid"] = memory_grid

        self.truncate_to(step_number)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionLog":
        log = cls(
            task_id=d.get("task_id", ""),
            instruction=d.get("instruction", ""),
            metadata=d.get("metadata"),
        )
        for sd in d.get("steps", []):
            log.steps.append(StepRecord.from_dict(sd))
        return log

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ExecutionLog":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
