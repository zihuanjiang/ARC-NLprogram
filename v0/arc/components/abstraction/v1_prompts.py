"""
Pydantic models and prompts for v1 grid abstraction.

NOTE: Prompt contents and full Pydantic model definitions are closed-source.
Placeholder stubs are provided below to preserve interface compatibility.
"""
from typing import List, Literal
from pydantic import BaseModel, Field

Color = Literal["black", "blue", "red", "green", "yellow", "grey", "pink", "orange", "purple", "brown"]


class Coordinate(BaseModel):
    """Single pixel location (0-indexed)."""
    pass


class BoundingBox(BaseModel):
    """Tight rectangle enclosing an object."""
    pass


class GridObject(BaseModel):
    """A distinct contiguous or patterned set of pixels."""
    pass


class Grid(BaseModel):
    """Whole-grid description."""
    pass


class GridPair(BaseModel):
    """Input/output pair for one task."""
    pass


# ---------------------------------------------------------------------------
# Prompts  (closed-source — only stub strings are provided)
# ---------------------------------------------------------------------------

TRAIN_PAIR_DESCRIBE_PROMPT = "[REDACTED — closed-source prompt]"

TRAIN_PAIR_ABSTRACT_PROMPT = "[REDACTED — closed-source prompt]"

TEST_ABSTRACT_PROMPT = "[REDACTED — closed-source prompt]"
