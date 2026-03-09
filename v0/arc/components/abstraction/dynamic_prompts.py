"""
Pydantic models and prompts for dynamic schema-based grid abstraction.

NOTE: Prompt contents and full Pydantic model definitions are closed-source.
Placeholder stubs are provided below to preserve interface compatibility.
"""
from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field


class SchemaFieldDefinition(BaseModel):
    """Definition of a single field in the schema."""
    pass


class SchemaObjectDefinition(BaseModel):
    """Definition of a complete object schema."""
    pass


class GeneratedGridObjectSchema(BaseModel):
    """The GridObject schema structure that the LLM should generate."""
    pass


SCHEMA_GENERATION_TEMPLATE = GeneratedGridObjectSchema.model_json_schema()

# ---------------------------------------------------------------------------
# Prompts  (closed-source — only stub strings are provided)
# ---------------------------------------------------------------------------

SCHEMA_GENERATION_PROMPT = "[REDACTED — closed-source prompt]"

TRAIN_PAIR_ABSTRACT_PROMPT_DYNAMIC = "[REDACTED — closed-source prompt]"

TEST_ABSTRACT_PROMPT_DYNAMIC = "[REDACTED — closed-source prompt]"
