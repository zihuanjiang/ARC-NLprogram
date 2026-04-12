"""
LLM model configurations.

NOTE: Specific model names and provider configurations are closed-source.
Add your own model configurations below to use the solver.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """A dataclass to hold the configuration for an LLM API call."""
    model_name: str
    temperature: float = 1
    max_tokens: int = 100000
    extra_body: Dict[str, Any] = field(default_factory=dict)


MODEL_CONFIGURATIONS: Dict[str, ModelConfig] = {
    # Add your own entries here. Example:
    #
    # "my-model": ModelConfig(
    #     model_name="provider/model-name",
    #     temperature=1.0,
    #     max_tokens=16384,
    #     extra_body={},
    # ),
}
