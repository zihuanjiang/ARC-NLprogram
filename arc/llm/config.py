from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    """A dataclass to hold the configuration for an LLM API call."""
    model_name: str
    temperature: float = 1
    max_tokens: int = 100000
    extra_body: Dict[str, Any] = field(default_factory=dict)

# Predefined model configurations
MODEL_CONFIGURATIONS = {
    "grok_fast": ModelConfig(
        model_name="x-ai/grok-4-fast",
        temperature=1,
        extra_body={
            "reasoning": {
                "effort": "low",
                "enabled": True,
                "exclude": False,
            }
        }
    ),
    "grok-code-fast-1": ModelConfig(
        model_name="x-ai/grok-code-fast-1",
        temperature=1,
        extra_body={
            "reasoning": {
                "effort": "medium",
                "enabled": True,
                "exclude": True, 
            }
        }
    ),
    "gemini-3-pro-preview": ModelConfig(
        model_name="google/gemini-3-pro-preview",
        temperature=1, 
        extra_body={}
    ),
    "gpt-5": ModelConfig(
        model_name="openai/gpt-5",
        temperature=1.0,
        extra_body={
            "reasoning": {"effort": "high", "enabled": True, "exclude": True}
        }
    ),
    "gpt-5.1": ModelConfig(
        model_name="openai/gpt-5.1",
        temperature=1.0,
        extra_body={
            "reasoning": {"effort": "high", "enabled": True, "exclude": True}
        }
    ),
    "grok-4.1-fast": ModelConfig(
        model_name="x-ai/grok-4.1-fast",
        temperature=1,
        extra_body={
            "reasoning": {"effort": "low", "enabled": False, "exclude": True}
        }
    ),
    "gpt-4o": ModelConfig(
        model_name="openai/gpt-4o",
        temperature=0.0,
        max_tokens=4096
    ),
    "gemini-2.5-flash" : ModelConfig(
        model_name="google/gemini-2.5-flash",
        temperature=1,
        extra_body={}
    ),
    "gemma-3-4b" : ModelConfig(
        model_name="google/gemma-3-4b-it:free",
        temperature=1,
        max_tokens=16384,
        extra_body={},
    ),
    "gemma-3-12b" : ModelConfig(
        model_name="google/gemma-3-12b-it:free",
        temperature=1,
        max_tokens=16384,
        extra_body={}
    ),
    "gemma-3-12b-backup" : ModelConfig(
        model_name="google/gemma-3-12b-it",
        temperature=1,
        max_tokens=16384,
        extra_body={
            "provider": {
                "only": ["deepinfra/bf16"],
                "allow_fallbacks": False
            }
        }
    ),
    "gemma-3-27b-backup" : ModelConfig(
        model_name="google/gemma-3-27b-it",
        temperature=1,
        max_tokens=16384,
        extra_body={
            "provider": {
                "only": ["deepinfra", "Parasail"],
                "allow_fallbacks": False
            }
        }
    ),
    "gemma-3-27b" : ModelConfig(
        model_name="google/gemma-3-27b-it:free",
        temperature=1,
        extra_body={}
    ),
    "deepseek-3.2-speciale" : ModelConfig(
        model_name="deepseek/deepseek-v3.2-speciale",
        temperature=1,
        extra_body={"include_reasoning": True}
    ),
    "deepseek-3.2" : ModelConfig(
        model_name="deepseek/deepseek-v3.2",
        temperature=1,
        extra_body={"include_reasoning": True}
    ),
    "qwen-3-4b" : ModelConfig(
        model_name="qwen/qwen3-4b:free",
        temperature=1,
        max_tokens=16384,
        extra_body={}
    ),
    "qwen-3-next-80b-a3b-instruct" : ModelConfig(
        model_name="qwen/qwen3-next-80b-a3b-instruct:free",
        temperature=1,
        extra_body={}
    ),
}