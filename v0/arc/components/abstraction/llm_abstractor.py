# abstraction/llm_abstractor.py
import os
from typing import Optional
from abc import abstractmethod

from .base import Abstractor
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS

class LLMAbstractor(Abstractor):
    """
    Baseclass Abstractor that uses LLM to generate grid abstractions.
    """
    def __init__(self, model_key: str, include_train_input: bool = True, include_test_input: bool = True):
        super().__init__(include_train_input=include_train_input, include_test_input=include_test_input)

        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found in MODEL_CONFIGURATIONS.")
        self.model_key = model_key
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))