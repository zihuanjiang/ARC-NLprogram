"""
Abstraction components package.
"""
from .abstractor_registry import get_registry, register_abstractor

# Export commonly used classes and functions
from .base import Abstractor
from .v1 import LLMAbstractor_v1
from .llm_abstractor import LLMAbstractor
from .heuristic_abstractor import HeuristicAbstractor
from .dynamic_abstractor import DynamicAbstractor

# Register default abstractors
_registry = get_registry()

_registry.register('llm', LLMAbstractor_v1)
_registry.register('dynamic', DynamicAbstractor)
_registry.register('heuristic', HeuristicAbstractor)

# Export main classes
from .base import Abstractor
from .abstractor_registry import AbstractorRegistry, register_abstractor, get_registry

__all__ = [
    'Abstractor',
    'AbstractorRegistry',
    'register_abstractor',
    'get_registry',
    'LLMAbstractor_v1',
    'DynamicAbstractor',
    'HeuristicAbstractor',
]
