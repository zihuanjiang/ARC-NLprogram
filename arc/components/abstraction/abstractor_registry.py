# abstraction/abstractor_registry.py

"""
Registry for abstractor classes. This allows adding new abstractor types
without modifying the MergedAbstractor class.

To add a new abstractor:
1. Create your abstractor class inheriting from Abstractor
2. Register it using register_abstractor() or the @register_abstractor decorator

Example:
    from arc.abstraction import register_abstractor, Abstractor
    from arc.data.ARCTask import ARCTask
    
    class MyNewAbstractor(Abstractor):
        def __init__(self, include_train_input=True, include_test_input=True):
            super().__init__(include_train_input, include_test_input)
        
        def abstract_train_pairs(self, task: ARCTask) -> tuple[list[dict], list]:
            # Your implementation
            return [...], []
        
        def abstract_test_grids(self, task: ARCTask, grid_abstraction=None) -> tuple[list[dict], list]:
            # Your implementation
            return [...], []
    
    # Register it
    register_abstractor('my_new', MyNewAbstractor)
    
    # Now you can use it in MergedAbstractor config:
    # enabled_abstractors = {'my_new': True, ...}
"""

from typing import Dict, Type, Callable, Any, Optional
from .base import Abstractor


class AbstractorRegistry:
    """
    Registry for abstractor classes. Maps abstractor names to their class constructors.
    """
    def __init__(self):
        self._registry: Dict[str, Type[Abstractor]] = {}
        self._factory_functions: Dict[str, Callable[[Dict[str, Any]], Abstractor]] = {}
    
    def register(
        self,
        name: str,
        abstractor_class: Optional[Type[Abstractor]] = None,
        factory: Optional[Callable[[Dict[str, Any]], Abstractor]] = None
    ):
        """
        Register an abstractor class or factory function.
        
        Args:
            name: The name to register the abstractor under (e.g., 'llm', 'image', 'ascii')
            abstractor_class: The abstractor class to register. Must be callable with **kwargs.
            factory: Optional factory function that takes config dict and returns an Abstractor instance.
                    If provided, this takes precedence over abstractor_class.
        
        Usage:
            # Register a class directly
            registry.register('my_abstractor', MyAbstractorClass)
            
            # Register with a factory function (for complex initialization)
            registry.register('my_abstractor', factory=lambda config: MyAbstractorClass(**config))
        """
        if factory is not None:
            self._factory_functions[name] = factory
        elif abstractor_class is not None:
            # Create a factory that instantiates the class with the config
            self._factory_functions[name] = lambda config: abstractor_class(**config)
        else:
            raise ValueError("Either abstractor_class or factory must be provided")
    
    def create(self, name: str, config: Dict[str, Any]) -> Abstractor:
        """
        Create an instance of the registered abstractor.
        
        Args:
            name: The name of the abstractor to create
            config: Configuration dict to pass to the abstractor constructor
        
        Returns:
            An instance of the registered abstractor
        """
        if name not in self._factory_functions:
            raise ValueError(
                f"Abstractor '{name}' is not registered. "
                f"Available abstractors: {list(self._factory_functions.keys())}"
            )
        
        factory = self._factory_functions[name]
        return factory(config)
    
    def is_registered(self, name: str) -> bool:
        """Check if an abstractor is registered."""
        return name in self._factory_functions
    
    def list_registered(self) -> list[str]:
        """List all registered abstractor names."""
        return list(self._factory_functions.keys())


# Global registry instance
_default_registry = AbstractorRegistry()


def register_abstractor(
    name: str,
    abstractor_class: Optional[Type[Abstractor]] = None,
    factory: Optional[Callable[[Dict[str, Any]], Abstractor]] = None,
    registry: Optional[AbstractorRegistry] = None
):
    """
    Decorator or function to register an abstractor.
    
    Usage as decorator:
        @register_abstractor('my_abstractor')
        class MyAbstractor(Abstractor):
            ...
    
    Usage as function:
        register_abstractor('my_abstractor', MyAbstractorClass)
        register_abstractor('my_abstractor', factory=lambda config: MyAbstractor(**config))
    """
    if registry is None:
        registry = _default_registry
    
    def decorator(cls_or_func):
        if isinstance(cls_or_func, type):
            registry.register(name, abstractor_class=cls_or_func)
        else:
            registry.register(name, factory=cls_or_func)
        return cls_or_func
    
    # If called as a function (not decorator)
    if abstractor_class is not None or factory is not None:
        registry.register(name, abstractor_class=abstractor_class, factory=factory)
        return None
    
    # If called as a decorator
    return decorator


def get_registry() -> AbstractorRegistry:
    """Get the default abstractor registry."""
    return _default_registry
