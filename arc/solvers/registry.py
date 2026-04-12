from typing import Type, Dict, Any

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        def decorator(cls: Type):
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in {self.name} registry.")
            self._registry[name] = cls
            return cls
        return decorator

    def get_implementation(self, name: str) -> Type:
        if name not in self._registry:
            raise ValueError(f"'{name}' is not registered in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

AbstractorRegistry = Registry("Abstractor")
ProgramGeneratorRegistry = Registry("ProgramGenerator")
ProgramExecutorRegistry = Registry("ProgramExecutor")
JudgeRegistry = Registry("Judge")
