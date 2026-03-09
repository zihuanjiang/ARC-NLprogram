# solvers/base.py
from abc import ABC, abstractmethod
from arc.data.ARCTask import ARCTask

class Solver(ABC):
    """
    The abstract base class for any ARC task solving pipeline.
    The test harness will interact with objects that implement this interface.
    Takes an ARCTask, 
    """
    @abstractmethod
    def solve(self, task: ARCTask) -> dict:
        """
        Takes a full ARCTask and returns a dictionary containing the
        predicted solution and any other useful logs or artifacts.
        """
        pass
