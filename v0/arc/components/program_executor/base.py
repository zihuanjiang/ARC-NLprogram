# program_executor/base.py
from abc import ABC, abstractmethod
from typing import Optional
from arc.data.ARCTask import ARCTask
class ProgramExecutor(ABC):
    """
    The abstract base class for any component that can execute a set of
    instructions to solve an ARC task.
    """
    @abstractmethod
    def execute(
        self,
        task: ARCTask,
        instructions: str,
        abstractions: Optional[dict] = None
    ) -> dict:
        
        """
        Takes a task, a set of instructions, and abtraction (optionally) and outputs a solution.

        Args:
            task (ARCTask): The full task object, including training and test pairs.
            instructions (str): The natural language program to be executed.
            abstractions (Optional[dict]): Optional structured data from aprevious abstraction stage.

        Returns:
            dict: A dictionary containing at least the 'predicted_grid', and other optional logs.
        """
        pass