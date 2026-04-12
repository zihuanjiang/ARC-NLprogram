from abc import ABC, abstractmethod
from typing import Optional

from arc.data.ARCTask import ARCTask

class ProgramGenerator(ABC):
    """
    The abstract base class for any component that can synthesize a
    natural language program (a set of instructions) to solve an ARC task.
    """

    @abstractmethod
    def generate(
        self,
        task: ARCTask,
        abstractions: Optional[dict] = None
    ) -> dict:
        """
        Takes a task and optional object abstractions, and returns a
        natural language program for solving the task.

        Args:
            task (ARCTask): The full ARC task object, containing training pairs
                            which are used as demonstrations for the rule.
            abstractions (Optional[dict]): Optional structured data from a
                                           previous abstraction stage, which can
                                           be used to enrich the prompt.

        Returns:
            dict: A dictionary containing at least the generated 'instructions'
                  string, and potentially other artifacts like the reasoning
                  trace or the model used.
        """
        pass