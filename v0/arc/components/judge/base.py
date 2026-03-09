from abc import ABC, abstractmethod
from arc.data.ARCTask import ARCTask

class Judge(ABC):
    """
    Abstract base class for a Judge that evaluates the correctness of generated instructions.
    """
    
    @abstractmethod
    def judge(self, task: ARCTask, instructions: str) -> dict:
        """
        Evaluates the instructions against the task's training examples.
        
        Args:
            task (ARCTask): The ARC task containing training examples.
            instructions (str): The natural language program to evaluate.
            
        Returns:
            dict: A dictionary containing the verdict and analysis.
        """
        pass
