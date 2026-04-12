# evaluator/success_rate.py
from .base import Metric
from typing import List

class Success_rate(Metric):
    def __init__(self):
        super().__init__(name="success_rate")
    
    def evaluate(self, output: List[List[int]], ground_truth: List[List[int]]) -> float:
        return output == ground_truth
