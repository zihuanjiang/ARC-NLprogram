# evaluator/pointwise_acc.py
from typing import List
from .base import Metric

class Pointwise_acc(Metric):
    def __init__(self):
        super().__init__(name="pointwise_acc")
    
    def evaluate(self, output: List[List[int]], ground_truth: List[List[int]]) -> float:
        """
        Accuracy over the union of cells using IOU approach.
        Non-overlapping cells count as incorrect.
        If both grids are empty, returns 1.0.
        """
        try:
            # Compute intersection and union sizes
            intersection_size, union_size = self._compute_intersection_union(output, ground_truth)
            
            if union_size == 0:
                # Both grids are empty
                return 1.0

            correct = 0

            # Get maximum rows to iterate through
            max_rows = max(len(output), len(ground_truth))
            
            # Count correct in intersection region (cells that exist in both grids)
            for r in range(max_rows):
                len_output = self._get_row_length(output, r)
                len_gt = self._get_row_length(ground_truth, r)
                
                # Compare overlapping columns in this row
                intersection_cols = min(len_output, len_gt)
                for c in range(intersection_cols):
                    output_color = self._get(output, r, c)
                    gt_color = self._get(ground_truth, r, c)
                    if output_color is not None and gt_color is not None and output_color == gt_color:
                        correct += 1
                
                # Non-overlapping cells (output-only or GT-only) count as incorrect
                # They are already accounted for in union_size but not in correct count
            
            return correct / union_size
        except Exception as e:
            print(f"Error in evaluation with pointwise accuracy with error message: {e}")
            print(f"Output type: {type(output)}")
            print(f"Ground truth type: {type(ground_truth)}")
            if isinstance(output, list):
                print(f"Output len: {len(output)}")
                if len(output) > 0: print(f"Output[0] type: {type(output[0])}")
            return -1
