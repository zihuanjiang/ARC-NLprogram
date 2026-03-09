# evaluator/weighted_f1.py
from collections import defaultdict
from typing import List
from .base import Metric

class Weighted_f1(Metric):
    def __init__(self):
        super().__init__(name="weighted_f1")
    
    def evaluate(self, output: List[List[int]], ground_truth: List[List[int]]) -> float:
        """
        Pixel-wise, support-weighted F1 across classes using IOU approach.
        - TP_c: cells where output==ground_truth==c (in intersection).
        - FP_c: cells predicted as c where ground_truth != c or is missing (includes output-only area).
        - FN_c: cells where ground_truth==c but output != c or is missing (includes GT-only area).
        Weighted by ground-truth support per class (sum over union uses only GT counts for weights).
        """
        try:
            # Compute intersection and union sizes
            intersection_size, union_size = self._compute_intersection_union(output, ground_truth)
            
            if union_size == 0:
                # Both grids are empty
                return 1.0

            TP = defaultdict(int)
            FP = defaultdict(int)
            FN = defaultdict(int)
            support = defaultdict(int)  # ground-truth support per class

            # Get maximum rows to iterate through
            max_rows = max(len(output), len(ground_truth))

            # Process all rows with IOU approach
            for r in range(max_rows):
                len_output = self._get_row_length(output, r)
                len_gt = self._get_row_length(ground_truth, r)
                
                # Intersection area: cells that exist in both grids
                intersection_cols = min(len_output, len_gt)
                for c in range(intersection_cols):
                    g = self._get(ground_truth, r, c)
                    o = self._get(output, r, c)
                    
                    if g is not None:
                        support[g] += 1
                    
                    if o is not None and g is not None:
                        if o == g:
                            TP[g] += 1
                        else:
                            FP[o] += 1
                            FN[g] += 1
                    elif o is not None:
                        # Output exists but GT doesn't (shouldn't happen in intersection, but safe)
                        FP[o] += 1
                    elif g is not None:
                        # GT exists but output doesn't (shouldn't happen in intersection, but safe)
                        FN[g] += 1

                # Output-only area (extra columns in output): count as FP for predicted classes
                for c in range(intersection_cols, len_output):
                    o = self._get(output, r, c)
                    if o is not None:
                        FP[o] += 1

                # GT-only area (extra columns in ground truth): count as FN for GT classes and add to support
                for c in range(intersection_cols, len_gt):
                    g = self._get(ground_truth, r, c)
                    if g is not None:
                        support[g] += 1
                        FN[g] += 1

            # Compute weighted F1
            def f1(tp, fp, fn) -> float:
                # F1 = 2TP / (2TP + FP + FN); safe for zeros
                denom = (2 * tp + fp + fn)
                return (2 * tp / denom) if denom > 0 else 0.0

            # Classes to consider: all seen in GT or output
            classes = set(list(support.keys()) + list(TP.keys()) + list(FP.keys()) + list(FN.keys()))
            total_support = sum(support.values())

            if total_support == 0:
                # No ground-truth pixels (both empty or GT empty). Define F1 as 1.0 if output also empty, else 0.0
                output_nonempty = union_size > 0 and any(
                    self._get_row_length(output, r) > 0 
                    for r in range(len(output))
                )
                return 1.0 if not output_nonempty else 0.0

            weighted_sum = 0.0
            for c in classes:
                w = support[c] / total_support if total_support > 0 else 0.0
                weighted_sum += w * f1(TP[c], FP[c], FN[c])
            return weighted_sum
        except Exception as e:
            print(f"Error in evaluation with weighted f1 with error message: {e}")
            return -1
