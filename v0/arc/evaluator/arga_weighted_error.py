# evaluator/arga_weighted_error.py
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from .base import Metric


class ARGAWeightedError(Metric):
    """
    ARGA-AAAI23 weighted error metric.
    
    This metric implements the scoring logic from ARGA-AAAI23's calculate_score method.
    It compares grids pixel-by-pixel and applies weighted penalties:
    - Penalty of 2: when one pixel is background and the other is not (object/background misclassification)
    - Penalty of 1: when both pixels are non-background but have different colors (color error)
    
    Returns normalized error (error / total_pixels) where lower is better.
    """
    
    def __init__(self):
        super().__init__(name="arga_weighted_error")
    
    def _get_background_color(self, grid: List[List[int]]) -> int:
        """
        Determine the background color of a grid.
        Background is defined as the most common color, or 0 if present.
        
        Args:
            grid: The grid to analyze
            
        Returns:
            The background color value
        """
        if not grid or not grid[0]:
            return 0
        
        # Flatten grid to count colors
        colors = []
        for row in grid:
            colors.extend(row)
        
        if not colors:
            return 0
        
        # If 0 (black) is present, it's typically background
        if 0 in colors:
            return 0
        
        # Otherwise, return the most common color
        color_counts = Counter(colors)
        return color_counts.most_common(1)[0][0]
    
    def evaluate(self, output_grid: List[List[int]], ground_truth: List[List[int]]) -> float:
        """
        Evaluate the output grid using ARGA weighted error metric.
        Uses IOU-based approach to handle row-by-row dimension differences.
        
        Args:
            output_grid: output grid (may not form a valid matrix)
            ground_truth: Ground truth grid matrix
            
        Returns:
            float: Normalized error score (error / union_pixels), where lower is better.
                  Returns 0.0 if both grids are empty, -1 on error.
        """
        try:
            # Compute intersection and union sizes
            intersection_size, union_size = self._compute_intersection_union(output_grid, ground_truth)
            
            if union_size == 0:
                # Both grids are empty
                return 0.0
            
            # Determine background color from ground truth
            background_color = self._get_background_color(ground_truth)
            
            error_score = 0
            
            # Get maximum rows to iterate through
            max_rows = max(len(output_grid), len(ground_truth))
            
            # Compare intersection region (cells that exist in both grids)
            for r in range(max_rows):
                len_output = self._get_row_length(output_grid, r)
                len_gt = self._get_row_length(ground_truth, r)
                
                # Compare overlapping columns in this row
                intersection_cols = min(len_output, len_gt)
                for c in range(intersection_cols):
                    output_color = self._get(output_grid, r, c)
                    gt_color = self._get(ground_truth, r, c)
                    
                    # Both should be valid in intersection region
                    if output_color is not None and gt_color is not None:
                        if output_color != gt_color:
                            # Colors don't match
                            if (output_color == background_color) or (gt_color == background_color):
                                # One is background, the other is not - object/background misclassification
                                error_score += 2
                            else:
                                # Both are non-background but different colors - color error
                                error_score += 1
                
                # Handle output-only cells (extra columns in output)
                for c in range(intersection_cols, len_output):
                    output_color = self._get(output_grid, r, c)
                    if output_color is not None:
                        # Output-only cell: treat as error (object/background misclassification)
                        error_score += 2
                
                # Handle GT-only cells (extra columns in ground truth)
                for c in range(intersection_cols, len_gt):
                    gt_color = self._get(ground_truth, r, c)
                    if gt_color is not None:
                        # GT-only cell: treat as error (object/background misclassification)
                        error_score += 2
            
            # Normalize by union size
            normalized_error = error_score / union_size
            return normalized_error
            
        except Exception as e:
            print(f"Error in evaluation with ARGA weighted error with error message: {e}")
            print(f"Output type: {type(output_grid)}")
            print(f"Ground truth type: {type(ground_truth)}")
            if isinstance(output_grid, list):
                print(f"Output len: {len(output_grid)}")
                if len(output_grid) > 0:
                    print(f"Output[0] type: {type(output_grid[0])}")
            return -1
