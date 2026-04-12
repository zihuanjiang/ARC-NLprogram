# evaluator/base.py
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from arc.data.ARCTask import ARCTask
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    The abstract base class for metric.
    """
    def __init__(
        self, 
        name: str
    ):
        """
        Args:
            name: metric name
        """
        self.name = name
    
    @abstractmethod
    def evaluate(
        self, 
        output_grid: list[list[int]], 
        ground_truth: list[list[int]],
    ) -> float:
        """
        Evaluate the output grid using the stored metric

        Args:
            output_grid (list[list[int]]): output grid (may not form a valid matrix).
            ground_truth (list[list[int]]): Ground truth grid matrix.

        Returns:
            float: Metric output
        """
        pass
    
    def run(self, task: ARCTask, output_grids: list[list[list[int]]]) -> List[Dict[str, float]]:
        """Easy interface to Evaluate the output grids using the specified method."""
        result = []
        assert len(output_grids) == len(task.testExamples), "Number of output grids is not equal to number of ground truth grids"
        for output_grid, ground_truth in zip(output_grids, task.testExamples):
            result.append(self.evaluate(output_grid, ground_truth['output']))
        return result

    def _shape(self, grid: List[List[int]]) -> Tuple[int, int]:
        """Get the maximum dimensions of a grid (handles irregular rows)."""
        if not grid:
            return (0, 0)
        max_cols = max((len(row) for row in grid), default=0)
        return (len(grid), max_cols)
    
    def _get_row_length(self, grid: List[List[int]], r: int) -> int:
        """Get the length of a specific row, handling out-of-bounds."""
        if r < 0 or r >= len(grid):
            return 0
        row = grid[r]
        return len(row) if row else 0
    
    def _get(self, grid: List[List[int]], r: int, c: int) -> Optional[int]:
        """Safe accessor: returns None if out-of-bounds."""
        if r < 0 or r >= len(grid): return None
        row = grid[r]
        if c < 0 or (not row) or c >= len(row): return None
        return row[c]
    
    def _compute_intersection_union(self, a: List[List[int]], b: List[List[int]]) -> Tuple[int, int]:
        """
        Compute intersection and union sizes using IOU approach.
        Handles row-by-row dimension differences.
        
        Args:
            a: First grid (output)
            b: Second grid (ground truth)
            
        Returns:
            Tuple of (intersection_size, union_size)
            - intersection_size: number of cells that exist in both grids
            - union_size: number of cells that exist in either grid
        """
        # Get maximum dimensions
        max_rows = max(len(a), len(b))
        
        intersection_size = 0
        union_size = 0
        
        for r in range(max_rows):
            # Get row lengths for both grids at this row
            len_a = self._get_row_length(a, r)
            len_b = self._get_row_length(b, r)
            
            # Intersection: cells that exist in both rows
            intersection_cols = min(len_a, len_b)
            intersection_size += intersection_cols
            
            # Union: all cells in either row
            union_cols = max(len_a, len_b)
            union_size += union_cols
        
        return intersection_size, union_size
    
    def _union_dims(self, a: List[List[int]], b: List[List[int]]) -> Tuple[int, int, int, int, int, int]:
        """
        Legacy method for backward compatibility.
        Returns dimensions assuming rectangular grids.
        Note: This is deprecated in favor of row-by-row IOU approach.
        """
        ra, ca = self._shape(a)
        rb, cb = self._shape(b)
        r_overlap, c_overlap = min(ra, rb), min(ca, cb)
        return ra, ca, rb, cb, r_overlap, c_overlap
