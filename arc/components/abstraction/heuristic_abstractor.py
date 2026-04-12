# abstraction/heuristic_abstractor.py

import os
import copy
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

from .base import Abstractor
from arc.data.ARCTask import ARCTask

# Import ARGA components
try:
    # Add the ARGA path to sys.path if needed
    arga_path = Path(__file__).parent / "ARGA-AAAI23"
    if str(arga_path) not in os.sys.path:
        os.sys.path.insert(0, str(arga_path))

    from task import Task as ARGATask
    from image import Image
    from ARCGraph import ARCGraph
    ARGA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ARGA components: {e}")
    ARGA_AVAILABLE = False

from arc.solvers.registry import AbstractorRegistry

@AbstractorRegistry.register("heuristic")
class HeuristicAbstractor(Abstractor):
    """
    Abstractor that uses the heuristic search method from ARGA-AAAI23.
    This approach tries different graph abstractions and transformation sequences
    to find patterns that transform input grids to output grids.
    """

    def __init__(
        self,
        time_limit: int = 1800,  # Shorter time limit for individual abstractions
        shared_frontier: bool = True,
        do_constraint_acquisition: bool = True,
        include_train_input: bool = True,
        include_test_input: bool = True,
        **kwargs, #TODO
    ):
        """
        Initialize the heuristic abstractor.

        Args:
            time_limit: Maximum time in seconds to spend searching for a solution
            shared_frontier: Whether to use shared frontier in ARGA search
            do_constraint_acquisition: Whether to use constraint acquisition
            include_train_input: Generate grid abstraction for training pairs
            include_test_input: Generate grid abstraction for test grids
        """
        super().__init__(include_train_input=include_train_input, include_test_input=include_test_input)
        self.abstraction = None

        if not ARGA_AVAILABLE:
            print("Warning: ARGA-AAAI23 components not available. HeuristicAbstractor will return empty results.")
            self.time_limit = time_limit
            self.shared_frontier = shared_frontier
            self.do_constraint_acquisition = do_constraint_acquisition
            return

        self.time_limit = time_limit
        self.shared_frontier = shared_frontier
        self.do_constraint_acquisition = do_constraint_acquisition

    def abstract_train_pairs(
        self,
        task: ARCTask,
    ) -> tuple[list[dict], list]:
        """
        Generate abstractions for training pairs using ARGA heuristic search.

        Returns:
            Tuple of (list of dicts, usage list), where each dict represents a training pair with structure:
            {
                'input': abstraction_dict,
                'output': abstraction_dict
            }
        """
        if not self.include_train_input:
            return [], []

        # Convert ARC task to ARGA format
        arga_task = self._arc_task_to_arga_task(task)
        self.abstraction = None

        # Run ARGA search to find solution
        try:
            self.abstraction = arga_task.solve(
                shared_frontier=self.shared_frontier,
                time_limit=self.time_limit,
                do_constraint_acquisition=self.do_constraint_acquisition,
                save_images=False
            )

            # Create abstractions for each training pair
            train_abstractions = []
            for i, (input_grid, output_grid) in enumerate(zip(task.trainingExamples, [pair['output'] for pair in task.trainingExamples])):
                # Get the abstracted graph for this pair using the found abstraction
                if self.abstraction:
                    try:
                        input_abstracted = getattr(arga_task.train_input[i], Image.abstraction_ops[self.abstraction])()
                        output_abstracted = getattr(arga_task.train_output[i], Image.abstraction_ops[self.abstraction])()

                        # Convert ARCGraph to dict format
                        input_abs_dict = self._arcgraph_to_dict(input_abstracted)
                        output_abs_dict = self._arcgraph_to_dict(output_abstracted)

                        train_abstractions.append({
                            'input': input_abs_dict,
                            'output': output_abs_dict
                        })
                    except Exception as e:
                        print(f"Warning: Failed to create abstraction for training pair {i}: {e}")
                        # Fallback: use empty abstraction
                        train_abstractions.append({
                            'input': {},
                            'output': {}
                        })
                else:
                    # No solution found, return empty abstractions
                    train_abstractions.append({
                        'input': {},
                        'output': {}
                    })

            return train_abstractions, []

        except Exception as e:
            print(f"Warning: ARGA search failed: {e}")
            # Return empty abstractions for all pairs
            return [{'input': {}, 'output': {}} for _ in task.trainingExamples], []

    def abstract_test_grids(
        self,
        task: ARCTask,
        grid_abstraction: Optional[Dict] = None,
    ) -> tuple[list[dict], list]:
        """
        Generate abstractions for test grids using ARGA heuristic search.

        Args:
            task: The ARC task
            grid_abstraction: Optional training abstractions (not used in heuristic approach)

        Returns:
            Tuple of (list of dicts, usage list), where each dict represents a test grid abstraction
        """
        if not self.include_test_input:
            return [], []

        # Convert ARC task to ARGA format
        arga_task = self._arc_task_to_arga_task(task)

        # Run ARGA search to find solution
        try:
            # Create abstractions for test grids
            test_abstractions = []
            for i, test_example in enumerate(task.testExamples):
                test_grid = test_example['input']

                if self.abstraction:
                    try:
                        # Get the abstracted graph for test grid using the found abstraction
                        test_abstracted = getattr(arga_task.test_input[i], Image.abstraction_ops[self.abstraction])()

                        # Convert ARCGraph to dict format
                        test_abs_dict = self._arcgraph_to_dict(test_abstracted)
                        test_abstractions.append(test_abs_dict)

                    except Exception as e:
                        print(f"Warning: Failed to create abstraction for test grid {i}: {e}")
                        test_abstractions.append({})
                else:
                    # No solution found, return empty abstraction
                    test_abstractions.append({})

            grid_abstraction.update({"test": test_abstractions})
            return grid_abstraction, []

        except Exception as e:
            print(f"Warning: ARGA search failed for test grids: {e}")
            # Return empty abstractions for all test grids
            return [{} for _ in task.testExamples], []

    def _arc_task_to_arga_task(self, task: ARCTask) -> 'ARGATask':
        """Convert ARC task to ARGA task format."""
        # Create temporary JSON file for ARGA
        task_data = {
            "train": [
                {"input": pair["input"], "output": pair["output"]}
                for pair in task.trainingExamples
            ],
            "test": [
                {"input": example["input"], "output": example["output"]}
                for example in task.testExamples
            ]
        }

        # Create temporary file
        temp_file = f"/tmp/{task.task_id}_temp.json"
        with open(temp_file, 'w') as f:
            json.dump(task_data, f)

        # Ensure images directory exists for ARGA
        # ARGA extracts task_id from image names (first part before '_')
        # So it will try to save to images/{task.task_id}/
        images_dir = f"images/{task.task_id}"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)

        try:
            # Create ARGA task
            arga_task = ARGATask(temp_file)
            return arga_task
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _arcgraph_to_dict(self, arc_graph: 'ARCGraph') -> Dict[str, Any]:
        """
        Convert ARCGraph object to dictionary format for JSON serialization.

        Args:
            arc_graph: The ARCGraph object to convert

        Returns:
            Dictionary representation of the graph abstraction
        """
        if arc_graph is None:
            return {}

        try:
            graph_dict = {
                'abstraction_type': arc_graph.abstraction,
                'nodes': [],
                'edges': [],
                'metadata': {
                    'width': arc_graph.width,
                    'height': arc_graph.height,
                    'task_id': arc_graph.task_id,
                    'is_multicolor': arc_graph.is_multicolor,
                    'most_common_color': arc_graph.most_common_color,
                    'least_common_color': arc_graph.least_common_color
                }
            }

            # Add nodes
            for node_id, node_data in arc_graph.graph.nodes(data=True):
                node_dict = {
                    'id': node_id,
                    'color': node_data.get('color'),
                    'size': node_data.get('size'),
                    'pixels': node_data.get('nodes', [])
                }
                graph_dict['nodes'].append(node_dict)

            # Add edges
            for edge in arc_graph.graph.edges(data=True):
                edge_dict = {
                    'source': edge[0],
                    'target': edge[1],
                    'direction': edge[2].get('direction')
                }
                graph_dict['edges'].append(edge_dict)

            return graph_dict

        except Exception as e:
            print(f"Warning: Failed to convert ARCGraph to dict: {e}")
            return {}
