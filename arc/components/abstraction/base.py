# abstraction/base.py
import io
import base64
import numpy as np
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from abc import ABC, abstractmethod
from arc.data.ARCTask import ARCTask
from arc.utils.plotting import ARC_COLORS, ARC_CMAP, COLOR_MAP

class Abstractor(ABC):
    """
    The abstract base class for any component that can synthesize a
    json style graph abstraction of the grid to solve an ARC task.
    """
    def __init__(
        self, 
        include_train_input: bool = True, 
        include_test_input: bool = True
    ):
        """
        Args:
            include_train_input: Generate grid abstraction for training pairs.
            include_test_input: Generate grid abstraction for test grid.
        """
        self.include_train_input = include_train_input
        self.include_test_input = include_test_input

    @abstractmethod
    def abstract_train_pairs(
        self,
        task: ARCTask,
    ) -> tuple[list[dict], list]:
        """
        Takes a task and returns a dictionary of grid abstractions

        Args:
            task (ARCTask): The full ARC task object, containing training pairs
                            which are used as demonstrations for the rule.

        Returns:
            tuple[list[dict], list]: A tuple containing:
                - list of dictionaries containing the grid abstraction for input-output grid pairs
                - list of usage statistics
        """
        pass
    
    @abstractmethod
    def abstract_test_grids(
        self,
        task: ARCTask,
        grid_abstraction: Optional[dict] = None,
    ) -> tuple[list[dict], list]:
        """
        Takes a task and returns a dictionary of grid abstractions

        Args:
            task (ARCTask): The full ARC task object, containing training pairs
                            which are used as demonstrations for the rule.
            grid_abstraction (Optional[dict]): Optional grid_abstraction dict with
                                               training grid abstraction.

        Returns:
            tuple[list[dict], list]: A tuple containing:
                - list of dictionaries containing the grid abstraction for output grids
                - list of usage statistics
        """
        pass
    
    def abstract(
        self,
        task: ARCTask,
    ) -> tuple[dict, list]:
        """
        Takes a task and returns a dictionary of grid abstractions

        Args:
            task (ARCTask): The full ARC task object, containing training pairs
                            which are used as demonstrations for the rule.

        Returns:
            tuple[dict, list]: A tuple containing:
                - dictionary containing the grid abstraction of training pairs (key 'train') and test example (key 'test')
                - list of aggregated usage statistics
        """
        grid_abstraction = None
        total_usage = []
        if self.include_train_input:
            print(f"Generating grid abstraction for task {task.task_id} training pairs")
            train_abstraction, train_usage = self.abstract_train_pairs(task)
            grid_abstraction = {'train': train_abstraction}
            total_usage.extend(train_usage)
        if self.include_test_input:
            print(f"Generating grid abstraction for task {task.task_id} test grid")
            grid_abstraction, test_usage = self.abstract_test_grids(task, grid_abstraction)
            total_usage.extend(test_usage)
        return grid_abstraction, total_usage
    
    def resize_image(
        self, 
        image_base64: str, 
        image_dimension: tuple[int],
    ) -> str:
        """
        Resize an image base64 string to the desired image dimension.

        Args:
            image_base64: The input image represented in base64 format.
            image_dimension: The output image dimension represent as tuple of integer.

        Returns:
            str: The resized image represented in base64 format. 
        """
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize(image_dimension)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return resized_base64

    def base64_from_grid(
        self,
        grid: list[list[int]],
    ) -> str:
        """
        Generate base64 representation of an input grid from matrix form.

        Args:
            grid: matrix form of the input grid
        
        Returns:
            str: base64 representation of the input grid
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.viz_grid(grid, COLOR_MAP, ax=ax)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", pad_inches=0, dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        return image_base64
    
    def viz_grid(
        self,
        grid: list[list[int]], 
        color_map: dict[int, str], 
        ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Inherited from Jeremy Berman (https://github.com/jerber/arc-lang-public).
        Visualizes a grid of integer cells as colored squares on a given matplotlib Axes.

        Each integer in the grid is mapped to a color defined in color_map.
        A slight grey border is drawn between cells.

        Parameters:
            grid (list[list[int]]): A 2D list representing the grid of integers.
            color_map (dict[int, str]): A mapping from integer values to color strings.
                                        Example: {0: 'white', 1: 'blue', 2: 'red'}
            ax (plt.Axes, optional): An Axes object to plot on. If None, a new figure and axis are created.

        Returns:
            plt.Axes: The Axes with the plotted grid.
        """
        # Make a local copy of the grid and convert to a NumPy array.
        grid = grid.copy()
        grid_np = np.array(grid)
        rows, cols = grid_np.shape

        # Establish an ordering for the colormap based on sorted keys.
        ordered_keys = sorted(color_map.keys())
        mapping = {val: idx for idx, val in enumerate(ordered_keys)}

        # Map grid values to indices.
        mapped_grid = np.vectorize(mapping.get)(grid_np)

        # Create a ListedColormap and a BoundaryNorm for crisp cell boundaries.
        cmap = mcolors.ListedColormap([color_map[val] for val in ordered_keys])
        norm = mcolors.BoundaryNorm(
            np.arange(-0.5, len(ordered_keys) + 0.5, 1), len(ordered_keys)
        )

        # Create an axis if not provided.
        if ax is None:
            fig, ax = plt.subplots()

        # Display the grid.
        ax.imshow(mapped_grid, cmap=cmap, norm=norm)

        # Set tick positions to align gridlines with cell boundaries.
        ax.set_xticks([x - 0.5 for x in range(1 + cols)])
        ax.set_yticks([y - 0.5 for y in range(1 + rows)])

        # Draw gridlines with a slight grey border between cells.
        ax.grid(which="both", color="white", linewidth=3)

        # Remove tick labels.
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Hide the axes spines for a cleaner look.
        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax