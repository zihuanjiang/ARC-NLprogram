# arc/utils/plotting.py

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import io
import base64

COLOR_MAP: dict[int, str] = {
    0: "black",
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # grey
    6: "#F012BE",  # pink
    7: "#FF851B",  # orange
    8: "#9d00ff",  # purple
    9: "#870C25",  # brown
}

# ARC_COLORS = [
#     "#000000","#2596be","#f93c31","#4fcc30","#ffdc00",
# #     "#999999","#e53aa3","#ff851b","#87d8f1","#921231",
# ]
ARC_COLORS = list(COLOR_MAP.values())

ARC_CMAP = ListedColormap(ARC_COLORS, name="arc10")

def plot_example(
    task,
    split: str = "train",
    *,
    save_path: str | None = None,      # folder to save into
    save_image: bool = False,          # save image or not
    show_image: bool = True,           # show image or not
    **kwargs
):
    examples = task.trainingExamples if split == "train" else task.testExamples
    if save_image:
        if not save_path:
            raise ValueError('save_path not provided')
        os.makedirs(save_path, exist_ok=True)

    for idx, ex in enumerate(examples):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        input_grid = ex["input"]
        output_grid = ex["output"]

        input_title = "Input"
        output_title = "Output" if split == "train" else "Output (Solution)"

        # Plot Input
        axes[0].imshow(input_grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        axes[0].set_title(input_title)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Plot Output
        axes[1].imshow(output_grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        axes[1].set_title(output_title)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.suptitle(f"Task: {task.task_id} - {split.capitalize()} Example #{idx}", y=1.02)
        plt.tight_layout()

        if show_image:
            plt.show()
        
        if save_image:
            save_target = os.path.join(save_path, f"{split}_{idx}.png")
            fig.savefig(save_target, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_grid(ascii_grid):
    plt.imshow(ascii_grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

# --- Extend colormap with padding color ---
PAD_COLOR_INDEX = -1
PAD_COLOR_HEX = "#00FFFF"  # Teal (distinct, not white)
EXTENDED_COLORS = [PAD_COLOR_HEX] + [ARC_CMAP.colors[i] for i in range(ARC_CMAP.N)]
PADDED_CMAP = ListedColormap(EXTENDED_COLORS)

def pad_grid(grid, pad_val=PAD_COLOR_INDEX):
    """Pad a ragged 2D list/array to rectangular shape with pad_val."""
    if not isinstance(grid, (list, np.ndarray)):
        return grid  # skip invalid inputs
    if grid is None:
        return grid
    try:
        lens = list(map(len, grid))
        if not lens:
             return grid
    except TypeError:
        return grid
    H, W = len(grid), max(lens)
    fixed = np.full((H, W), pad_val, dtype=int)
    for i, row in enumerate(grid):
        fixed[i, :len(row)] = row
    # Shift so that pad (-1) → 0 in colormap
    return fixed + 1

def plot_solution_comparison(
    task_id, 
    test_input, 
    predicted_grid, 
    target_grid,
    *,
    save_path: str | None = None,      # folder to save into
    save_image: bool = False,          # save image or not
    show_image: bool = True,           # show image or not
):
    """
    Plots the test input, the predicted output, and the correct output side-by-side.
    """
    if save_image:
        if not save_path:
            raise ValueError('save_path not provided')
        os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    grids = [test_input, predicted_grid, target_grid]
    titles = ["Test Input", "Predicted Output", "Correct Solution"]

    for ax, grid, title in zip(axes, grids, titles):
        padded_grid = pad_grid(grid)
        ax.imshow(padded_grid, cmap=PADDED_CMAP, vmin=0, vmax=10, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle(f"Comparison for Task: {task_id}", y=1.02)
    plt.tight_layout()

    if show_image:
        plt.show()
    
    if save_image:
        save_target = os.path.join(save_path, f"{task_id}_solution_comparison.png")
        fig.savefig(save_target, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_multi_solution_comparison(
    task_id, 
    test_input, 
    predictions: dict, # {name: grid}
    target_grid,
    *,
    save_path: str | None = None,
    save_image: bool = False,
    show_image: bool = True,
):
    """
    Plots the test input, multiple predicted outputs, and the correct output side-by-side.
    """
    if save_image:
        if not save_path:
            raise ValueError('save_path not provided')
        os.makedirs(save_path, exist_ok=True)
    
    n_preds = len(predictions)
    n_cols = 2 + n_preds # Input + Preds + Target
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Prepare lists
    grids = [test_input]
    titles = ["Test Input"]
    
    for name, grid in predictions.items():
        grids.append(grid)
        titles.append(f"Pred: {name}")
        
    grids.append(target_grid)
    titles.append("Correct Solution")

    for ax, grid, title in zip(axes, grids, titles):
        if grid is None:
            # Handle None grid (e.g. error)
            ax.text(0.5, 0.5, "No Output", ha='center', va='center')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            padded_grid = pad_grid(grid)
            ax.imshow(padded_grid, cmap=PADDED_CMAP, vmin=0, vmax=10, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        
    fig.suptitle(f"Multi-Model Comparison for Task: {task_id}", y=1.02)
    plt.tight_layout()

    if show_image:
        plt.show()
    
    if save_image:
        save_target = os.path.join(save_path, f"{task_id}_multi_comparison.png")
        fig.savefig(save_target, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_task(
    task,
    *,
    save_path: str | None = None,      # folder to save into
    save_image: bool = False,          # save image or not
    show_image: bool = True,           # show image or not
    **kwargs
):
    """
    Plots all training and test examples for a task in a single figure,
    with each example on its own row.
    """
    train_ex = task.trainingExamples
    test_ex = task.testExamples
    all_ex = [(ex, "Train") for ex in train_ex] + [(ex, "Test") for ex in test_ex]
    n_ex = len(all_ex)

    if n_ex == 0:
        print("No examples to plot.")
        return

    # Layout logic: 1 example per row
    n_rows = n_ex
    n_cols = 2 # Input, Output
    
    # Adjust figsize based on number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows * 3))
    
    # Normalize axes to 2D array [row, col]
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (ex, split) in enumerate(all_ex):
        row = idx
        
        ax_in = axes[row, 0]
        ax_out = axes[row, 1]
        
        input_grid = ex["input"]
        output_grid = ex["output"]
        
        # Plot Input
        ax_in.imshow(input_grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        ax_in.set_title(f"{split} Example #{idx} Input", fontsize=10)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        
        # Plot Output
        ax_out.imshow(output_grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        ax_out.set_title(f"{split} Example #{idx} Output", fontsize=10)
        ax_out.set_xticks([])
        ax_out.set_yticks([])

    fig.suptitle(f"Task: {task.task_id}", y=1.02)
    plt.tight_layout()
    
    if show_image:
        plt.show()
    
    if save_image:
        if not save_path:
             raise ValueError('save_path not provided')
        os.makedirs(save_path, exist_ok=True)
        save_target = os.path.join(save_path, f"task_{task.task_id}_all.png")
        fig.savefig(save_target, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Needed to add here so that base 64 image can be used at later stages of the pipeline
def base64_from_grid(grid: list[list[int]]) -> str:
    """
    Generate base64 representation of an input grid from matrix form.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Use existing plotting logic but simplified for single grid
    # We can reuse plot_grid logic but we need it on a specific ax
    
    # Make a local copy of the grid and convert to a NumPy array.
    grid = np.array(grid)
    rows, cols = grid.shape

    # Create a ListedColormap and a BoundaryNorm for crisp cell boundaries.
    # We use ARC_CMAP directly
    
    # Display the grid.
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")

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

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", pad_inches=0, dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return image_base64