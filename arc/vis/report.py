"""
Visualization report — generates a multi-page PDF where each page
corresponds to one execution step.

Each page contains:
  - Step number and current action
  - The NL program with the current line starred
  - Local variables
  - Global memory summary (non-grid keys)
  - The grid rendered with the ARC colour-map

Usage::

    python -m arc.vis.report <log_file.json> <output.pdf>
"""
import io
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np


ARC_COLORS = [
    "#000000",   # 0 - black
    "#0074D9",   # 1 - blue
    "#FF4136",   # 2 - red
    "#2ECC40",   # 3 - green
    "#FFDC00",   # 4 - yellow
    "#AAAAAA",   # 5 - grey
    "#F012BE",   # 6 - pink
    "#FF851B",   # 7 - orange
    "#9d00ff",   # 8 - purple
    "#870C25",   # 9 - brown
]
ARC_CMAP = ListedColormap(ARC_COLORS, name="arc10")

PAGE_W, PAGE_H = 8.5, 11.0

MAX_PAIRS_PER_PAGE = 4


# -----------------------------------------------------------------------
# Task visualisation helpers
# -----------------------------------------------------------------------

def _render_grid_on_ax(ax: plt.Axes, grid) -> None:
    """Draw an ARC grid on *ax* with cell gridlines."""
    arr = np.array(grid, dtype=int)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    rows, cols = arr.shape
    ax.set_xticks([x - 0.5 for x in range(1, cols + 1)])
    ax.set_yticks([y - 0.5 for y in range(1, rows + 1)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="white", linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_task_pages(
    pdf: PdfPages,
    task_id: str,
    train_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> None:
    """Add one or more pages showing the task's train / test I/O pairs."""
    pairs: List[tuple] = []
    for i, ex in enumerate(train_examples):
        pairs.append((f"Train {i + 1}", ex["input"], ex["output"]))
    n_train = len(pairs)
    for i, ex in enumerate(test_examples):
        pairs.append((f"Test {i + 1}", ex["input"], ex["output"]))
    if not pairs:
        return

    for page_start in range(0, len(pairs), MAX_PAIRS_PER_PAGE):
        page_pairs = pairs[page_start : page_start + MAX_PAIRS_PER_PAGE]
        n_rows = len(page_pairs)

        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.text(
            0.5, 0.97,
            f"Task: {task_id}",
            fontsize=16, fontweight="bold", ha="center", va="top",
        )

        content_top = 0.92
        content_bot = 0.04
        row_h = (content_top - content_bot) / n_rows

        for i, (label, inp_grid, out_grid) in enumerate(page_pairs):
            global_idx = page_start + i
            row_top = content_top - i * row_h

            # Dashed separator before the first test pair
            if global_idx == n_train and n_train > 0:
                sep_y = row_top + row_h * 0.06
                fig.add_artist(
                    plt.Line2D(
                        [0.05, 0.95], [sep_y, sep_y],
                        transform=fig.transFigure,
                        color="#999999", linewidth=0.8, linestyle="--",
                    )
                )

            # Vertical row label
            fig.text(
                0.025, row_top - row_h * 0.5,
                label, fontsize=9, fontweight="bold",
                va="center", ha="center", rotation=90,
            )

            grid_top = row_top - 0.02
            grid_h = row_h * 0.82
            grid_bot = grid_top - grid_h

            ax_in = fig.add_axes([0.08, grid_bot, 0.36, grid_h])
            _render_grid_on_ax(ax_in, inp_grid)
            ax_in.set_title("Input", fontsize=8, pad=3)

            fig.text(
                0.48, row_top - row_h * 0.5,
                "\u2192", fontsize=18, ha="center", va="center",
            )

            ax_out = fig.add_axes([0.55, grid_bot, 0.36, grid_h])
            _render_grid_on_ax(ax_out, out_grid)
            ax_out.set_title("Output", fontsize=8, pad=3)

        pdf.savefig(fig)
        plt.close(fig)


def _format_program(pc_instruction: str, max_lines: int = 40) -> str:
    """Return the program text, trimming if too many lines."""
    lines = pc_instruction.splitlines()
    if len(lines) > max_lines:
        half = max_lines // 2
        lines = lines[:half] + ["  ..."] + lines[-(half - 1):]
    return "\n".join(lines)


def _format_vars(state: Dict[str, Any]) -> str:
    lvars = state.get("local_variables", {})
    if not lvars:
        return "(empty)"
    parts = []
    for k, v in lvars.items():
        parts.append(f"{k} = {v!r}")
    return "\n".join(parts)


def _format_memory_summary(memory: Dict[str, Any]) -> str:
    parts = []
    for k, v in memory.items():
        if k == "grid":
            h = len(v) if isinstance(v, list) else "?"
            w = len(v[0]) if isinstance(v, list) and v else "?"
            parts.append(f"grid: {h}x{w}")
        else:
            parts.append(f"{k} = {v!r}")
    return "\n".join(parts)


def _render_step_page(
    fig: plt.Figure,
    step,
    total_steps: int,
    instruction: str = "",
) -> None:
    """Render a single step onto *fig* (letter-size page)."""
    fig.clf()

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    action_str = step.interpreter_action or "(none)"
    if len(action_str) > 80:
        action_str = action_str[:77] + "..."
    fig.text(
        0.05, 0.97,
        f"Step {step.step_number} / {total_steps - 1}",
        fontsize=13, fontweight="bold", va="top",
    )
    fig.text(
        0.05, 0.945,
        f"Action: {action_str}",
        fontsize=8, fontfamily="monospace", va="top",
    )
    fig.text(
        0.05, 0.93,
        f"PC decision: {step.pc_action}",
        fontsize=8, va="top", color="#555555",
    )

    # ------------------------------------------------------------------
    # Program text (upper portion)
    # ------------------------------------------------------------------
    program_text = _format_program(step.pc_instruction)
    fig.text(
        0.05, 0.905,
        "Program:",
        fontsize=9, fontweight="bold", va="top",
    )
    fig.text(
        0.06, 0.89,
        program_text,
        fontsize=6, fontfamily="monospace", va="top",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FFFFF0",
            edgecolor="#CCCCCC",
            linewidth=0.5,
        ),
    )

    # ------------------------------------------------------------------
    # Grid (lower-left)
    # ------------------------------------------------------------------
    grid = step.post_memory.get("grid")
    if grid is not None:
        ax = fig.add_axes([0.06, 0.04, 0.42, 0.38])
        arr = np.array(grid, dtype=int)
        ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        rows, cols = arr.shape
        ax.set_xticks([x - 0.5 for x in range(1, cols + 1)])
        ax.set_yticks([y - 0.5 for y in range(1, rows + 1)])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which="both", color="white", linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Grid (post-step)", fontsize=8, pad=4)

    # ------------------------------------------------------------------
    # Local variables (lower-right, upper)
    # ------------------------------------------------------------------
    fig.text(
        0.55, 0.42,
        "Local Variables:",
        fontsize=9, fontweight="bold", va="top",
    )
    vars_text = _format_vars(step.post_state)
    fig.text(
        0.56, 0.40,
        vars_text,
        fontsize=7, fontfamily="monospace", va="top",
        linespacing=1.3,
    )

    # ------------------------------------------------------------------
    # Memory summary (lower-right, lower)
    # ------------------------------------------------------------------
    fig.text(
        0.55, 0.18,
        "Memory:",
        fontsize=9, fontweight="bold", va="top",
    )
    mem_text = _format_memory_summary(step.post_memory)
    fig.text(
        0.56, 0.16,
        mem_text,
        fontsize=7, fontfamily="monospace", va="top",
        linespacing=1.3,
    )

    # ------------------------------------------------------------------
    # Executor message
    # ------------------------------------------------------------------
    pc_msg = step.executor_result.get("pc_message", "")
    if pc_msg:
        fig.text(
            0.55, 0.07,
            f"Executor: {pc_msg}",
            fontsize=7, va="top", color="#333333", style="italic",
        )


def generate_pdf_report(
    execution_log,
    output_path: str,
    *,
    instruction: str = "",
    max_steps: Optional[int] = None,
    task=None,
) -> str:
    """Generate a PDF report from an :class:`ExecutionLog`.

    Parameters
    ----------
    execution_log : ExecutionLog
        The log to visualize.
    output_path : str
        Destination PDF file path.
    instruction : str
        The NL program text (for the title page).
    max_steps : int | None
        If given, only render the first *max_steps* steps.
    task : ARCTask | None
        If provided, a task-visualization page showing the train/test I/O
        pairs is inserted right after the title page.

    Returns
    -------
    str
        The path to the generated PDF.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    steps = execution_log.steps
    if max_steps is not None:
        steps = steps[:max_steps]
    total = len(steps)

    with PdfPages(output_path) as pdf:
        # --- Title page ---
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.text(
            0.5, 0.85,
            f"ARC Solver Execution Report",
            fontsize=18, fontweight="bold", ha="center", va="top",
        )
        fig.text(
            0.5, 0.80,
            f"Task: {execution_log.task_id}    Steps: {total}",
            fontsize=12, ha="center", va="top",
        )
        instr = instruction or execution_log.instruction
        if instr:
            fig.text(
                0.08, 0.72,
                "Instruction:",
                fontsize=10, fontweight="bold", va="top",
            )
            fig.text(
                0.10, 0.68,
                instr,
                fontsize=7, fontfamily="monospace", va="top",
                linespacing=1.3,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#F8F8FF",
                    edgecolor="#CCCCCC",
                ),
            )
        pdf.savefig(fig)
        plt.close(fig)

        # --- Task visualisation (train / test pairs) ---
        if task is not None:
            train_ex = getattr(task, "trainingExamples", [])
            test_ex = getattr(task, "testExamples", [])
            if train_ex or test_ex:
                _render_task_pages(pdf, execution_log.task_id, train_ex, test_ex)

        # --- One page per step ---
        for step in steps:
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            _render_step_page(fig, step, total, instruction=instr)
            pdf.savefig(fig)
            plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a PDF report from an execution log.",
    )
    parser.add_argument("log_file", help="Path to the execution log JSON")
    parser.add_argument("output_pdf", help="Path to the output PDF")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Only render the first N steps")
    parser.add_argument("--data_folder", default=None,
                        help="Path to ARC data folder (enables task visualisation page)")
    parser.add_argument("--dataset_set", default="training",
                        help="Dataset split (default: training)")
    args = parser.parse_args()

    from arc.log.step_logger import ExecutionLog

    log = ExecutionLog.load(args.log_file)

    task = None
    if args.data_folder:
        from arc.data.ARCTask import ARCTask
        task = ARCTask(folder=args.data_folder, set=args.dataset_set).load(log.task_id)

    generate_pdf_report(log, args.output_pdf, max_steps=args.max_steps, task=task)
    print(f"Report saved to {args.output_pdf} ({len(log.steps)} steps)")


if __name__ == "__main__":
    main()
