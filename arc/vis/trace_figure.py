"""Compact execution-trace figure generator for LaTeX inclusion.

Produces a multi-page PDF where each execution step occupies one row,
showing the current instruction, the post-step grid, local variables,
and memory state.  Two primary use-cases:

1. **Selected steps** (thesis body) — pass ``--steps 0,5,20,50,122``
2. **Full trace** (appendix) — pass ``--all --compact``

Usage::

    # Selected steps for the thesis body
    python -m arc.vis.trace_figure results/29c11459/29c11459_log.json \\
        -o figures/trace_body.pdf --steps 0,5,20,50,122

    # Full trace for appendix
    python -m arc.vis.trace_figure results/29c11459/29c11459_log.json \\
        -o figures/trace_appendix.pdf --all --compact

    # Also emit per-page PNGs (handy for \\includegraphics)
    python -m arc.vis.trace_figure results/29c11459/29c11459_log.json \\
        -o figures/trace.pdf --all --compact --png
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ── ARC colour-map ──────────────────────────────────────────────────

ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 pink
    "#FF851B",  # 7 orange
    "#9d00ff",  # 8 purple
    "#870C25",  # 9 brown
]
ARC_CMAP = ListedColormap(ARC_COLORS, name="arc10")


# ── Page geometry (fractions of figure unless noted) ────────────────

PAGE_W, PAGE_H = 8.5, 11.0  # inches

ML = 0.048  # left margin
MR = 0.020  # right margin
MT = 0.035  # top margin
MB = 0.020  # bottom margin

HEADER_H = 0.025
COL_HEAD_H = 0.018

# Column boundaries
COL1_L = ML
COL1_R = 0.30
COL2_L = 0.315
COL2_R = 0.565
COL3_L = 0.580
COL3_R = 1.0 - MR

# Vertical separator x-positions
VSEP_X1 = (COL1_R + COL2_L) / 2
VSEP_X2 = (COL2_R + COL3_L) / 2


# ── Helpers ─────────────────────────────────────────────────────────

def _trunc(s: str, n: int = 50) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _wrap_lines(s: str, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(s.strip(), width))


def _render_grid(fig: plt.Figure, grid, bbox: Tuple[float, ...]) -> None:
    """Draw a small ARC grid in *bbox* = (x, y, w, h) in figure coords."""
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape

    target_w, target_h = bbox[2], bbox[3]
    grid_aspect = cols / rows
    box_real_w = target_w * PAGE_W
    box_real_h = target_h * PAGE_H
    box_aspect = box_real_w / box_real_h

    if grid_aspect > box_aspect:
        w_inch = box_real_w
        h_inch = w_inch / grid_aspect
    else:
        h_inch = box_real_h
        w_inch = h_inch * grid_aspect

    w = w_inch / PAGE_W
    h = h_inch / PAGE_H
    x = bbox[0] + (target_w - w) / 2
    y = bbox[1] + (target_h - h) / 2

    ax = fig.add_axes([x, y, w, h], zorder=5)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_xticks([c - 0.5 for c in range(1, cols + 1)])
    ax.set_yticks([r - 0.5 for r in range(1, rows + 1)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="white", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Row renderer ────────────────────────────────────────────────────

def _draw_row(
    fig: plt.Figure,
    step: Dict[str, Any],
    row_top: float,
    row_h: float,
    *,
    font_scale: float = 1.0,
    shade: bool = False,
):
    """Render one step as a horizontal row."""
    row_bot = row_top - row_h

    # Alternating row background
    if shade:
        fig.patches.append(FancyBboxPatch(
            (ML, row_bot), 1.0 - ML - MR, row_h,
            boxstyle="square,pad=0",
            facecolor="#F7F7FA", edgecolor="none",
            transform=fig.transFigure, zorder=0,
        ))

    # Top separator
    fig.add_artist(plt.Line2D(
        [ML, 1.0 - MR], [row_top, row_top],
        transform=fig.transFigure, color="#CCCCCC", linewidth=0.4,
    ))

    step_num = step["step_number"]
    instr = step["current_instruction"].strip()
    action = step.get("interpreter_action", "")
    pc_action = step.get("pc_action", "")
    post_mem = step.get("post_memory", {})
    post_state = step.get("post_state", {})
    ex_result = step.get("executor_result", {})
    grid = post_mem.get("grid")

    fs = lambda base: base * font_scale  # noqa: E731

    # ── Column 1: step / instruction / action / PC ──────────────────
    c1x = COL1_L + 0.006

    fig.text(c1x, row_top - 0.06 * row_h,
             f"Step {step_num}",
             fontsize=fs(7.5), fontweight="bold", va="top")

    fig.text(c1x, row_top - 0.22 * row_h,
             _wrap_lines(instr, 32),
             fontsize=fs(6.2), fontfamily="monospace", va="top",
             color="#222222", linespacing=1.15)

    fig.text(c1x, row_top - 0.58 * row_h,
             _trunc(f"\u2192 {action}", 38),
             fontsize=fs(5.2), fontfamily="monospace", va="top",
             color="#666666")

    fig.text(c1x, row_top - 0.76 * row_h,
             f"PC: {pc_action}",
             fontsize=fs(5.2), va="top", color="#999999")

    # ── Vertical separators ─────────────────────────────────────────
    for vx in (VSEP_X1, VSEP_X2):
        fig.add_artist(plt.Line2D(
            [vx, vx], [row_top, row_bot],
            transform=fig.transFigure, color="#E0E0E0", linewidth=0.3,
        ))

    # ── Column 2: grid thumbnail ────────────────────────────────────
    if grid is not None:
        gp = 0.06 * row_h
        _render_grid(fig, grid, (
            COL2_L + 0.008,
            row_bot + gp,
            COL2_R - COL2_L - 0.016,
            row_h - 2 * gp,
        ))
    else:
        fig.text((COL2_L + COL2_R) / 2, (row_top + row_bot) / 2,
                 "\u2014",
                 fontsize=12, ha="center", va="center", color="#CCCCCC")

    # ── Column 3: locals / memory / executor message ────────────────
    c3x = COL3_L + 0.006

    # Locals
    fig.text(c3x, row_top - 0.06 * row_h,
             "Locals:", fontsize=fs(5.5),
             fontweight="bold", va="top", color="#555555")

    local_vars = post_state.get("local_variables", {})
    if local_vars:
        parts = []
        for k, v in local_vars.items():
            parts.append(f"{k}={v}")
        vars_str = _trunc(", ".join(parts), 52)
    else:
        vars_str = "(empty)"
    fig.text(c3x, row_top - 0.20 * row_h,
             vars_str,
             fontsize=fs(5.3), fontfamily="monospace", va="top",
             linespacing=1.15)

    # Memory
    fig.text(c3x, row_top - 0.42 * row_h,
             "Memory:", fontsize=fs(5.5),
             fontweight="bold", va="top", color="#555555")

    mem_parts = []
    for k, v in post_mem.items():
        if k == "grid":
            gh = len(v) if isinstance(v, list) else "?"
            gw = len(v[0]) if isinstance(v, list) and v else "?"
            mem_parts.append(f"grid: {gh}\u00d7{gw}")
        else:
            mem_parts.append(f"{k}={v}")
    fig.text(c3x, row_top - 0.56 * row_h,
             _trunc(", ".join(mem_parts), 52),
             fontsize=fs(5.3), fontfamily="monospace", va="top")

    # Executor message
    pc_msg = ex_result.get("pc_message", "")
    if pc_msg:
        fig.text(c3x, row_top - 0.78 * row_h,
                 f"\u2192 {_trunc(pc_msg, 48)}",
                 fontsize=fs(4.8), va="top",
                 color="#777777", style="italic")



# ── Page renderer ───────────────────────────────────────────────────

def _make_page(
    steps_on_page: List[Tuple[int, Dict]],
    rows_per_page: int,
    row_h: float,
    header: str,
    page_num: int,
    total_pages: int,
    *,
    font_scale: float = 1.0,
) -> plt.Figure:
    """Render one page of trace rows."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Page header
    fig.text(ML, 1.0 - MT, header,
             fontsize=10 * font_scale, fontweight="bold", va="top")
    fig.text(1.0 - MR, 1.0 - MT,
             f"Page {page_num}/{total_pages}  "
             f"(steps {steps_on_page[0][0]}\u2013{steps_on_page[-1][0]})",
             fontsize=7 * font_scale, va="top", ha="right", color="#888888")

    # Column headers
    ch_y = 1.0 - MT - HEADER_H
    for label, x, ha in [
        ("Step / Instruction", COL1_L + 0.006, "left"),
        ("Grid", (COL2_L + COL2_R) / 2, "center"),
        ("State", COL3_L + 0.006, "left"),
    ]:
        fig.text(x, ch_y, label,
                 fontsize=6.5 * font_scale, fontweight="bold",
                 va="bottom", ha=ha, color="#666666")

    hdr_line_y = ch_y - 0.004
    fig.add_artist(plt.Line2D(
        [ML, 1.0 - MR], [hdr_line_y, hdr_line_y],
        transform=fig.transFigure, color="#999999", linewidth=0.6,
    ))

    content_top = hdr_line_y - 0.002

    for ri, (orig_idx, step) in enumerate(steps_on_page):
        r_top = content_top - ri * row_h

        # Skip indicator for non-consecutive steps
        if ri > 0:
            prev_idx = steps_on_page[ri - 1][0]
            if orig_idx > prev_idx + 1:
                gap = orig_idx - prev_idx - 1
                fig.text(
                    0.5, r_top + 0.003,
                    f"\u00b7\u00b7\u00b7 {gap} step{'s' if gap != 1 else ''}"
                    f" omitted \u00b7\u00b7\u00b7",
                    fontsize=5.5 * font_scale, ha="center", va="bottom",
                    color="#BBBBBB", style="italic",
                )

        _draw_row(
            fig, step, r_top, row_h,
            font_scale=font_scale,
            shade=(ri % 2 == 1),
        )

    # Bottom separator
    bot_y = content_top - len(steps_on_page) * row_h
    fig.add_artist(plt.Line2D(
        [ML, 1.0 - MR], [bot_y, bot_y],
        transform=fig.transFigure, color="#CCCCCC", linewidth=0.4,
    ))

    return fig


# ── Public API ──────────────────────────────────────────────────────

def generate_trace_figure(
    log_path: str,
    output_path: str,
    *,
    step_indices: Optional[List[int]] = None,
    rows_per_page: int = 7,
    compact: bool = False,
    title: Optional[str] = None,
    emit_png: bool = False,
) -> str:
    """Generate a compact execution-trace PDF.

    Parameters
    ----------
    log_path : str
        Path to the ``*_log.json`` execution log.
    output_path : str
        Destination PDF path.
    step_indices : list[int] | None
        Step numbers to include.  ``None`` renders every step.
    rows_per_page : int
        Maximum rows per page (default 7; overridden to 10 in compact mode).
    compact : bool
        Smaller fonts and tighter spacing (recommended for appendix).
    title : str | None
        Override the default page header.
    emit_png : bool
        Also save each page as a separate PNG in the same directory.

    Returns
    -------
    str
        Path to the generated PDF.
    """
    with open(log_path) as f:
        data = json.load(f)

    steps = data["steps"]
    task_id = data.get("task_id", "unknown")

    if compact:
        rows_per_page = max(rows_per_page, 10)
        font_scale = 0.88
    else:
        font_scale = 1.0

    if step_indices is not None:
        selected = [(idx, steps[idx]) for idx in step_indices
                     if 0 <= idx < len(steps)]
    else:
        selected = list(enumerate(steps))

    if not selected:
        raise ValueError("No steps to render")

    usable = 1.0 - MT - MB - HEADER_H - COL_HEAD_H
    row_h = usable / rows_per_page
    total_pages = -(-len(selected) // rows_per_page)

    hdr = title or f"Execution Trace \u2014 Task {task_id}"

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    png_stem = os.path.splitext(os.path.basename(output_path))[0]

    with PdfPages(output_path) as pdf:
        for pg in range(total_pages):
            pg_start = pg * rows_per_page
            pg_steps = selected[pg_start: pg_start + rows_per_page]

            fig = _make_page(
                pg_steps, rows_per_page, row_h, hdr,
                pg + 1, total_pages,
                font_scale=font_scale,
            )

            pdf.savefig(fig)

            if emit_png:
                png_path = os.path.join(out_dir, f"{png_stem}_p{pg + 1}.png")
                fig.savefig(png_path, dpi=200, bbox_inches="tight")

            plt.close(fig)

    return output_path


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate a compact execution-trace PDF for LaTeX.",
    )
    p.add_argument("log_file", help="Path to *_log.json")
    p.add_argument("-o", "--output", required=True, help="Output PDF path")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--steps", type=str,
        help="Comma-separated step indices, e.g. 0,5,20,50,122")
    mode.add_argument(
        "--all", action="store_true",
        help="Include every step (appendix mode)")

    p.add_argument("--rows-per-page", type=int, default=7,
                    help="Rows per page (default 7; >=10 in compact)")
    p.add_argument("--compact", action="store_true",
                    help="Compact layout for appendix use")
    p.add_argument("--title", type=str, default=None,
                    help="Custom header title")
    p.add_argument("--png", action="store_true",
                    help="Also emit per-page PNGs")

    args = p.parse_args()

    indices = None
    if args.steps:
        indices = [int(x.strip()) for x in args.steps.split(",")]

    out = generate_trace_figure(
        args.log_file,
        args.output,
        step_indices=indices,
        rows_per_page=args.rows_per_page,
        compact=args.compact,
        title=args.title,
        emit_png=args.png,
    )

    with open(args.log_file) as f:
        n = len(json.load(f)["steps"])
    rendered = len(indices) if indices else n
    print(f"Trace figure saved to {out} ({rendered} steps rendered)")


if __name__ == "__main__":
    main()
