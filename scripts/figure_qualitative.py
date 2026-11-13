"""Qualitative-figure drawing shared by figures.py: flights on their last frame and in side view.

Each flight gets one column: optionally its last broadcast frame with the 3D paths projected by that
frame's camera, then a side view (height against distance along the flight). Every method is a
solid line (colour and marker carry identity); the ground truth is black dashed.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Okabe-Ito (published colour-blind-safe set); each arm keeps its colour in every figure and
# also has its own marker or line style, so identity never rests on colour alone.
ARMS = {
    "ArcPrior-cut": ("#D55E00", "o", "-"),
    "ArcPrior-full": ("#E69F00", "s", "-"),
    "LM-fitg": ("#009E73", "^", "-"),
    "B1": ("#0072B2", "D", "-"),
    "M5": ("#999999", "v", "-"),
}
# Arm names to the paper's notation (figures.py adds its own arms and labels).
DISPLAY = {"ArcPrior-cut": "ArcPrior-cut", "ArcPrior-full": "ArcPrior Base", "LM-fitg": "LM-fitg",
           "B1": "Grad et al.", "M5": "kNN-flights"}
# Side-view limits per panel (distance, height in m); figures.py sets them per figure.
QUALITATIVE_LIMITS = ((30, 9), (36, 9), (30, 9), (18, 9))
QUALITATIVE_PAD = 0.07
# Where each side view prints the methods' errors (axes fraction, alignment).
QUALITATIVE_TEXT = (0.03, 0.97, "left", "top")
# Tick spacing (distance, height) in m; None: 10 / 5 m and 2 m as before.
QUALITATIVE_TICKS = None
# Arms drawn in the qualitative figure, in legend order.
QUALITATIVE_ARMS = ("ArcPrior-full", "B1")
QUALITATIVE_COMPACT = False
# Figure height (in) with and without the frame row.
QUALITATIVE_HEIGHT = (3.05, 1.85)
# Ground truth is the one dashed line, black in the frame and in the side view.
GT_STYLE = {"color": "black", "ls": (0, (3, 1.5))}


def style() -> None:
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
                         "lines.linewidth": 1.4, "lines.markersize": 3.5, "legend.frameon": False,
                         "pdf.fonttype": 42, "axes.titlesize": 7.5, "axes.labelsize": 7.5})


def frame_view(ax, flight: dict) -> None:
    """The flight's last broadcast frame with the 3D paths projected by that frame's camera."""
    from figure_frames import camera, project, read_frame
    image = read_frame(flight["half"], flight["end"])
    height, width = image.shape[:2]
    cam = camera(flight["half"], flight["end"])
    ax.imshow(image)
    gt_px = project(flight["gt"], cam, width, height)
    ax.plot(gt_px[:, 0], gt_px[:, 1], lw=0.9, **GT_STYLE)
    for arm in QUALITATIVE_ARMS:
        colour, _, line = ARMS[arm]
        path_px = project(flight[arm], cam, width, height)
        ax.plot(path_px[:, 0], path_px[:, 1], color=colour, ls=line, lw=0.7)
    # A 16:9 crop around the track that always lies inside the frame, so every panel has the
    # same shape and size.
    ok = np.isfinite(gt_px).all(axis=1)
    centre = gt_px[ok].mean(axis=0)
    crop_w = max(np.ptp(gt_px[ok, 0]) * 1.3, np.ptp(gt_px[ok, 1]) * 1.3 * 16 / 9, 640.0)
    crop_w = min(crop_w, width, height * 16 / 9)
    crop_h = crop_w * 9 / 16
    x0 = float(np.clip(centre[0] - crop_w / 2, 0, width - crop_w))
    y0 = float(np.clip(centre[1] - crop_h / 2, 0, height - crop_h))
    ax.set_xlim(x0, x0 + crop_w)
    ax.set_ylim(y0 + crop_h, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def draw_qualitative(picks: list, titles: list, has_frames: bool, target: Path) -> None:
    """One column per flight: optionally its end frame with projected paths, then a side view.

    Each side view has its own axis limits, so every flight fills its panel.
    """
    rows = 2 if has_frames else 1
    fig, grid = plt.subplots(rows, 4, figsize=(6.9, QUALITATIVE_HEIGHT[0] if has_frames
                                                  else QUALITATIVE_HEIGHT[1]),
                             gridspec_kw={"height_ratios": [1, 1.3]} if has_frames else None,
                             squeeze=False)
    axes = grid[-1]
    if has_frames:
        for ax, flight in zip(grid[0], picks):
            frame_view(ax, flight)
    sides = []
    for flight in picks:
        gt = flight["gt"]
        ok = np.isfinite(gt[:, 0])
        start = gt[ok][0, :2]
        heading = gt[ok][-1, :2] - start
        heading = heading / np.linalg.norm(heading)
        paths = {"gt": gt, **{arm: flight[arm] for arm in QUALITATIVE_ARMS}}
        sides.append({name: ((path[:, :2] - start) @ heading, path[:, 2])
                      for name, path in paths.items()})
    for k, (ax, flight, title, side) in enumerate(zip(axes, picks, titles, sides)):
        ax.plot(*side["gt"], lw=1.2, label="Ground Truth", **GT_STYLE)
        lines = []
        for arm in QUALITATIVE_ARMS:
            colour, _, line = ARMS[arm]
            ax.plot(*side[arm], color=colour, ls=line, lw=0.9, label=DISPLAY[arm])
            lines.append(f"{DISPLAY[arm]} {flight[f'{arm}_rmse']:.2f} m")
        if QUALITATIVE_COMPACT:     # one line, RMSE in legend order
            lines = ["RMSE " + " / ".join(f"{flight[f'{arm}_rmse']:.2f}"
                                          for arm in QUALITATIVE_ARMS) + " m"]
        x_text, y_text, ha, va = QUALITATIVE_TEXT
        ax.text(x_text, y_text, "\n".join(lines), transform=ax.transAxes, fontsize=5.4, ha=ha,
                va=va, linespacing=1.3, color="#333333")
        title_ax = grid[0][k] if has_frames else ax
        title_ax.set_title(f"{title}, Apex {flight['apex']:.1f} m", fontsize=6.5, pad=3)
        limits = QUALITATIVE_LIMITS[k]
        if len(limits) == 4:            # explicit (x0, x1, z0, z1)
            x0, x_max, z0, z_max = limits
            ax.set_xlim(x0, x_max)
            ax.set_ylim(z0, z_max)
        else:
            x_max, z_max = limits
            # The same small margin on both sides keeps lines off the axes; ticks stay in 0..max.
            ax.set_xlim(-QUALITATIVE_PAD * x_max, (1 + QUALITATIVE_PAD) * x_max)
            ax.set_ylim(-QUALITATIVE_PAD * z_max, (1 + QUALITATIVE_PAD) * z_max)
        x_step, z_step = QUALITATIVE_TICKS or (10 if x_max > 20 else 5, 2)
        ax.set_xticks(np.arange(0, x_max + 1e-6, x_step))
        ax.set_yticks(np.arange(0, z_max + 1e-6, z_step))
        ax.set_xlabel("Distance (m)")
    axes[0].set_ylabel("Height (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=len(QUALITATIVE_ARMS) + 1,
                        fontsize=6.5,
                        bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6)
    for line in legend.get_lines():
        line.set_linewidth(1.6)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.4, rect=(0, 0, 1, 0.93 if has_frames else 0.88))
    fig.savefig(target, dpi=300)
    plt.close(fig)
