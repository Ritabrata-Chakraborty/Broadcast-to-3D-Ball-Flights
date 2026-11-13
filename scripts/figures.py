"""The paper's result figures, drawn from outputs/eval (test split, frozen arms).

Writes outputs/figures/: partial.pdf (median capped RMSE per share seen, GT and real inputs,
broadcast and static), cdf.pdf and apex.pdf (per-flight error distribution and its median by apex,
broadcast), scenarios.pdf (median per view), qualitative{,_gt}_noframes.pdf with the chosen flights
(qualitative{,_gt}_flights.csv). With --frames it also writes qualitative.pdf, qualitative_gt.pdf
and the 2D-track figures tracks_{broadcast,static}.pdf; these hold LP video frames (never commit).
Every method is a solid line (colour and marker carry identity), the ground truth black dashed.
Run in the ball3d env: scripts/py scripts/figures.py [--frames]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config  # noqa: E402
import report as R  # noqa: E402
import figure_qualitative as F3  # noqa: E402
from arcprior.metrics import PART_SHARES, e_full, in_scope, part_population  # noqa: E402

OUT = Config.FIGURES_DIR
PRED = Config.EVAL_ROOT / "pred"
DATA = Config.EVAL_DATA
CAP_M = 10.0
# Okabe-Ito colours, one per method in every figure, with its own marker.
STYLE = {
    "ArcPrior Base": ("#E69F00", "s"),
    "ArcPrior ft": ("#CC79A7", "P"),
    "ArcPrior + heads ft (det,cal)": ("#D55E00", "o"),
    "LM-fitg": ("#009E73", "^"),
    "LM-fitg, heads (det,cal)": ("#56B4E9", "v"),
    "kNN-flights": ("#999999", "x"),
    "Grad et al.": ("#0072B2", "D"),
    "SNv3D-size (own boxes)": ("#000000", "*"),
}
SHOW = {"GT": ("ArcPrior Base", "LM-fitg", "kNN-flights", "Grad et al."),
        "Real": ("ArcPrior + heads ft (det,cal)", "ArcPrior Base", "LM-fitg, heads (det,cal)",
                 "LM-fitg", "kNN-flights", "Grad et al.")}
SHOWN: dict = {}   # display names equal the arm names
VIEWS = ("broadcast", "camera01", "camera02", "camera03", "camera04", "camera05")


def sets() -> dict:
    return json.loads((Config.EVAL_ROOT / "freeze.json").read_text())["sets"]


def axis_break(ax, low: float = 2.0, cut: float = 2.75) -> None:
    """y from 0 with a break: the axis shows 0 at its bottom, skips 0-3 m and resumes at 3 m."""
    ax.set_ylim(low, CAP_M + 0.3)
    ax.set_yticks([low, 4, 6, 8, 10])
    ax.set_yticklabels(["0", "4", "6", "8", "10"])
    kw = {"transform": ax.get_yaxis_transform(), "color": "black", "lw": 0.7, "clip_on": False}
    for dy in (-0.12, 0.12):
        ax.plot([-0.02, 0.02], [cut + dy - 0.08, cut + dy + 0.08], **kw)


def share_curve(table: pd.DataFrame, method: str, scope: str) -> list[float]:
    """Median over the partial-flight population of each flight's capped RMSE at each share."""
    population = part_population()
    out = []
    for share in PART_SHARES:
        part = in_scope(table[(table["method"] == method) & np.isclose(table["share"], share)],
                        scope)
        part = part[part["group"].isin(population)]
        out.append(float(part.groupby("group")["rmse_cap"].mean().median()))
    return out


def partial_figure(frozen: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.8), sharey=True)
    for k, (column, scope) in enumerate((("GT", "broadcast"), ("GT", "static"),
                                         ("Real", "broadcast"), ("Real", "static"))):
        table = R.load_column(frozen, column, "part", "test")
        ax = axes[k]
        for method in SHOW[column]:
            if method not in set(table["method"]):
                continue
            colour, marker = STYLE[method]
            ax.plot(np.array(PART_SHARES) * 100, share_curve(table, method, scope), color=colour,
                    marker=marker, ls="-", label=SHOWN.get(method, method))
        ax.set_title(f"{column} Inputs, {scope.title()}")
        ax.set_xlabel("Flight Seen (%)")
        ax.set_xticks([30, 50, 70, 90])
        axis_break(ax)
    axes[0].set_ylabel("Median RMSE (m)")
    handles = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.0)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "partial.pdf")
    plt.close(fig)


def cdf_apex_figures(frozen: dict) -> None:
    """Two figures: the per-flight capped RMSE distribution (cdf.pdf) and its median by
    apex height (apex.pdf), broadcast, GT and real inputs side by side."""
    bins = [(1, 2), (2, 3), (3, 5), (5, 10), (10, 99)]
    labels = ["1-2", "2-3", "3-5", "5-10", ">10"]
    cdf_fig, cdf_axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    apex_fig, apex_axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    handles = {}
    for k, column in enumerate(("GT", "Real")):
        table = R.load_column(frozen, column, "full", "test")
        apex = table.groupby("group")["apex_true"].first()
        cdf_ax, apex_ax = cdf_axes[k], apex_axes[k]
        counts = None
        shown = [m for m in SHOW[column] if m != "kNN-flights"] + ["SNv3D-size (own boxes)"]
        for method in shown:
            values = e_full(table, method, "broadcast")
            colour, marker = STYLE[method]
            ordered = np.sort(values.to_numpy())
            line, = cdf_ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                                color=colour, lw=1.1, label=SHOWN.get(method, method))
            handles.setdefault(method, line)
            heights = apex.reindex(values.index)
            medians = [values[(heights >= lo) & (heights < hi)].median() for lo, hi in bins]
            counts = [int(((heights >= lo) & (heights < hi)).sum()) for lo, hi in bins]
            apex_ax.plot(range(len(bins)), medians, color=colour, marker=marker, lw=1.1)
        for ax in (cdf_ax, apex_ax):
            ax.set_title(f"{column} Inputs, Broadcast")
        cdf_ax.set_xscale("log")
        cdf_ax.set_xlim(0.1, CAP_M)
        cdf_ax.set_xticks([0.1, 0.3, 1, 3, 10])
        cdf_ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
        cdf_ax.set_xlabel("Flight RMSE (m)")
        apex_ax.set_xticks(range(len(bins)))
        apex_ax.set_xticklabels([f"{b}\n({c})" for b, c in zip(labels, counts)], fontsize=5.5)
        apex_ax.set_xlabel("Apex (m), Flights")
        apex_ax.set_ylim(0, CAP_M + 0.3)
    cdf_axes[0].set_ylabel("Share of Flights")
    apex_axes[0].set_ylabel("Median RMSE (m)")
    from matplotlib.lines import Line2D
    for fig, name in ((cdf_fig, "cdf.pdf"), (apex_fig, "apex.pdf")):
        proxies = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=1.1, markersize=3)
                   for m in handles]
        fig.legend(proxies, [SHOWN.get(m, m) for m in handles], loc="upper center", ncol=3,
                   fontsize=5.5,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.8)
        fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.8))
        fig.savefig(OUT / name)
        plt.close(fig)


def scenarios_figure() -> None:
    """Median capped RMSE per view: filled GT inputs, open real inputs, per scenario row of rows."""
    table = pd.read_csv(Config.EVAL_ROOT / "report" / "test" / "table_full.csv")
    series = (("GT", "ArcPrior Base", "GT: ArcPrior Base"),
              ("GT", "Grad et al.", "GT: Grad et al."),
              ("Real", "ArcPrior + heads ft (det,cal)", "Real: ArcPrior + heads ft (det,cal)"),
              ("Real", "ArcPrior Base", "Real: ArcPrior Base"),
              ("Real", "Grad et al.", "Real: Grad et al."))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    width = 0.15
    for k, (column, method, label) in enumerate(series):
        rows = table[(table["setting"] == column) & (table["method"] == method)]
        values = [float(rows[rows["scope"] == v]["median"].iloc[0]) for v in VIEWS]
        colour, marker = STYLE[method]
        filled = column == "GT"
        x = np.arange(len(VIEWS)) + (k - 2) * width
        ax.scatter(x, values, s=16, marker=marker, color=colour if filled else "white",
                   edgecolors=colour, lw=0.9, zorder=3, label=label)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_ylim(0.8, 12)
    ax.set_xticks(range(len(VIEWS)))
    ax.set_xticklabels(["bcast", "cam01", "cam02", "cam03", "cam04", "cam05"])
    ax.set_ylabel("Median RMSE (m)")
    ax.legend(loc="upper center", fontsize=5.5, ncol=2, bbox_to_anchor=(0.5, 1.3),
              columnspacing=0.8, handletextpad=0.2)
    ax.text(0.99, 0.03, "Filled: GT Inputs   Open: Real Inputs", transform=ax.transAxes,
            ha="right", fontsize=5.6, color="#444444")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "scenarios.pdf")
    plt.close(fig)


def frames_of(folder: str, method: str, fold: str) -> pd.DataFrame:
    table = pd.read_csv(PRED / folder / f"fold_{fold}_test_frames.csv.gz")
    table = table[(table["method"] == method) & (table["view"] == 0)].copy()
    for axis in "xyz":
        table[axis] = table[f"{axis}_predicted"] / Config.DM_PER_M
    return table


def collect_flights(arms: dict, data_set: str = "gt_full") -> list[dict]:
    """Broadcast test flights with each arm's per-frame path, capped RMSE and answered share."""
    flights = []
    for fold in sorted(Config.FOLDS):
        z = np.load(DATA / data_set / f"fold_{fold}" / "test.npz")
        tables = {}
        for arm, entries in arms.items():
            folder, method = entries[fold][0].split(":", 1)
            tables[arm] = frames_of(folder, method, fold)
        for i in np.flatnonzero(z["view"] == 0):
            n = int(z["n"][i])
            gt = z["xyz"][i, :n]
            frames = np.arange(n) + int(z["a"][i])
            record = {"fold": fold, "half": int(z["half"][i]), "group": int(z["group"][i]),
                      "gt": gt, "apex": float(np.nanmax(gt[:, 2])), "end": int(frames[-1]),
                      "masked": bool(z["track_masked"][i])}
            for arm, table in tables.items():
                rows = table[(table["group"] == record["group"]) &
                             (table["half"] == record["half"])].set_index("file_name")
                path = rows.reindex(frames)[["x", "y", "z"]].to_numpy()
                has_gt = np.isfinite(gt[:, 0])
                ok = np.isfinite(path[:, 0]) & has_gt
                record[arm] = path
                error = np.full(int(has_gt.sum()), CAP_M)
                error[ok[has_gt]] = np.minimum(np.linalg.norm(path[ok] - gt[ok], axis=1), CAP_M)
                record[f"{arm}_rmse"] = float(np.sqrt((error ** 2).mean()))
                record[f"{arm}_share"] = float(ok.sum() / max(has_gt.sum(), 1))
            flights.append(record)
    return flights


def pick_flights(flights: list[dict], arms: tuple, rank_arm: str, targets: tuple) -> tuple:
    """Among flights with an apex of at least 3 m that every arm answers on at least 80 % of the
    frames, the flight nearest each target percentile of rank_arm's capped RMSE."""
    pool = {f["group"]: f for f in flights
            if f["apex"] >= 3.0 and all(f[f"{arm}_share"] >= 0.8 for arm in arms)}
    rank = pd.Series({g: f[f"{rank_arm}_rmse"] for g, f in pool.items()}).rank(pct=True) * 100
    groups = [int((rank - target).abs().idxmin()) for target in targets]
    titles = [f"({letter}) {rank[g]:.0f}th Percentile" for letter, g in zip("abcd", groups)]
    return [pool[g] for g in groups], titles, rank, len(pool)


def draw_set(picks: list, titles: list, arms: tuple, labels: dict, stem: str,
             has_frames: bool) -> None:
    F3.DISPLAY.update(labels)
    F3.QUALITATIVE_ARMS = arms
    F3.QUALITATIVE_TICKS = (10.0, 2.0)   # x every 10 m, y every 2 m (user)
    F3.QUALITATIVE_TEXT = (0.5, 0.02, "center", "bottom")   # in the band below the pitch
    F3.QUALITATIVE_COMPACT = True
    # The first arm is drawn wider underneath, so it shows where a later arm overlaps it.
    F3.QUALITATIVE_LW = {arms[0]: (1.9, 1.4)}
    F3.QUALITATIVE_HEIGHT = (2.9, 1.8)
    F3.draw_qualitative(picks, titles, False, OUT / f"{stem}_noframes.pdf")
    if has_frames:
        F3.draw_qualitative(picks, titles, True, OUT / f"{stem}.pdf")


def write_picks(picks: list, titles: list, rank: pd.Series, arms: tuple, name: str) -> None:
    rows = [{"title": t, "fold": f["fold"], "half": f["half"], "group": f["group"],
             "masked_track": f["masked"], "rank_pct": round(float(rank[f["group"]]), 1),
             "apex_m": round(f["apex"], 2),
             **{f"{F3.DISPLAY[a]}_rmse_m": round(f[f"{a}_rmse"], 3) for a in arms}}
            for f, t in zip(picks, titles)]
    pd.DataFrame(rows).to_csv(OUT / name, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


FIG5_REPLACE = {2: 354, 3: 396}     # panel index -> flight (group)
# Side-view limits per panel (x0, x1, z0, z1) in metres, set by the user (2026-11-13).
FIG5_LIMITS = ((-5.0, 10.0, -2.0, 4.0), (-5.0, 30.0, -2.0, 6.0), (-5.0, 20.0, -2.0, 10.0),
               (-5.0, 35.0, -2.0, 6.0))


def qualitative_figure(frozen: dict, has_frames: bool) -> list[dict]:
    """Two figures on the same four broadcast flights: among flights with an apex of at least 3 m
    that ArcPrior Base, ArcPrior + heads ft (det,cal) and Grad et al. (real inputs, seed 0) all
    answer on at least 80 % of the frames, those nearest the 25th and 50th percentile of the
    error of ArcPrior + heads ft and, by hand (FIG5_REPLACE), two flights near the 75th and 100th.
    Main figure: those three methods from video; supplement: the same flights with GT inputs
    (ArcPrior Base, Grad et al.)."""
    colour = {"real-base": STYLE["ArcPrior Base"][0],
              "heads-ft": STYLE["ArcPrior + heads ft (det,cal)"][0],
              "real-grad": STYLE["Grad et al."][0]}
    for arm, c in colour.items():
        F3.ARMS[arm] = (c, "o", "-")
    real = {"real-base": frozen["real_full"]["ArcPrior base"],
            "heads-ft": frozen["heads_full"]["ArcPrior heads_ft"],
            "real-grad": frozen["real_full"]["Grad et al."]}
    arms = ("real-base", "heads-ft", "real-grad")
    picks, titles, rank, size = pick_flights(collect_flights(real), arms, "heads-ft",
                                             (25, 50, 75, 100))
    # User, 2026-11-13: panels (c) and (d) replaced by the next flights nearest the 75th and
    # 100th percentiles, in Fig. 5 and the supplement figures alike.
    pool = {f["group"]: f for f in collect_flights(real)}
    shown, shown_titles = list(picks), list(titles)
    for k, group in FIG5_REPLACE.items():
        shown[k] = pool[group]
        shown_titles[k] = f"({'abcd'[k]}) {rank[group]:.0f}th Percentile"
    F3.QUALITATIVE_LIMITS = FIG5_LIMITS
    draw_set(shown, shown_titles, arms, {"real-base": "ArcPrior Base (Real)",
                                   "heads-ft": "ArcPrior + Heads ft (Real)",
                                   "real-grad": "Grad et al. (Real)"}, "qualitative", has_frames)
    write_picks(shown, shown_titles, rank, arms, "qualitative_flights.csv")
    print(f"pool: {size} flights")
    gt = {"ArcPrior-full": frozen["gt_full"]["ArcPrior base"],
          "B1": frozen["gt_full"]["Grad et al."]}
    gt_arms = ("ArcPrior-full", "B1")
    by_flight = {(f["fold"], f["half"], f["group"]): f for f in collect_flights(gt)}
    gt_picks = [by_flight[(f["fold"], f["half"], f["group"])] for f in shown]
    F3.QUALITATIVE_LIMITS = FIG5_LIMITS
    draw_set(gt_picks, shown_titles, gt_arms, {"ArcPrior-full": "ArcPrior Base (GT)",
                                         "B1": "Grad et al. (GT)"}, "qualitative_gt",
             has_frames)
    write_picks(gt_picks, shown_titles, rank, gt_arms, "qualitative_gt_flights.csv")
    return shown


# ---------------------------------------------------------------- 2D ball tracks (frames)


TRACKS = {   # label: (colour, marker); the GT is the black dashed line
    "YOLO": ("#999999", "x"),
    "YOLO + Tracker": ("#0072B2", "o"),
    "YOLO + Tracker ft": ("#D55E00", "s"),
}
TRACK_CAP_PX = 100.0


def flight_tracks(fold: str, view: int, half: int, group: int) -> dict | None:
    """Pixel tracks (1920 px wide) of one test view-flight: GT, YOLO top box, tracker, ft."""
    import input_quality as I
    data = {name: np.load(DATA / f"{name}_src" / f"fold_{fold}" / "test.npz")
            for name in ("gt", "real", "heads")}
    ref = data["gt"]
    hits = np.flatnonzero((ref["view"] == view) & (ref["half"] == half) & (ref["group"] == group))
    if not len(hits):
        return None
    i = int(hits[0])
    n = int(ref["n"][i])
    sub = {k: ref[k][i:i + 1] for k in ("view", "half", "a", "n", "uv_gt")}
    tracks = {"GT": ref["uv_gt"][i, :n].astype(float),
              "YOLO": I.top_detection(sub, I.extract_dir("published", fold))[0, :n],
              "YOLO + Tracker": data["real"]["uv_det"][i, :n].astype(float),
              "YOLO + Tracker ft": data["heads"]["uv_det"][i, :n].astype(float)}
    return {"view": view, "half": half, "group": group, "a": int(ref["a"][i]), "n": n,
            "rot": ref["rot"][i, :n].astype(float), "intr": ref["intr"][i, :n].astype(float),
            "tracks": tracks}


def to_last_frame(uv: np.ndarray, rot: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Move each frame's pixel into the last frame by the camera rotation (broadcast pans)."""
    def k_matrix(v):
        return np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1.0]])
    out = np.full_like(uv, np.nan)
    end = k_matrix(intr[-1]) @ rot[-1]
    for t in np.flatnonzero(np.isfinite(uv).all(axis=1)):
        h = end @ rot[t].T @ np.linalg.inv(k_matrix(intr[t]))
        point = h @ np.r_[uv[t], 1.0]
        out[t] = point[:2] / point[2]
    return out


def view_frame(view: int, half: int, frame: int) -> np.ndarray:
    import cv2
    path = Config.DATA_ROOT / Config.LP / VIEWS[view] / Config.HALVES[half] / "input.mkv"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path} frame {frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def track_panels(flights: list[dict], titles: list[str], target: Path) -> None:
    """One row per flight or camera; one image column per track (the flight's last frame, the
    track and the GT moved into it), so no track hides another; last column: per-frame distance
    to the projected GT ball (px, capped), drawn in layers (wide Tracker under narrow Tracker ft,
    YOLO as thin crosses on top)."""
    # Exact layout (inches): images at the crop's 3:2 with small fixed gaps, then the plot.
    left, gap, plot_left, plot_w, right = 0.17, 0.04, 0.42, 1.2, 0.06
    img_w = (6.9 - left - right - plot_left - plot_w - gap * (len(TRACKS) - 1)) / len(TRACKS)
    img_h = img_w * 2 / 3
    top, row_gap, bottom = 0.42, 0.2, 0.32
    fig_h = top + len(flights) * img_h + (len(flights) - 1) * row_gap + bottom
    fig = plt.figure(figsize=(6.9, fig_h))
    grid = []
    for k in range(len(flights)):
        y = fig_h - top - img_h - k * (img_h + row_gap)
        xs = [left + c * (img_w + gap) for c in range(len(TRACKS))]
        xs.append(xs[-1] + img_w + plot_left)
        widths = [img_w] * len(TRACKS) + [plot_w]
        grid.append([fig.add_axes((x / 6.9, y / fig_h, w / 6.9, img_h / fig_h))
                     for x, w in zip(xs, widths)])
    layers = {"YOLO + Tracker": (3.2, 0.6, None, 3), "YOLO + Tracker ft": (0.9, 1.0, "s", 4),
              "YOLO": (0.4, 1.0, "x", 5)}     # line width, alpha, marker, z-order
    for k, (flight, title) in enumerate(zip(flights, titles)):
        image = view_frame(flight["view"], flight["half"], flight["a"] + flight["n"] - 1)
        height, width = image.shape[:2]
        scale = Config.IMAGE_WIDTH_PX / width
        moved = {name: to_last_frame(uv, flight["rot"], flight["intr"])
                 for name, uv in flight["tracks"].items()}
        gt = moved["GT"][np.isfinite(moved["GT"]).all(axis=1)]
        centre = gt.mean(axis=0)
        crop_w = max(np.ptp(gt[:, 0]) * 1.2, np.ptp(gt[:, 1]) * 1.2 * 3 / 2, 300.0)
        crop_w = min(crop_w, width * scale, height * scale * 3 / 2)
        crop_h = crop_w * 2 / 3
        x0 = float(np.clip(centre[0] - crop_w / 2, 0, width * scale - crop_w))
        y0 = float(np.clip(centre[1] - crop_h / 2, 0, height * scale - crop_h))
        for c, (name, (colour, marker)) in enumerate(TRACKS.items()):
            ax = grid[k][c]
            ax.imshow(image, extent=(0, width * scale, height * scale, 0))
            ax.plot(*moved["GT"].T, lw=0.9, **F3.GT_STYLE)
            ax.plot(*moved[name].T, color=colour, lw=0.5 if name == "YOLO" else 0.8,
                    marker=marker, ms=1.8, mew=0.5)
            ax.set_xlim(x0, x0 + crop_w)
            ax.set_ylim(y0 + crop_h, y0)
            ax.set_aspect("auto")       # the crop is already 3:2, like the axes
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(title, fontsize=6.5)
            if k == 0:
                ax.set_title(name, fontsize=6.5, pad=3)
        ax = grid[k][-1]
        steps = np.arange(flight["n"])
        for name, (colour, _) in TRACKS.items():
            line_w, alpha, marker, order = layers[name]
            gap = np.linalg.norm(flight["tracks"][name] - flight["tracks"]["GT"], axis=1)
            ax.plot(steps, np.minimum(gap, TRACK_CAP_PX), color=colour, lw=line_w, alpha=alpha,
                    marker=marker, ms=2.0, mew=0.5, zorder=order, solid_capstyle="round")
        ax.axhline(30.0, color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_ylim(-4, TRACK_CAP_PX + 4)
        ax.set_yticks([0, 30, 60, 100])
        ax.tick_params(labelsize=6)
        ax.set_ylabel("Distance (px)", fontsize=6.5)
        if k == 0:
            ax.set_title("Distance to GT", fontsize=6.5, pad=3)
        if k == len(flights) - 1:
            ax.set_xlabel("Frame in Flight", fontsize=6.5)
    handles = [plt.Line2D([], [], lw=1.2, label="Ground Truth", **F3.GT_STYLE)]
    handles += [plt.Line2D([], [], color=c, marker=m, lw=1.2, ms=3, label=name)
                for name, (c, m) in TRACKS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6)
    fig.savefig(target, dpi=300)
    plt.close(fig)


def track_figures(picks: list[dict]) -> None:
    """2D ball tracks on video frames: the four broadcast flights of the qualitative figure, and
    the first of them seen by all static cameras 01-05. Holds LP frames: never commit."""
    flights = [flight_tracks(p["fold"], 0, p["half"], p["group"]) for p in picks]
    titles = [f"({letter}) Broadcast" for letter in "abcd"]
    track_panels(flights, titles, OUT / "tracks_broadcast.pdf")
    # The first of the four flights that every static camera 01-05 sees.
    for pick in picks:
        static = [flight_tracks(pick["fold"], v, pick["half"], pick["group"])
                  for v in range(1, 6)]
        if all(f is not None for f in static):
            break
    print("static tracks: flight", pick["group"])
    static_titles = [f"Camera {v:02d}" for v, f in zip(range(1, 6), static) if f is not None]
    track_panels([f for f in static if f is not None], static_titles, OUT / "tracks_static.pdf")
    rows = [{"figure": fig, "view": f["view"], "half": f["half"], "group": f["group"],
             **{f"{name}_near_pct": round(100 * float(np.nanmean(np.linalg.norm(
                 f["tracks"][name] - f["tracks"]["GT"], axis=1) <= 30)), 1) for name in TRACKS}}
            for fig, group in (("broadcast", flights), ("static", static)) for f in group if f]
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", action="store_true", help="also write qualitative.pdf")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    F3.style()
    frozen = sets()
    partial_figure(frozen)
    cdf_apex_figures(frozen)
    scenarios_figure()
    picks = qualitative_figure(frozen, args.frames)
    if args.frames:
        track_figures(picks)


if __name__ == "__main__":
    main()
