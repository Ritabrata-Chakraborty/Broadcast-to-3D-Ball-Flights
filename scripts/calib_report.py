"""Report of the calibration comparison: decision rule, tables and plots as images.

Runs at any point of the queue: results not scored yet stay blank and the verdict stays pending.
Output in outputs/calibration/: RESULTS.md, summary.csv, comparison.csv, ablations.csv, speed.csv,
figures/*.png and tables/*.png.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import Config  # noqa: E402

INDEPENDENT = ("WC14-test", "TSWC-test")
DERIVED = ("ISSIA", "LP-broadcast", "LP-static")
FOCAL_DATASETS = INDEPENDENT + DERIVED
LINE_ANNOTATED = ("SN22-test-center", "SN23-test", "WC14-test")
ALL_DATASETS = ("SN22-test-center", "SN23-test", "WC14-test", "TSWC-test") + DERIVED
PAPER_ZERO_SHOT = {"SN23-test": 44.8, "SN22-test-center": 70.1}
PAPER_FINE_TUNED = {"SN23-test": 56.2, "SN22-test-center": 75.8}
FAMILIES = {"pnlcalib": ("pnlcalib", "pnlcalib_mv"),
            "soccermaster": ("soccermaster", "soccermaster_d1")}
NAMES = {"pnlcalib": "PnLCalib SV", "pnlcalib_mv": "PnLCalib MV", "soccermaster": "SoccerMaster",
         "soccermaster_d1": "SoccerMaster, PnLCalib peaks",
         "pnlcalib_ft_wc14": "PnLCalib fine-tuned WC14",
         "pnlcalib_ft_tswc": "PnLCalib fine-tuned TSWC"}
COLORS = {"pnlcalib": "#8fbcf0", "pnlcalib_mv": "#1c5cab", "soccermaster": "#f5a983",
          "soccermaster_d1": "#b8451a"}
# Mirrors scripts/run_calibration.sh, so these rows stay visible until its solve stage writes them.
EXTRA_RUNS = [{"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": d,
               "label": "PnLCalib SV, inference.py thresholds"} for d in FOCAL_DATASETS] + [
    {"model": "pnlcalib_ft_wc14", "tag": "kp0.1274_ln0.1439", "dataset": "WC14-test",
     "label": "PnLCalib fine-tuned WC14"},
    {"model": "pnlcalib_ft_tswc", "tag": "kp0.2432_ln0.8482", "dataset": "TSWC-test",
     "label": "PnLCalib fine-tuned TSWC"}]
INK, MUTED, GRID, SURFACE, HEADER = "#2b2b2a", "#6b6a63", "#e4e3de", "#fcfcfb", "#efeeea"
OUT = Config.CALIB_ROOT


# ---------------------------------------------------------------- Metrics


def load_frames(model: str, tag: str, dataset: str) -> pd.DataFrame | None:
    """Scores of one run, or None if it has not been scored yet."""
    path = OUT / "frames" / model / tag / f"{dataset}.csv.gz"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"id": str}, low_memory=False)


def focal_errors(frames: pd.DataFrame) -> pd.Series:
    """Relative focal error on images with a reference; no camera counts as infinite."""
    with_ref = frames.loc[frames["f_ref"].notna()]
    error = (with_ref["f_pred"] / with_ref["f_ref"] - 1.0).abs()
    return error.where(with_ref["ok"].astype(bool)).fillna(np.inf)


def final_score(frames: pd.DataFrame) -> float:
    """Official FS: completeness times mean JaC5 over images with a camera."""
    is_ok = frames["ok"].astype(bool)
    return float(is_ok.mean() * frames.loc[is_ok, "jac5"].mean())


def cluster_ids(frames: pd.DataFrame, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Resampling cluster and stratum per image: see the plan's decision rule."""
    if dataset == "TSWC-test":
        return frames["id"].str.split("__").str[1].to_numpy(), np.zeros(len(frames), int)
    if dataset == "ISSIA":
        return (frames["unit"] + "/" + frames["id"]).to_numpy(), frames["unit"].to_numpy()
    if dataset.startswith("LP"):
        return (frames["unit"] + "/" + frames["clip"].astype(str)).to_numpy(), \
            np.zeros(len(frames), int)
    return frames["id"].to_numpy(), np.zeros(len(frames), int)


def resample_indices(clusters: np.ndarray, strata: np.ndarray, rng: np.random.Generator):
    """Yield image indices for each paired bootstrap draw."""
    groups: dict = {}
    for i, (cluster, stratum) in enumerate(zip(clusters, strata)):
        groups.setdefault(stratum, {}).setdefault(cluster, []).append(i)
    members = {s: [np.array(v) for v in g.values()] for s, g in groups.items()}
    for _ in range(Config.CALIB_BOOTSTRAP_DRAWS):
        picks = []
        for parts in members.values():
            picks += [parts[j] for j in rng.integers(len(parts), size=len(parts))]
        yield np.concatenate(picks)


def interval(draws: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval that stays defined when some draws are infinite."""
    return (float(np.percentile(draws, 2.5, method="lower")),
            float(np.percentile(draws, 97.5, method="higher")))


def paired_median_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, strata: np.ndarray
                       ) -> tuple[float, float, float]:
    """Median(a) - median(b) with its paired interval; infinite errors sort as failures."""
    rng = np.random.default_rng(Config.CALIB_SEED)
    with np.errstate(invalid="ignore"):
        draws = np.array([np.median(a[i]) - np.median(b[i])
                          for i in resample_indices(clusters, strata, rng)])
        point = np.median(a) - np.median(b)
    draws = np.nan_to_num(draws, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return (0.0 if np.isnan(point) else float(point), *interval(draws))


def paired_fs_diff(a: pd.DataFrame, b: pd.DataFrame, dataset: str) -> tuple[float, float, float]:
    """FS(a) - FS(b) with its paired interval over images."""
    check_paired(a, b, dataset)
    rng = np.random.default_rng(Config.CALIB_SEED)
    clusters, strata = cluster_ids(a, dataset)

    def score(ok: np.ndarray, jac: np.ndarray, index: np.ndarray) -> float:
        return float(ok[index].mean() * (jac[index].sum() / max(ok[index].sum(), 1)))

    ok_a, ok_b = a["ok"].to_numpy(bool), b["ok"].to_numpy(bool)
    jac_a, jac_b = a["jac5"].fillna(0).to_numpy(), b["jac5"].fillna(0).to_numpy()
    draws = np.array([score(ok_a, jac_a, i) - score(ok_b, jac_b, i)
                      for i in resample_indices(clusters, strata, rng)])
    return (final_score(a) - final_score(b), *interval(draws))


def check_paired(a: pd.DataFrame, b: pd.DataFrame, dataset: str) -> None:
    if a["id"].tolist() != b["id"].tolist():
        raise ValueError(f"{dataset}: models were not scored on the same images")


def compare_focal(a: pd.DataFrame, b: pd.DataFrame, dataset: str) -> dict:
    """Paired comparison of two models' focal errors on the same images."""
    check_paired(a, b, dataset)
    has_ref = a["f_ref"].notna().to_numpy()
    clusters, strata = cluster_ids(a, dataset)
    point, low, high = paired_median_diff(focal_errors(a).to_numpy(), focal_errors(b).to_numpy(),
                                          clusters[has_ref], strata[has_ref])
    return {"dataset": dataset, "metric": "median focal error", "diff": point, "ci_low": low,
            "ci_high": high, "winner": winner_of(low, high, lower_is_better=True)}


def winner_of(low: float, high: float, lower_is_better: bool) -> str:
    """Winner of pnlcalib minus soccermaster, or 'none' if the interval contains 0."""
    if low <= 0 <= high:
        return "none"
    first_lower = high < 0
    return "pnlcalib" if first_lower == lower_is_better else "soccermaster"


def decide(comparison: pd.DataFrame) -> dict:
    """Plan's decision rule, in its order of tiers; pending until every comparison exists."""
    if comparison["winner"].isna().any():
        return {"winner": "pending", "decided_by": "results still being scored", "tiers": []}
    tiers = (("independent focal references", INDEPENDENT, "median focal error"),
             ("dataset camera focal references", DERIVED, "median focal error"),
             ("line fit on annotated SoccerNet", ("SN22-test-center", "SN23-test"), "FS"))
    log = []
    for name, datasets, metric in tiers:
        rows = comparison.loc[comparison["dataset"].isin(datasets)
                              & (comparison["metric"] == metric)]
        wins = rows["winner"].value_counts()
        pnl, soccer = int(wins.get("pnlcalib", 0)), int(wins.get("soccermaster", 0))
        log.append({"tier": name, "pnlcalib_wins": pnl, "soccermaster_wins": soccer})
        if pnl != soccer:
            return {"winner": "pnlcalib" if pnl > soccer else "soccermaster", "decided_by": name,
                    "tiers": log}
    return {"winner": "no detectable difference", "decided_by": "all tiers tied", "tiers": log}


def oracle_rows(data: dict) -> pd.DataFrame:
    """Focal error against the camera solved from annotated lines; supporting, not deciding."""
    rows = []
    for dataset in LINE_ANNOTATED:
        oracle = pd.read_csv(OUT / "oracle" / f"{dataset}.csv", dtype={"id": str})
        usable = oracle.loc[oracle["jac5"] >= Config.CALIB_ORACLE_MIN_JAC, ["id", "f_oracle"]]
        for model in Config.CALIB_MODELS:
            if data[model][dataset] is None:
                rows.append({"Dataset": dataset, "Model": NAMES[model], "Of": len(oracle)})
                continue
            frames = data[model][dataset].drop(columns="f_ref").merge(usable, on="id")
            errors = focal_errors(frames.rename(columns={"f_oracle": "f_ref"}))
            rows.append({"Dataset": dataset, "Model": NAMES[model], "Images": len(errors),
                         "Of": len(oracle), "Median error %": percent(errors.median()),
                         "Within 5 %": percent((errors <= 0.05).mean()),
                         "Within 10 %": percent((errors <= 0.10).mean())})
    return pd.DataFrame(rows)


def noise_floor() -> dict:
    """Disagreement of the two hand-annotated focal references on WC14, in percent."""
    from calib_common import annotated_homography, focal_from_plane
    width, height = 1280, 720
    oracle = pd.read_csv(OUT / "oracle" / "WC14-test.csv", dtype={"id": str})
    oracle = oracle.loc[oracle["ok"]]
    focal = [focal_from_plane(np.linalg.inv(annotated_homography("WC14-test", i)),
                              (width / 2.0, height / 2.0)) * Config.CALIB_EVAL_WH[0] / width
             for i in oracle["id"]]
    ratio = (oracle["f_oracle"].to_numpy() / np.array(focal) - 1.0) * 100.0
    ratio = np.abs(ratio[np.isfinite(ratio)])
    return {"images": int(len(ratio)), "median_pct": float(np.median(ratio)),
            "p90_pct": float(np.percentile(ratio, 90))}


def row_of(frames: pd.DataFrame | None, dataset: str, label: str) -> dict:
    """One summary row for a model on a dataset; only its names if it is not scored yet."""
    if frames is None:
        return {"model": label, "dataset": dataset}
    is_ok = frames["ok"].astype(bool)
    solved = frames.loc[is_ok]
    row = {"model": label, "dataset": dataset, "images": len(frames), "cr": is_ok.mean()}
    errors = focal_errors(frames)
    row.update(with_reference=len(errors), excluded=int(frames["f_ref"].isna().sum()))
    if len(errors):
        row.update(focal_median=float(errors.median()), within_5=float((errors <= 0.05).mean()),
                   within_10=float((errors <= 0.10).mean()),
                   focal_ratio_median=float((solved["f_pred"] / solved["f_ref"]).median()))
    if "jac5" in frames and solved["jac5"].notna().any():
        row.update(jac5=solved["jac5"].mean(), jac10=solved["jac10"].mean(),
                   jac20=solved["jac20"].mean(), fs=final_score(frames))
    row.update(median_kp=frames["n_kp"].median(), median_lines=frames["n_ln"].median())
    return row


# ---------------------------------------------------------------- Images


def percent(value: float) -> str:
    if pd.isna(value):
        return ""
    return "no camera" if np.isinf(value) else f"{value * 100:.1f}"


def draw_table(table: pd.DataFrame, title: str, name: str, note: str = "") -> None:
    """Render a table as an image, with the first column left-aligned."""
    table = table.astype(object).where(table.notna(), "").astype(str)
    widths = [max(table[c].str.len().max(), round(len(str(c)) * 1.25)) + 3 for c in table.columns]
    widths[0] += 3  # Left-aligned cells carry extra padding that would clip a long first column.
    fig, ax = plt.subplots(figsize=(max(sum(widths) * 0.085, 5.0), 0.27 * (len(table) + 1) + 0.1),
                           facecolor=SURFACE)
    ax.axis("off")
    cells = ax.table(cellText=table.values, colLabels=table.columns,
                     colWidths=[w / sum(widths) for w in widths], cellLoc="center",
                     bbox=[0, 0, 1, 1])
    cells.auto_set_font_size(False)
    cells.set_fontsize(9)
    for (r, c), cell in cells.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK, ha="left" if c == 0 else "center",
                            weight="bold" if r == 0 else "normal")
        cell.set_facecolor(HEADER if r == 0 else SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    if note:
        ax.text(0, -0.015, note, transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    (OUT / "tables").mkdir(exist_ok=True)
    fig.savefig(OUT / "tables" / f"{name}.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def save_figure(fig, name: str) -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_bars(values: dict, datasets: tuple, labels: dict, top: float, split: int, ylabel: str,
              title: str, name: str) -> None:
    """Grouped bars per dataset and variant; no bar or text where a value is not scored yet."""
    fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
    width = 0.2
    for i, model in enumerate(Config.CALIB_MODELS):
        pos = np.arange(len(datasets)) + (i - 1.5) * width
        heights = [v if np.isfinite(v) else 0.0 for v in values[model]]
        ax.bar(pos, heights, width * 0.92, color=COLORS[model], label=labels[model])
        for x, value in zip(pos, values[model]):
            if np.isnan(value):
                continue
            if np.isinf(value):
                ax.text(x, top * 0.01, "no camera", ha="center", va="bottom", rotation=90,
                        fontsize=7, color=INK)
                continue
            ax.text(x, min(value, top * 0.94) + top * 0.01, f"{value:.1f}", ha="center",
                    fontsize=7, color=INK)
    ax.set_xticks(range(len(datasets)), datasets, fontsize=9)
    ax.axvline(split - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    style(ax)
    save_figure(fig, name)


def plot_focal_bars(data: dict, labels: dict) -> None:
    """Median focal error per dataset; a failed median is shown as text, not a bar."""
    values = {m: [np.nan if data[m][d] is None else focal_errors(data[m][d]).median() * 100
                  for d in FOCAL_DATASETS] for m in Config.CALIB_MODELS}
    finite = [v for row in values.values() for v in row if np.isfinite(v)]
    plot_bars(values, FOCAL_DATASETS, labels, max(finite) * 1.2 if finite else 1.0,
              len(INDEPENDENT), "median focal length error (%)",
              "Focal length error: independent references (left), dataset camera (right)",
              "focal_error")


def plot_focal_cdf(data: dict, labels: dict) -> None:
    """Share of images within a focal error, per dataset; no camera never reaches the curve."""
    fig, axes = plt.subplots(1, len(FOCAL_DATASETS), figsize=(14, 3.8), sharey=True,
                             facecolor=SURFACE)
    grid = np.logspace(-3, 0.3, 200)
    for ax, dataset in zip(axes, FOCAL_DATASETS):
        for model in Config.CALIB_MODELS:
            if data[model][dataset] is None:
                continue
            errors = focal_errors(data[model][dataset]).to_numpy()
            if len(errors):
                ax.plot(grid * 100, [(errors <= g).mean() * 100 for g in grid],
                        color=COLORS[model], lw=2, label=labels[model])
        ax.set_xscale("log")
        ax.set_title(dataset, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("focal length error (%)", fontsize=9, color=INK)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_ylabel("% of images, no camera = never", fontsize=9, color=INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, names, frameon=False, fontsize=9, ncols=4, loc="lower center")
    fig.savefig(OUT / "figures" / "focal_error_cdf.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_final_score(data: dict, labels: dict) -> None:
    """Final Score: annotated lines, then agreement with the dataset camera."""
    datasets = LINE_ANNOTATED + DERIVED
    values = {m: [np.nan if data[m][d] is None else final_score(data[m][d]) * 100
                  for d in datasets] for m in Config.CALIB_MODELS}
    plot_bars(values, datasets, labels, 105, len(LINE_ANNOTATED),
              "Final Score (CR x JaC at 5 px)",
              "Line fit: hand-annotated lines (left), lines from the dataset camera (right)",
              "final_score")


def plot_tuning(tuning: pd.DataFrame) -> None:
    """Final Score on SN23-valid over the threshold grid, one panel per variant."""
    fig, axes = plt.subplots(1, len(Config.CALIB_MODELS), figsize=(14, 3.6), facecolor=SURFACE)
    for ax, model in zip(axes, Config.CALIB_MODELS):
        table = tuning.loc[tuning["model"] == model].pivot(index="kp_th", columns="line_th",
                                                           values="fs") * 100
        ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(table.shape[1]), table.columns, fontsize=7)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=7)
        best = np.unravel_index(np.nanargmax(table.to_numpy()), table.shape)
        for (r, c), value in np.ndenumerate(table.to_numpy()):
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 55 else INK,
                    weight="bold" if (r, c) == best else "normal")
        ax.set_title(NAMES[model], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("line threshold", fontsize=8, color=INK)
    axes[0].set_ylabel("keypoint threshold", fontsize=8, color=INK)
    save_figure(fig, "threshold_tuning")


# ---------------------------------------------------------------- Report


def comparison_rows(pnl: str, soccer: str, data: dict) -> list[dict]:
    """PnLCalib against one SoccerMaster variant, blank where either is not scored yet."""
    rows = []
    pairs = [("median focal error", d) for d in FOCAL_DATASETS]
    pairs += [("FS", d) for d in LINE_ANNOTATED + DERIVED]
    for metric, dataset in pairs:
        row = {"pnlcalib_variant": NAMES[pnl], "soccermaster_variant": NAMES[soccer],
               "dataset": dataset, "metric": metric}
        a, b = data[pnl][dataset], data[soccer][dataset]
        if a is not None and b is not None:
            if metric == "FS":
                point, low, high = paired_fs_diff(a, b, dataset)
                winner = winner_of(low, high, lower_is_better=False)
                row.update({"diff": point, "ci_low": low, "ci_high": high, "winner": winner})
            else:
                row.update({k: v for k, v in compare_focal(a, b, dataset).items()
                            if k in ("diff", "ci_low", "ci_high", "winner")})
        rows.append(row)
    return rows


def write_results(decision: dict, pending: list[str]) -> None:
    """Short index of the report, stating what is still blank."""
    lines = ["# Calibration comparison", "", f"Verdict: **{decision['winner']}** "
             f"({decision['decided_by']}).", ""]
    if pending:
        lines += ["Not scored yet, blank in every table and figure:", ""]
        lines += [f"- {item}" for item in pending] + [""]
    lines += ["## Figures", ""]
    lines += [f"- `figures/{p.name}`" for p in sorted((OUT / "figures").glob("*.png"))]
    lines += ["", "## Tables", ""]
    lines += [f"- `tables/{p.name}`" for p in sorted((OUT / "tables").glob("*.png"))]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    chosen = json.loads((OUT / "chosen_thresholds.json").read_text())
    selected = {family: max(members, key=lambda m: chosen[m]["fs"])
                for family, members in FAMILIES.items()}
    labels = {m: NAMES[m] + (" (selected)" if m in selected.values() else "")
              for m in Config.CALIB_MODELS}
    data = {m: {d: load_frames(m, chosen[m]["tag"], d) for d in ALL_DATASETS}
            for m in Config.CALIB_MODELS}
    summary = pd.DataFrame([row_of(data[m][d], d, labels[m])
                            for m in Config.CALIB_MODELS for d in ALL_DATASETS])
    summary.to_csv(OUT / "summary.csv", index=False)

    # The verdict reads the selected pair; the other SoccerMaster variant shows if it depends on it.
    pnl = selected["pnlcalib"]
    rows = []
    for soccer in sorted(FAMILIES["soccermaster"], key=lambda m: m != selected["soccermaster"]):
        rows += comparison_rows(pnl, soccer, data)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    decision = decide(comparison.loc[
        comparison["soccermaster_variant"] == NAMES[selected["soccermaster"]]])
    decision.update(selected=selected, thresholds={m: chosen[m] for m in selected.values()})

    extra_path = OUT / "extra_runs.json"
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else EXTRA_RUNS
    ablations = summary.to_dict("records")
    for run in extra:
        frames = load_frames(run["model"], run["tag"], run["dataset"])
        ablations.append(row_of(frames, run["dataset"], run["label"]))
    ablations = pd.DataFrame(ablations)
    ablations.to_csv(OUT / "ablations.csv", index=False)

    speed = pd.DataFrame([json.loads((OUT / "speed" / f"{m}.json").read_text())
                          for m in Config.CALIB_MODELS if (OUT / "speed" / f"{m}.json").exists()])
    speed.drop(columns=["network_ms_repeats"]).to_csv(OUT / "speed.csv", index=False)

    plot_focal_bars(data, labels)
    plot_focal_cdf(data, labels)
    plot_final_score(data, labels)
    plot_tuning(pd.read_csv(OUT / "tuning.csv"))

    focal = summary.loc[summary["dataset"].isin(FOCAL_DATASETS)]
    draw_table(pd.DataFrame({
        "Dataset": focal["dataset"], "Model": focal["model"],
        "Images": focal["with_reference"].astype("Int64"),
        "Excluded": focal["excluded"].astype("Int64"),
        "Median error %": focal["focal_median"].map(percent),
        "Within 5 %": focal["within_5"].map(percent),
        "Within 10 %": focal["within_10"].map(percent),
        "Camera returned %": focal["cr"].map(percent),
        "f / reference": focal["focal_ratio_median"].round(3)}),
        "Focal length (K) error", "focal_error",
        "No camera counts as failure. Excluded: no usable reference.")
    lines = summary.loc[summary["dataset"].isin(LINE_ANNOTATED + DERIVED)]
    draw_table(pd.DataFrame({
        "Dataset": lines["dataset"], "Model": lines["model"],
        "Images": lines["images"].astype("Int64"),
        "JaC5": lines["jac5"].map(percent), "JaC10": lines["jac10"].map(percent),
        "JaC20": lines["jac20"].map(percent), "CR": lines["cr"].map(percent),
        "FS": lines["fs"].map(percent)}),
        "Line fit at 960x540", "line_fit",
        "ISSIA and LP use lines from the dataset camera, so they measure agreement with it.")
    shown = comparison.copy()
    for column in ("diff", "ci_low", "ci_high"):
        shown[column] = shown[column].map(percent)
    draw_table(shown.drop(columns="pnlcalib_variant").rename(columns={
        "soccermaster_variant": "Against", "dataset": "Dataset", "metric": "Metric",
        "diff": "Difference", "ci_low": "95 % low", "ci_high": "95 % high", "winner": "Winner"}),
        f"{NAMES[pnl]} minus SoccerMaster. Verdict: {decision['winner']}", "comparison",
        f"Decided by: {decision['decided_by']}. Differences in percentage points.")
    keys = ablations.loc[ablations["dataset"].isin(FOCAL_DATASETS)]
    draw_table(pd.DataFrame({
        "Dataset": keys["dataset"], "Variant": keys["model"],
        "Median error %": keys["focal_median"].map(percent),
        "Within 5 %": keys["within_5"].map(percent), "CR %": keys["cr"].map(percent),
        "FS": keys["fs"].map(percent), "Median keypoints": keys["median_kp"],
        "Median lines": keys["median_lines"]}),
        "Ablations: focal length error by variant", "ablations",
        "TSWC has no FS because it has no annotated lines.")
    if len(speed):
        draw_table(pd.DataFrame({
            "Model": speed["model"].map(NAMES), "Network ms": speed["network_ms"].round(1),
            "Solver ms": speed["solver_ms"].round(1), "Total ms": speed["total_ms"].round(1),
            "Images per s": speed["fps"].round(2), "Peak GPU MB": speed["peak_gpu_mb"].round(0)}),
            "Inference speed, one image at a time", "speed",
            "RTX 3060 Laptop, float32, batch 1, solver on one CPU thread.")
    floor = noise_floor()
    decision["reference_noise_floor"] = floor
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1))
    draw_table(oracle_rows(data).astype({"Images": "Int64"}),
               "Focal length error against the annotated-line camera", "focal_error_oracle",
               f"Supporting evidence, not in the decision rule. The annotated references differ "
               f"by {floor['median_pct']:.1f} % (median) on {floor['images']} WC14 images.")
    repro = pd.read_csv(OUT / "reproduction.csv")
    draw_table(pd.DataFrame({
        "Dataset": repro["dataset"], "Weights": repro["weights"], "Metric": repro["metric"],
        "Published": repro["published"], "Ours": repro["ours"].round(1),
        "Gap": repro["gap"].round(1), "Pass": repro["is_pass"]}),
        "Reproduction of PnLCalib's published numbers", "reproduction",
        "WC14 FS reaches 85.2 with upstream's GPU image resize (separate test); this run uses CPU.")

    pending = [f"{row['model']} on {row['dataset']}" for row in ablations.to_dict("records")
               if pd.isna(row.get("images"))]
    write_results(decision, pending)

    identity = summary.loc[summary["dataset"].isin(PAPER_ZERO_SHOT)
                           & summary["model"].str.startswith("SoccerMaster")]
    print(summary.round(3).T.to_string())
    print(comparison.round(4).to_string())
    print(json.dumps(decision, indent=1))
    print(identity[["model", "dataset", "fs"]].round(3).to_string(), PAPER_ZERO_SHOT,
          PAPER_FINE_TUNED)
    print(f"pending: {len(pending)} runs")


if __name__ == "__main__":
    main()
