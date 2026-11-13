"""Freeze the LP airborne flights and splits, with oracle fits, into outputs/flights/.

Flights are selected once on LP broadcast and mapped to the static views, so one physical flight
has one split in every view. Run in the ball3d env: scripts/py scripts/build_flights.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.oracle import fit_oracle

CAM_COLS = ["rot_x", "rot_y", "rot_z", "tx", "ty", "tz", "fx", "fy", "princ_x", "princ_y"]
REF_W_PX, REF_H_PX = 640.0, 360.0
VIEWS = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])

# ---------------------------------------------------------------- Loading


def load_unit(view: str, half: str) -> dict:
    """All per-frame tables of one LP view and half; ball pixels match ball3d's shipped runs."""
    base = Config.DATA_ROOT / Config.LP / view / half
    meta = json.loads((base / "sequence_metadata.json").read_text())
    gt = pd.read_csv(base / "track/ball_3d-gt.csv").drop_duplicates("file_name")
    pivots = pd.read_csv(base / "track/ball_pivot_point-gt.csv")
    merged = pd.read_csv(base / "dev/df_merged_ball_player.csv",
                         usecols=["file_name", "xk", "yk_original", "close_contact",
                                  "high_contact"])
    tracked = pd.read_csv(base / "track/ball_3d.basic_fitg.csv",
                          usecols=["file_name", "xk", "yk", "x_pitch2D", "y_pitch2D"],
                          low_memory=False)
    return {"view": view, "half": half, "width": float(meta["width"]),
            "height": float(meta["height"]),
            "gt": gt.set_index("file_name")[["x", "y", "z"]] / Config.DM_PER_M,
            "pivots": np.sort(pivots.loc[pivots["pivot_point"] == 1, "file_name"].unique()),
            "pauses": pd.read_csv(base / "dev/pauses.csv"),
            "cams": pd.read_csv(base / "camera_smooth.csv").drop_duplicates("frame_index")
            .set_index("frame_index")[CAM_COLS],
            "merged": merged.drop_duplicates("file_name").set_index("file_name"),
            "tracked": tracked.drop_duplicates("file_name").set_index("file_name")}


def has_flag(unit: dict, frame: int, column: str) -> bool:
    """True if a contact flag fires within two frames of a pivot."""
    window = unit["merged"][column].reindex(np.arange(frame - 2, frame + 3))
    return bool(window.fillna(0).astype(float).to_numpy().any())


# ---------------------------------------------------------------- Selection


def select_broadcast(half: str) -> pd.DataFrame:
    """The frozen flight rule applied to broadcast contact-to-contact segments."""
    unit = load_unit("broadcast", half)
    pivots = unit["pivots"]
    size = int(max(pivots.max(), unit["gt"].index.max())) + 2
    paused = np.zeros(size, bool)
    for start, end in zip(unit["pauses"]["start_pause"], unit["pauses"]["end_pause"]):
        paused[int(start):int(end)] = True

    rows = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        a, b = int(a), int(b)
        n = b - a + 1
        if paused[a:b + 1].any() or not Config.MIN_FRAMES <= n <= Config.MAX_FRAMES:
            continue
        xyz = unit["gt"].reindex(np.arange(a, b + 1)).to_numpy(float)
        valid = ~np.isnan(xyz[:, 2])
        if valid.mean() < Config.MIN_COVERAGE:
            continue
        apex = float(np.nanmax(xyz[:, 2]))
        dist = float(np.nansum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)))
        speed = dist * Config.FPS / (n - 1)
        is_contact = has_flag(unit, a, "close_contact") or has_flag(unit, a, "high_contact")
        if apex < Config.MIN_APEX_M or speed > Config.MAX_SPEED_MPS or not is_contact:
            continue
        rows.append({"half": half, "broadcast_a": a, "broadcast_b": b, "n": n, "apex_m": apex,
                     "speed_mps": speed, "z0_m": float(xyz[valid, 2][:2].mean()),
                     "start_high": has_flag(unit, a, "high_contact"),
                     "end_high": has_flag(unit, b, "high_contact")})
    return pd.DataFrame(rows)


def fold_splits(flights: pd.DataFrame, train_half: str) -> pd.Series:
    """Other half is test. Train half is train then validation by time, with a gap."""
    in_half = flights["half"] == train_half
    cut = float(flights.loc[in_half, "broadcast_a"].quantile(1.0 - Config.VAL_SHARE))
    split = pd.Series("test", index=flights.index)
    split[in_half] = "drop"
    split[in_half & (flights["broadcast_b"] < cut - Config.SPLIT_GAP_FRAMES)] = "train"
    split[in_half & (flights["broadcast_a"] >= cut)] = "val"
    return split


def assign_splits(flights: pd.DataFrame) -> pd.DataFrame:
    """Cross-fit over halves, so every flight has one model that never saw it."""
    for fold, half in Config.HALF_FOLDS.items():
        flights[f"split_{fold}"] = fold_splits(flights, half)
    flights["group"] = np.arange(len(flights))
    flights["start_class"] = np.digitize(flights["z0_m"], Config.START_BINS_M)
    return flights


# ---------------------------------------------------------------- Per-view arrays


def view_rows(unit: dict, flights: pd.DataFrame) -> list[dict]:
    """Match the frozen flights to one view's pivots and collect their fixed-length arrays."""
    offset = 0
    if unit["view"] != "broadcast":
        offset = Config.LP_VIEW_OFFSETS[unit["half"]][unit["view"]]
    pivots = unit["pivots"]
    scale = Config.IMAGE_WIDTH_PX / unit["width"]
    height = unit["height"] * scale
    out = []
    for _, flight in flights[flights["half"] == unit["half"]].iterrows():
        a = int(flight["broadcast_a"]) - offset
        b = int(flight["broadcast_b"]) - offset
        if unit["view"] != "broadcast":
            near_a = pivots[np.argmin(np.abs(pivots - a))]
            near_b = pivots[np.argmin(np.abs(pivots - b))]
            tol = Config.VIEW_MATCH_FRAMES
            if abs(near_a - a) > tol or abs(near_b - b) > tol or near_b <= near_a:
                continue
            a, b = int(near_a), int(near_b)
        n = b - a + 1
        if not Config.MIN_FRAMES <= n <= Config.MAX_FRAMES:
            continue

        track = np.arange(a, a + Config.MAX_FRAMES)
        cams = unit["cams"].reindex(track).ffill().bfill().to_numpy(float)
        if np.isnan(cams).any():
            continue
        rot = Rotation.from_rotvec(cams[:, :3]).as_matrix()
        trans = cams[:, 3:6] / Config.DM_PER_M
        intr = np.c_[cams[:, 6] * unit["width"] / REF_W_PX * scale,
                     cams[:, 7] * unit["height"] / REF_H_PX * scale,
                     cams[:, 8] * unit["width"] / REF_W_PX * scale,
                     cams[:, 9] * unit["height"] / REF_H_PX * scale]
        xyz = unit["gt"].reindex(track).to_numpy(float)
        xyz[n:] = np.nan
        if (~np.isnan(xyz[:n, 2])).mean() < Config.MIN_COVERAGE:
            continue

        flipped = xyz * np.array([1.0, 1.0, -1.0])
        x_cam = np.einsum("nij,nj->ni", rot, flipped) + trans
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = np.c_[intr[:, 0] * x_cam[:, 0] / x_cam[:, 2] + intr[:, 2],
                       intr[:, 1] * x_cam[:, 1] / x_cam[:, 2] + intr[:, 3]]
            visible = (x_cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Config.IMAGE_WIDTH_PX)
            visible &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if visible[:n].mean() < Config.MIN_VISIBLE:
            continue

        box = unit["tracked"].reindex(track)
        # ball3d fits (xk, yk), the bottom of the ball; yk_original is its centre.
        det = np.c_[box["xk"].to_numpy(float), box["yk"].to_numpy(float)] * scale
        det[n:] = np.nan
        pitch = box[["x_pitch2D", "y_pitch2D"]].to_numpy(float) / Config.DM_PER_M
        pitch[n:] = np.nan
        out.append({"view": unit["view"], "group": int(flight["group"]),
                    "half": unit["half"], "split_a": flight["split_a"],
                    "split_b": flight["split_b"], "a": a, "n": n,
                    "start_high": bool(flight["start_high"]),
                    "end_high": bool(flight["end_high"]), "image_h": height, "xyz": xyz,
                    "rot": rot, "trans": trans, "intr": intr, "uv_gt": uv, "uv_det": det,
                    "pitch_xy": pitch,
                    "visible": visible, "start_class": int(flight["start_class"])})
    return out


def oracle_job(row: dict) -> dict:
    """Oracle 3D fit of one flight in one view."""
    return fit_oracle(row["xyz"][:row["n"]], Config.FPS, Config.G_MPS2,
                      Config.SUBSTEPS, Config.K3_MAX, Config.KL_KS_MAX)


def save_split(rows: list[dict], fold: str, split: str, out: Path) -> None:
    """One archive per fold and split, so training cannot read its test flights by accident."""
    chosen = [r for r in rows if r[f"split_{fold}"] == split]
    (out / f"fold_{fold}").mkdir(exist_ok=True)
    np.savez_compressed(
        out / f"fold_{fold}" / f"{split}.npz",
        half=np.array([Config.HALVES.index(r["half"]) for r in chosen], np.int8),
        view=np.array([VIEWS.index(r["view"]) for r in chosen], np.int8),
        group=np.array([r["group"] for r in chosen], np.int32),
        a=np.array([r["a"] for r in chosen], np.int32),
        n=np.array([r["n"] for r in chosen], np.int32),
        start_high=np.array([r["start_high"] for r in chosen]),
        end_high=np.array([r["end_high"] for r in chosen]),
        start_class=np.array([r["start_class"] for r in chosen], np.int8),
        image_h=np.array([r["image_h"] for r in chosen], np.float32),
        **{key: np.stack([r[key] for r in chosen]).astype(np.float32)
           for key in ("xyz", "rot", "trans", "intr", "uv_gt", "uv_det", "pitch_xy")},
        visible=np.stack([r["visible"] for r in chosen]),
        oracle_params=np.stack([r["oracle"]["params"] for r in chosen]).astype(np.float32),
        oracle_rmse=np.array([r["oracle"]["rmse_spin_m"] for r in chosen], np.float32),
        drag_rmse=np.array([r["oracle"]["rmse_drag_m"] for r in chosen], np.float32))


# ---------------------------------------------------------------- Offset diagnosis


def diagnose_offsets(rows: list[dict]) -> pd.DataFrame:
    """Split detection minus projected GT into steady per-flight bias and frame-level jitter."""
    table = []
    for row in rows:
        if row["split_a"] != "train":
            continue
        n = row["n"]
        diff = row["uv_det"][:n] - row["uv_gt"][:n]
        ok = np.isfinite(diff).all(1) & row["visible"][:n]
        if ok.sum() < Config.MIN_FRAMES:
            continue
        err = np.linalg.norm(diff[ok], axis=1)
        inlier = err <= Config.OUTLIER_PX
        if inlier.sum() < Config.MIN_FRAMES:
            table.append({"view": row["view"], "outlier_share": 1.0 - inlier.mean()})
            continue
        clean = diff[ok][inlier]
        table.append({"view": row["view"], "error_px": float(np.median(err[inlier])),
                      "bias_px": float(np.linalg.norm(np.median(clean, axis=0))),
                      "bias_u_px": float(np.median(clean[:, 0])),
                      "bias_v_px": float(np.median(clean[:, 1])),
                      "jitter_px": float(np.median(np.linalg.norm(
                          clean - np.median(clean, axis=0), axis=1))),
                      "outlier_share": float(1.0 - inlier.mean())})
    return pd.DataFrame(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Config.FLIGHTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    flights = assign_splits(pd.concat([select_broadcast(h) for h in Config.HALVES],
                                      ignore_index=True))
    flights.to_csv(out / "flights.csv", index=False)
    for fold in Config.HALF_FOLDS:
        print(f"fold {fold} flights:", flights.groupby(f"split_{fold}").size().to_dict())

    rows = []
    for view in VIEWS:
        for half in Config.HALVES:
            rows += view_rows(load_unit(view, half), flights)
    counts = pd.Series([r["view"] for r in rows]).value_counts().sort_index()
    print("view flights:", counts.to_dict())

    fits = Parallel(n_jobs=args.workers, verbose=1)(delayed(oracle_job)(r) for r in rows)
    for row, fit in zip(rows, fits):
        row["oracle"] = fit
    for fold in Config.HALF_FOLDS:
        for split in ("train", "val", "test"):
            save_split(rows, fold, split, out)

    table = pd.DataFrame([{"view": r["view"], "half": r["half"], "group": r["group"],
                           "n": r["n"], "rmse_spin_m": r["oracle"]["rmse_spin_m"],
                           "rmse_drag_m": r["oracle"]["rmse_drag_m"]} for r in rows])
    table.to_csv(out / "view_flights.csv", index=False)
    print("oracle RMSE, 3D Euclidean, by view (m):")
    print(table.groupby("view")[["rmse_drag_m", "rmse_spin_m"]].median().round(3).to_string())

    offsets = diagnose_offsets(rows)
    offsets.to_csv(out / "offsets_train.csv", index=False)
    print("detection vs projected GT, train flights, px at 1920 wide (medians):")
    print(offsets.groupby("view").median(numeric_only=True).round(2).to_string())


if __name__ == "__main__":
    main()
