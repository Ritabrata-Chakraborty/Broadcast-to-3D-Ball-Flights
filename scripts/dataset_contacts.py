"""Grad et al.'s own contact points, from their released preprocessing (dev/ folder).

At each contact frame: the ball's ground point, or the player's foot point at a high contact
(their player-overlap rule). Used as the contact points of their optimiser with GT inputs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config

VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])


def rule_flags() -> tuple[dict, dict]:
    """(view, half) -> the dataset's high-contact rule and its pitch point (m) per frame."""
    out, points = {}, {}
    for view in Config.VIEWS:
        for half, name in enumerate(Config.HALVES):
            path = Config.DATA_ROOT / Config.LP / VIEW_NAMES[view] / name / "dev" / \
                "df_merged_ball_player.csv"
            table = pd.read_csv(path, usecols=["file_name", "close_contact", "high_contact",
                                               "x_pitch2D", "y_pitch2D"], low_memory=False)
            flag = table["close_contact"].astype(bool) & table["high_contact"].astype(bool)
            out[(view, half)] = flag.groupby(table["file_name"]).any()
            first = table.drop_duplicates("file_name").set_index("file_name")
            points[(view, half)] = first[["x_pitch2D", "y_pitch2D"]] / Config.DM_PER_M
    return out, points


def ball3d_anchors(arrays: dict, points: dict) -> np.ndarray:
    """Grad et al.'s own pitch point at each contact frame (nearest row inside the flight)."""
    pitch = np.full_like(arrays["pitch_xy"], np.nan)
    for i in range(len(arrays["n"])):
        n, a = int(arrays["n"][i]), int(arrays["a"][i])
        table = points[(int(arrays["view"][i]), int(arrays["half"][i]))]
        for frame, step in ((0, 1), (n - 1, -1)):
            for k in range(Config.ANCHOR_SEARCH_FRAMES + 1):
                f = a + frame + step * k
                if f in table.index and np.isfinite(table.loc[f].to_numpy(float)).all():
                    pitch[i, frame] = table.loc[f].to_numpy(float)
                    break
    return pitch
