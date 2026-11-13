"""Assemble the fine-tuned-head extractions per training half and the fold -> extraction map.

heads_half_<k> holds the broadcast extraction made with the heads trained on half k
(heads_broadcast_<a|b>) and the static extraction made with the static heads trained on half k
(heads_static_half_<k>), as symlinks. Each test block uses the heads trained on the other half:
folds 1-2 (test blocks in half 1) use heads_half_2, folds 3-4 use heads_half_1.
Output: outputs/extract/heads_half_{1,2}/ and outputs/extract/heads_map.json.
Run in the ball3d env: scripts/py scripts/link_heads.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config

BROADCAST = {"half_1": "heads_broadcast_a", "half_2": "heads_broadcast_b"}
FOLD_HEADS = {"1": "half_2", "2": "half_2", "3": "half_1", "4": "half_1"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Config.EXTRACT_DIR
    for half in Config.HALVES:
        out = root / f"heads_{half}"
        out.mkdir(parents=True, exist_ok=True)
        for source in (root / BROADCAST[half], root / f"heads_static_{half}"):
            for path in sorted(source.iterdir()):
                is_unit = path.name.startswith(("broadcast_", "camera"))
                if is_unit and source.name.startswith("heads_broadcast") == \
                        path.name.startswith("broadcast_"):
                    link = out / path.name
                    if not link.exists():
                        link.symlink_to(path.resolve())
    heads_map = {fold: str(root / f"heads_{half}") for fold, half in FOLD_HEADS.items()}
    (root / "heads_map.json").write_text(json.dumps(heads_map, indent=1))
    print(json.dumps(heads_map, indent=1))


if __name__ == "__main__":
    main()
