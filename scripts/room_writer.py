"""Parallel generator for the Sionna RT room episodes (room / room_val splits).

One Sionna process uses only ~20 % of an H100, so several writers share one GPU. Writer j of K
computes seeds start+j, start+j+K, ... in order, with exactly the call and atomic save that
pipeline.make_split uses; episodes are deterministic per seed, so the result is identical to a
serial run. Existing episodes are skipped (safe to restart). Then run room_robustness.py, which
finds all episodes on disk and only evaluates.

    for j in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
        python scripts/room_writer.py room 80000 80200 $j 6 & done; wait      # ~6 h on one H100
    for j in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
        python scripts/room_writer.py room_val 81000 81100 $j 6 & done; wait  # ~3 h
"""
import argparse
import os
import time

import numpy as np

from mdsense import Config
from mdsense.body import sample_scenario
from mdsense.channel import dense_channel
from mdsense.pipeline import cache_root

ROOM = {"scene": "room_8x10x3", "max_depth": 3, "background": True}   # as in room_robustness.py
SNAPSHOT = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["room", "room_val"])
    ap.add_argument("start", type=int)
    ap.add_argument("stop", type=int, help="exclusive")
    ap.add_argument("j", type=int, help="writer index, 0 <= j < K")
    ap.add_argument("K", type=int, help="number of writers")
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    cfg = Config()
    d = os.path.join(cache_root(a.root, cfg, "sionna", SNAPSHOT, None, ROOM), a.split)
    os.makedirs(d, exist_ok=True)
    for seed in range(a.start + a.j, a.stop, a.K):
        path = os.path.join(d, f"ep{seed}.npy")
        if os.path.exists(path):
            continue
        t0 = time.time()
        scn = sample_scenario(seed, cfg.scenario, cfg.radio)
        H = dense_channel(scn, cfg.radio, cfg.scenario.duration, backend="sionna",
                          snapshot_slots=SNAPSHOT, **ROOM)
        np.save(path + f".tmp{a.j}.npy", H)
        os.replace(path + f".tmp{a.j}.npy", path)
        print(f"[w{a.j}] ep{seed} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
