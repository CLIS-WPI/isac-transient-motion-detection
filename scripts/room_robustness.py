"""Multipath room robustness check (pre-registered in SCIENTIFIC_DECISIONS.md, 2026-10-01).

Registered uniform / burst / DE-CuSum and the selected trigger configs, NO retuning, on the
Sionna RT room split (seeds 80000+). Threshold-only recalibration on room_val (81000+) only if
a registered policy exceeds 1.25 FA/min (secondary). Never loads the test split (40000+).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/room_robustness.py --n 300 --workers 16
"""
import argparse
import dataclasses
import json
import os
import pickle
import time

import numpy as np

from mdsense import Config
from mdsense.pipeline import (make_split, Split, factory, fit_llr, run_split, calibrate_threshold,
                              limit_blas_threads, cache_fingerprint)
from mdsense.evaluate import aggregate, by_seed, paired_bootstrap, paired_event_fails

ROOM = {"scene": "room_8x10x3", "max_depth": 3, "background": True}
SNAPSHOT = 8
BUDGETS = (0.025, 0.05, 0.10)
SEED_ROOM, SEED_ROOM_VAL, N_ROOM_VAL = 80_000, 81_000, 100
FA_FLAG = 1.25


def summary(outs):
    a = aggregate(outs)
    n_fail = sum(o["fail"] for o in outs if o["has_event"])
    return {"p_fail": a["p_fail"], "n_event": a["n_event"], "events_in_time": a["n_event"] - n_fail,
            "fa_per_min": a["fa_per_min"], "cost_monitor": a["cost_monitor"],
            "cost_total": a["cost_total"], "median_delay": a["median_delay"]}


def compare(outs, P):
    """DE vs burst as in the paper (events D <= 0.6 s); trigger - DE over all events."""
    out = {}
    if "burst" in outs and "decusum" in outs:
        fb, fd = paired_event_fails(outs["burst"], outs["decusum"], P.go_max_event_duration)
        gain, lo, hi = paired_bootstrap(fb, fd)
        out["burst_minus_decusum"] = {"gain": gain, "ci95": [lo, hi], "n_paired": len(fb)}
    if "trigger" in outs and "decusum" in outs:
        t, de = by_seed(outs["trigger"]), by_seed(outs["decusum"])
        seeds = [s for s in sorted(set(t) & set(de)) if t[s]["has_event"]]
        diff, lo, hi = paired_bootstrap([t[s]["fail"] for s in seeds], [de[s]["fail"] for s in seeds])
        out["trigger_minus_decusum"] = {"p_fail_diff": diff, "ci95": [lo, hi], "n_paired": len(seeds)}
    return out


def evaluate(split, policies, cfg, thresholds, log, tag):
    res = {}
    for b, pols in policies.items():
        outs = {k: run_split(split, mk, cfg, llr=llr, threshold=thresholds[b][k])
                for k, (mk, llr) in pols.items()}
        row = {k: summary(o) for k, o in outs.items()}
        row.update(compare(outs, cfg.pilot))
        res[str(b)] = row
        for k, s in row.items():
            if k == "burst_minus_decusum":
                log(f"[{tag}] b={b} burst-DE gain={s['gain']:+.3f} CI=[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] n={s['n_paired']}")
            elif k == "trigger_minus_decusum":
                log(f"[{tag}] b={b} trigger-DE diff={s['p_fail_diff']:+.3f} CI=[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] n={s['n_paired']}")
            else:
                log(f"[{tag}] b={b} {k:8s} P_fail={s['p_fail']:.3f} in_time={s['events_in_time']}/{s['n_event']} "
                    f"FA/min={s['fa_per_min']:.2f} cost_mon={s['cost_monitor'] / 32:.1f} "
                    f"cost_tot={s['cost_total'] / 32:.1f} probes/s delay_med={s['median_delay'] * 1e3:.0f}ms")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--n", type=int, required=True, help="room episodes (300, or 200 per the pre-registered budget rule)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.environ["MDSENSE_WORKERS"] = str(max(1, args.workers))
    limit_blas_threads(1)
    t_start = time.time()

    cfg = Config()
    with open("results/preregistration.json") as f:
        prereg = json.load(f)
    norm = lambda x: json.loads(json.dumps(x, default=float))
    if norm(dataclasses.asdict(cfg)) != norm(prereg["config"]):
        raise SystemExit("Config differs from the pre-registered one.")
    with open("results/selected_models.pkl", "rb") as f:
        registered = pickle.load(f)
    with open(os.path.join(args.out, "trigger.json")) as f:
        trig = json.load(f)["selected"]
    P = cfg.pilot

    def log(m):
        print(m, flush=True)

    # policies: registered (params, A, LLR) + selected trigger (LLR = train-fitted burst LLR of its look)
    train = Split(make_split(cfg, "train", P.n_train, P.seed_train, args.root)).preload(cfg)
    policies, A0 = {}, {}
    for b in BUDGETS:
        policies[b], A0[b] = {}, {}
        for k in ("uniform", "burst", "decusum"):
            v = registered[b][k]
            policies[b][k] = (factory(v["kind"], cfg, b, **v["params"]), v["llr"])
            A0[b][k] = v["A"]
        r = trig.get(str(b))
        if r is not None:
            llr = fit_llr(train, factory("burst", cfg, b, burst_len=r["burst_len"], spacing=r["spacing"]), cfg)
            policies[b]["trigger"] = (factory("trigger", cfg, b, burst_len=r["burst_len"], spacing=r["spacing"],
                                              p_slow=r["p_slow"], tau=r["tau"], n_hold=r["n_hold"]), llr)
            A0[b]["trigger"] = r["A"]

    fp = cache_fingerprint(cfg, "sionna", SNAPSHOT, None, ROOM)
    log(f"[room] {ROOM} snapshot_slots={SNAPSHOT} fingerprint={fp} n={args.n} seeds {SEED_ROOM}+")
    t0 = time.time()
    room = Split(make_split(cfg, "room", args.n, SEED_ROOM, args.root, backend="sionna",
                            snapshot_slots=SNAPSHOT, room=ROOM)).preload(cfg)
    assert room.seeds == list(range(SEED_ROOM, SEED_ROOM + args.n))
    t_room = time.time() - t0
    res = {"note": "robustness check, no retuning; SCIENTIFIC_DECISIONS.md 2026-10-01; "
                   "test split (seeds 40000+) never loaded",
           "room": ROOM, "snapshot_slots": SNAPSHOT, "fingerprint": fp, "n_room": args.n,
           "seeds_room": [SEED_ROOM, SEED_ROOM + args.n - 1], "primary": {}, "secondary": None}
    res["primary"] = evaluate(room, policies, cfg, A0, log, "room")

    g = {b: res["primary"][str(b)] for b in (0.025, 0.05)}
    holds = all(r["burst_minus_decusum"]["gain"] >= P.go_min_abs_gain and r["burst_minus_decusum"]["ci95"][0] > 0
                and r["decusum"]["fa_per_min"] <= FA_FLAG for r in g.values())
    res["main_result_holds"] = bool(holds)
    log(f"[holds] main result holds in the room: {holds}  (gain >= {P.go_min_abs_gain}, CI_low > 0, "
        f"DE FA <= {FA_FLAG} at 0.025 and 0.05)")

    flagged = [(b, k) for b in BUDGETS for k in ("uniform", "burst", "decusum")
               if res["primary"][str(b)][k]["fa_per_min"] > FA_FLAG]
    res["fa_flagged"] = [[b, k] for b, k in flagged]
    if flagged:
        log(f"[secondary] FA > {FA_FLAG}/min for {flagged}: threshold-only recalibration on room_val")
        t1 = time.time()
        rv = Split(make_split(cfg, "room_val", N_ROOM_VAL, SEED_ROOM_VAL, args.root, backend="sionna",
                              snapshot_slots=SNAPSHOT, room=ROOM)).preload(cfg)
        assert min(rv.seeds) >= SEED_ROOM_VAL
        A1, recal = {}, {}
        for b, pols in policies.items():
            A1[b], recal[str(b)] = {}, {}
            for k, (mk, llr) in pols.items():
                A, agg, ok = calibrate_threshold(rv, mk, llr, cfg, P.fa_max_per_min)
                A1[b][k] = A
                recal[str(b)][k] = {"A_registered": A0[b][k], "A_room_val": float(A), "fa_ok": bool(ok),
                                    "room_val_fa_per_min": agg["fa_per_min"]}
                log(f"[recal] b={b} {k:8s} A {A0[b][k]:.2f} -> {A:.2f}{'' if ok else '  [FA limit not met]'}")
        res["secondary"] = {"label": "SECONDARY: threshold-only recalibration on room_val; costs not re-matched",
                            "seeds_room_val": [SEED_ROOM_VAL, SEED_ROOM_VAL + N_ROOM_VAL - 1],
                            "thresholds": recal, "room_val_gen_s": time.time() - t1,
                            "results": evaluate(room, policies, cfg, A1, log, "room-secondary")}

    res["timing_s"] = {"room_split": t_room, "total": time.time() - t_start}
    path = os.path.join(args.out, "room.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    log(f"[done] {path}  total {res['timing_s']['total'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
