"""Post-hoc two-rate trigger baseline (pre-registered in SCIENTIFIC_DECISIONS.md, 2026-10-01).

Tune TriggerPolicy on val (cost matched to the registered burst, FA <= 1/min), then run the
selected trigger and the REGISTERED burst / DE-CuSum on a fresh split (seeds 70000+).
Never loads the test split (seeds 40000+).

    PYTHONPATH=. python scripts/trigger_baseline.py --root data --out analysis --workers 16
"""
import argparse
import dataclasses
import json
import os
import pickle
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

from mdsense import Config
from mdsense.pipeline import (make_split, Split, factory, fit_llr, run_split,
                              calibrate_threshold, limit_blas_threads)
from mdsense.evaluate import aggregate, by_seed, paired_bootstrap

BUDGETS = (0.025, 0.05)
TAUS = (0.0, 2.0, 4.0)
N_HOLDS = (2, 4, 8)
SEED_FRESH2, N_FRESH2 = 70_000, 300
COST_TOL = 0.05


def calibrate_pslow(val, cfg, b, L, d, tau, nh, llr, A, target, iters=10, tol=0.03):
    """P_slow (slots) so realised val monitoring cost matches `target`; cost falls with P_slow."""
    span = L * d

    def cost(p):
        mk = factory("trigger", cfg, b, burst_len=L, spacing=d, p_slow=p, tau=tau, n_hold=nh)
        return aggregate(run_split(val, mk, cfg, llr=llr, threshold=A))["cost_monitor"]
    lo, hi = span, 20_000                    # back-to-back ... slower than one look per episode
    c_lo, c_hi = cost(lo), cost(hi)
    if c_hi > target * (1 + tol):
        return hi, c_hi, False               # fast mode alone exceeds the budget: infeasible
    if c_lo < target * (1 - tol):
        return lo, c_lo, True                # cannot spend the budget (fails the 5 % check later)
    a, z = np.log(lo), np.log(hi)
    p, c = hi, c_hi
    for _ in range(iters):
        p = int(round(np.exp(0.5 * (a + z))))
        c = cost(p)
        if abs(c - target) <= tol * target:
            break
        if c > target:
            a = np.log(p)
        else:
            z = np.log(p)
    return p, c, True


def tune(val, cfg, b, L, d, tau, nh, llr, target, fa_max):
    """Alternation as in run_pilot.py: A -> P_slow -> (A -> P_slow) x2 -> final A."""
    def mk(p):
        return factory("trigger", cfg, b, burst_len=L, spacing=d, p_slow=p, tau=tau, n_hold=nh)
    p = int(round(L / b))                     # burst-equivalent period as the starting point
    A, _, _ = calibrate_threshold(val, mk(p), llr, cfg, fa_max)
    p, c, feas = calibrate_pslow(val, cfg, b, L, d, tau, nh, llr, A, target)
    if not feas:
        return {"feasible": False, "p_slow": p, "cost_monitor": c}
    for _ in range(2):
        A, _, _ = calibrate_threshold(val, mk(p), llr, cfg, fa_max)
        p, c, feas = calibrate_pslow(val, cfg, b, L, d, tau, nh, llr, A, target)
    A, agg, ok = calibrate_threshold(val, mk(p), llr, cfg, fa_max)
    cost_ok = abs(agg["cost_monitor"] / target - 1) <= COST_TOL
    return {"feasible": feas, "p_slow": int(p), "A": float(A), "fa_ok": bool(ok),
            "cost_ok": bool(cost_ok), "eligible": bool(feas and ok and cost_ok), **agg}


def summary(outs):
    a = aggregate(outs)
    n_fail = sum(o["fail"] for o in outs if o["has_event"])
    return {"p_fail": a["p_fail"], "n_event": a["n_event"], "events_in_time": a["n_event"] - n_fail,
            "fa_per_min": a["fa_per_min"], "cost_monitor": a["cost_monitor"],
            "cost_total": a["cost_total"], "median_delay": a["median_delay"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--prereg", default="results/preregistration.json")
    ap.add_argument("--models", default="results/selected_models.pkl")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.environ["MDSENSE_WORKERS"] = str(max(1, args.workers))
    limit_blas_threads(1)
    t_start = time.time()

    cfg = Config()
    with open(args.prereg) as f:
        prereg = json.load(f)
    norm = lambda x: json.loads(json.dumps(x, default=float))
    if norm(dataclasses.asdict(cfg)) != norm(prereg["config"]):
        raise SystemExit("Config differs from the pre-registered one.")
    with open(args.models, "rb") as f:
        registered = pickle.load(f)
    P = cfg.pilot
    fa_max = P.fa_max_per_min

    def log(m):
        print(m, flush=True)

    train = Split(make_split(cfg, "train", P.n_train, P.seed_train, args.root)).preload(cfg)
    val = Split(make_split(cfg, "val", P.n_val, P.seed_val, args.root)).preload(cfg)
    res = {"note": "post-hoc trigger baseline; see SCIENTIFIC_DECISIONS.md 2026-10-01; "
                   "test split (seeds 40000+) never loaded",
           "grid": {"look_grid": [list(x) for x in P.look_grid], "tau": TAUS, "n_hold": N_HOLDS},
           "val": {}, "selected": {}, "fresh2": {}}

    selected = {}
    for b in BUDGETS:
        reg = registered[b]
        target = reg["burst"]["val"]["cost_monitor"]
        log(f"[tune] b={b} target cost {target / cfg.radio.num_re:.1f} probes/s (registered burst, val)")
        rows, best = [], None
        for L, d in P.look_grid:
            llr = fit_llr(train, factory("burst", cfg, b, burst_len=L, spacing=d), cfg)
            if (L, d) == (reg["burst"]["params"]["burst_len"], reg["burst"]["params"]["spacing"]):
                same = all(np.allclose(x, y) for p, q in zip(llr.params, reg["burst"]["llr"].params)
                           for x, y in zip(p, q))
                log(f"  LLR {L}x{d} matches registered burst LLR: {same}")
            for tau in TAUS:
                for nh in N_HOLDS:
                    r = {"look": f"{L}x{d}", "burst_len": L, "spacing": d, "tau": tau, "n_hold": nh,
                         **tune(val, cfg, b, L, d, tau, nh, llr, target, fa_max)}
                    rows.append(r)
                    if not r["feasible"]:
                        log(f"  look={L}x{d:<2d} tau={tau:g} N={nh}: infeasible "
                            f"(cost {r['cost_monitor'] / 32:.1f} probes/s at P_slow={r['p_slow']})")
                        continue
                    log(f"  look={L}x{d:<2d} tau={tau:g} N={nh}: P_slow={r['p_slow']:<5d} A={r['A']:6.2f} "
                        f"P_fail={r['p_fail']:.3f} FA/min={r['fa_per_min']:.2f} "
                        f"cost={r['cost_monitor'] / target:.3f}x{'' if r['eligible'] else '  [not eligible]'}")
                    if r["eligible"] and (best is None or r["p_fail"] < best[0]["p_fail"]):
                        best = (r, llr)
        res["val"][str(b)] = rows
        if best is None:
            log(f"[select] b={b}: no eligible trigger config")
            res["selected"][str(b)] = None
            continue
        r, llr = best
        selected[b] = (r, llr)
        res["selected"][str(b)] = r
        log(f"[select] b={b}: look={r['look']} tau={r['tau']:g} N_hold={r['n_hold']} "
            f"P_slow={r['p_slow']} A={r['A']:.2f} val P_fail={r['p_fail']:.3f}")

    fresh2 = Split(make_split(cfg, "fresh2", N_FRESH2, SEED_FRESH2, args.root)).preload(cfg)
    assert min(fresh2.seeds) >= SEED_FRESH2 and len(fresh2.seeds) == N_FRESH2
    for b in BUDGETS:
        reg = registered[b]
        outs = {}
        for k in ("burst", "decusum"):
            v = reg[k]
            outs[k] = run_split(fresh2, factory(v["kind"], cfg, b, **v["params"]), cfg,
                                llr=v["llr"], threshold=v["A"])
        if b in selected:
            r, llr = selected[b]
            mk = factory("trigger", cfg, b, burst_len=r["burst_len"], spacing=r["spacing"],
                         p_slow=r["p_slow"], tau=r["tau"], n_hold=r["n_hold"])
            outs["trigger"] = run_split(fresh2, mk, cfg, llr=llr, threshold=r["A"])
        row = {k: summary(o) for k, o in outs.items()}
        if "trigger" in outs:
            t, de = by_seed(outs["trigger"]), by_seed(outs["decusum"])
            seeds = [s for s in sorted(set(t) & set(de)) if t[s]["has_event"]]
            diff, lo, hi = paired_bootstrap([t[s]["fail"] for s in seeds], [de[s]["fail"] for s in seeds])
            row["trigger_minus_decusum"] = {"p_fail_diff": diff, "ci95": [lo, hi], "n_paired": len(seeds)}
        res["fresh2"][str(b)] = row
        for k, s in row.items():
            if k == "trigger_minus_decusum":
                log(f"[fresh2] b={b} trigger-DE P_fail diff={s['p_fail_diff']:+.3f} "
                    f"CI=[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] n={s['n_paired']}")
            else:
                log(f"[fresh2] b={b} {k:8s} P_fail={s['p_fail']:.3f} in_time={s['events_in_time']}/{s['n_event']} "
                    f"FA/min={s['fa_per_min']:.2f} cost_mon={s['cost_monitor'] / 32:.1f} "
                    f"cost_tot={s['cost_total'] / 32:.1f} probes/s delay_med={s['median_delay'] * 1e3:.0f}ms")

    res["runtime_s"] = time.time() - t_start
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "trigger.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    log(f"[done] {path}  {res['runtime_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
