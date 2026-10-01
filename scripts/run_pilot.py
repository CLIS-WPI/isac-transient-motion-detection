"""Pilot: tuned periodic burst vs adapted DE-CuSum (and equal-budget uniform).

    PYTHONPATH=. python scripts/run_pilot.py --root data --out results          # full
    # channels land in data/<fingerprint>/, not data/train directly
    PYTHONPATH=. python scripts/run_pilot.py --quick --root /tmp/d --out /tmp/r # smoke test

Order of operations (enforced): generate data -> fit LLRs on train -> tune on val
(including feature ablation) -> write preregistration.json -> event-free FA split ->
test split once -> verdict.
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
from mdsense.config import SCIENTIFIC_DECISIONS
from mdsense.pipeline import (make_split, Split, factory, fit_llr, run_split,
                              calibrate_threshold, calibrate_mu, limit_blas_threads,
                              feature_ablation)
from mdsense.evaluate import aggregate, paired_bootstrap, fa_rate_ci, paired_event_fails


def tune_budget(b, cfg, train, val, log):
    P = cfg.pilot
    fa_max = P.fa_max_per_min
    sel = {}

    best = None
    for nw in P.uniform_nw_grid:
        mk = factory("uniform", cfg, b, n_window=nw)
        llr = fit_llr(train, mk, cfg)
        A, agg, ok = calibrate_threshold(val, mk, llr, cfg, fa_max)
        log(f"  uniform nw={nw:<3d} A={A:7.2f} val {fmt(agg)}{'' if ok else '  [FA limit not met]'}")
        if ok and (best is None or agg["p_fail"] < best["val"]["p_fail"]):
            best = {"kind": "uniform", "params": {"n_window": nw}, "A": A, "llr": llr, "val": agg}
    sel["uniform"] = best

    best, burst_by_look = None, {}
    for L, d in P.look_grid:
        mk = factory("burst", cfg, b, burst_len=L, spacing=d)
        llr = fit_llr(train, mk, cfg)
        A, agg, ok = calibrate_threshold(val, mk, llr, cfg, fa_max)
        burst_by_look[(L, d)] = (llr, A, agg)
        log(f"  burst   look={L}x{d:<2d} A={A:7.2f} val {fmt(agg)}{'' if ok else '  [FA limit not met]'}")
        if ok and (best is None or agg["p_fail"] < best["val"]["p_fail"]):
            best = {"kind": "burst", "params": {"burst_len": L, "spacing": d}, "A": A, "llr": llr,
                    "val": agg}
    sel["burst"] = best
    if best is None:
        return sel
    target = best["val"]["cost_monitor"]            # realised cost of the tuned burst

    best = None
    for (L, d), (llr, A0, _) in burst_by_look.items():   # LLR of this look design, fitted on train
        for h in P.decusum_h_grid:
            mu, c, feas = calibrate_mu(val, cfg, b, L, h, llr, A0, target, d)
            if not feas:
                log(f"  decusum look={L}x{d:<2d} h={h:<4} infeasible (cost {c:.0f} > {target:.0f} REs/s)")
                continue
            for _ in range(2):                       # alternate: FA threshold <-> cost match
                mk = factory("decusum", cfg, b, burst_len=L, spacing=d, mu=mu, h=h)
                A, agg, ok = calibrate_threshold(val, mk, llr, cfg, fa_max)
                mu, c, feas = calibrate_mu(val, cfg, b, L, h, llr, A, target, d)
            mk = factory("decusum", cfg, b, burst_len=L, spacing=d, mu=mu, h=h)
            A, agg, ok = calibrate_threshold(val, mk, llr, cfg, fa_max)
            log(f"  decusum look={L}x{d:<2d} h={h:<4} mu={mu:6.3f} A={A:7.2f} val {fmt(agg)}"
                f"{'' if ok else '  [FA limit not met]'}")
            cost_ok = abs(agg["cost_monitor"] / target - 1) <= 0.05
            if ok and cost_ok and (best is None or agg["p_fail"] < best["val"]["p_fail"]):
                best = {"kind": "decusum", "params": {"burst_len": L, "spacing": d, "mu": mu, "h": h},
                        "A": A, "llr": llr, "val": agg}
    sel["decusum"] = best
    return sel


def fmt(a):
    return (f"P_fail={a['p_fail']:.3f} FA/min={a['fa_per_min']:.2f} "
            f"cost={a['cost_monitor'] / 32:.1f} probes/s (total {a['cost_total'] / 32:.1f}) "
            f"delay_med={a['median_delay'] * 1e3:.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=16,
                    help="CPU processes for episode loops (fork). 1 = serial. "
                         "Bench on this host: 16 < 8 wall time. Also MDSENSE_WORKERS.")
    args = ap.parse_args()
    os.environ["MDSENSE_WORKERS"] = str(max(1, args.workers))
    limit_blas_threads(1)

    cfg = Config()
    if args.quick:
        cfg = dataclasses.replace(cfg, pilot=dataclasses.replace(
            cfg.pilot, budgets=(0.05,), go_budgets=(0.05,), n_train=60, n_val=60, n_test=80,
            n_neg=40, uniform_nw_grid=(16,), look_grid=((16, 8),), decusum_h_grid=(5.0,)))
    P = cfg.pilot
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    log(f"[cpu] workers={os.environ['MDSENSE_WORKERS']} BLAS threads=1 "
        f"(fork pool reused; noise/busy/H cached in parent)")
    train = Split(make_split(cfg, "train", P.n_train, P.seed_train, args.root)).preload(cfg)
    val = Split(make_split(cfg, "val", P.n_val, P.seed_val, args.root)).preload(cfg)

    selected = {}
    ablation = {}
    for b in P.budgets:
        log(f"[tune] budget {b:.3f} ({b / cfg.radio.slot_duration:.0f} probes/s nominal)")
        selected[b] = tune_budget(b, cfg, train, val, log)
        burst = selected[b].get("burst")
        if P.ablation_on_val and burst is not None:
            mk = factory("burst", cfg, b, **burst["params"])
            log(f"[ablation] b={b:.3f} burst look on val (energy / Doppler / both)")
            ablation[b] = feature_ablation(train, val, mk, cfg, P.fa_max_per_min)
            for name, row in ablation[b].items():
                log(f"  {name:8s} A={row['A']:7.2f} P_fail={row['p_fail']:.3f} "
                    f"FA/min={row['fa_per_min']:.2f}{'' if row['ok'] else '  [FA limit not met]'}")

    prereg = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "decisions": SCIENTIFIC_DECISIONS,
              "config": dataclasses.asdict(cfg),
              "ablation_val": {str(b): v for b, v in ablation.items()},
              "selected": {str(b): {k: (None if v is None else
                                        {"params": v["params"], "A": v["A"], "val": v["val"]})
                                    for k, v in s.items()} for b, s in selected.items()}}
    with open(os.path.join(args.out, "preregistration.json"), "w") as f:
        json.dump(prereg, f, indent=2, default=float)
    with open(os.path.join(args.out, "selected_models.pkl"), "wb") as f:
        pickle.dump(selected, f)
    log("[prereg] written; selection frozen. long-negative FA then the test split once")

    if P.n_neg > 0:
        neg = Split(make_split(cfg, "neg", P.n_neg, P.seed_neg, args.root,
                               force_event=False)).preload(cfg)
        log(f"[neg] {P.n_neg} event-free episodes seed>={P.seed_neg} (does not retune A)")
    else:
        neg = None

    test = Split(make_split(cfg, "test", P.n_test, P.seed_test, args.root)).preload(cfg)
    results, verdicts, neg_results = {}, {}, {}
    for b, sel in selected.items():
        outs = {}
        for k, v in sel.items():
            if v is None:
                continue
            mk = factory(v["kind"], cfg, b, **v["params"])
            if neg is not None:
                neg_results.setdefault(b, {})[k] = aggregate(
                    run_split(neg, mk, cfg, llr=v["llr"], threshold=v["A"]))
                log(f"[neg]  b={b:.3f} {k:8s} {fmt(neg_results[b][k])}")
            outs[k] = run_split(test, mk, cfg, llr=v["llr"], threshold=v["A"])
            log(f"[test] b={b:.3f} {k:8s} {fmt(aggregate(outs[k]))}")
        res = {k: aggregate(o) for k, o in outs.items()}
        if b in P.go_budgets and not ("burst" in outs and "decusum" in outs):
            missing = [k for k in ("burst", "decusum") if k not in outs]
            verdicts[b] = {"go": False, "invalid": f"no feasible tuned config for {missing}"}
            log(f"[go?] b={b:.3f} INVALID comparison: no feasible config for {missing}")
        if "burst" in outs and "decusum" in outs and b in P.go_budgets:
            fb, fd = paired_event_fails(outs["burst"], outs["decusum"], P.go_max_event_duration)
            gain, lo, hi = paired_bootstrap(fb, fd)
            fa_d = res["decusum"]["fa_per_min"]
            if P.go_cost_basis != "monitor":
                raise ValueError(P.go_cost_basis)
            ratio = res["decusum"]["cost_monitor"] / res["burst"]["cost_monitor"]
            ratio_total = res["decusum"]["cost_total"] / res["burst"]["cost_total"]
            fa_neg = (neg_results.get(b, {}).get("decusum", {}) or {}).get("fa_per_min", np.nan)
            crit = {"gain>=min": gain >= P.go_min_abs_gain, "CI_low>0": lo > 0,
                    "FA_ok": fa_d <= P.go_fa_slack * P.fa_max_per_min,
                    "cost_ok": P.go_cost_ratio[0] <= ratio <= P.go_cost_ratio[1],
                    "FA_neg_ok": (not np.isfinite(fa_neg)) or fa_neg <= P.go_fa_neg_slack * P.fa_max_per_min}
            verdicts[b] = {"n_events": len(fb), "gain": gain, "ci": [lo, hi], "fa_decusum": fa_d,
                           "fa_ci_decusum": fa_rate_ci(outs["decusum"]), "cost_ratio": ratio,
                           "cost_total_ratio": ratio_total, "fa_neg_decusum": fa_neg,
                           "criteria": crit, "go": all(crit.values())}
            log(f"[go?] b={b:.3f} n={len(fb)} gain={gain:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
                f"FA_DE={fa_d:.2f}/min FA_neg={fa_neg:.2f}/min cost_mon={ratio:.2f} "
                f"cost_tot={ratio_total:.2f} -> {crit}")
        results[b] = res
    go = bool(verdicts) and all(v["go"] for v in verdicts.values())
    invalid = [b for b, v in verdicts.items() if "invalid" in v]
    log(f"[verdict] {'INVALID (fix setup, not a result)' if invalid else 'GO' if go else 'NO-GO'}"
        f"  (all pre-registered go budgets must pass)")
    log(f"[time] {(time.time() - t_start) / 60:.1f} min")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"results": {str(b): r for b, r in results.items()},
                   "neg": {str(b): v for b, v in neg_results.items()},
                   "ablation_val": {str(b): v for b, v in ablation.items()},
                   "verdicts": {str(b): v for b, v in verdicts.items()}, "go": go,
                   "decisions": SCIENTIFIC_DECISIONS},
                  f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))


if __name__ == "__main__":
    main()
