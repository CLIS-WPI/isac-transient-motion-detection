"""Post-hoc analyses for the paper. Never touches the test split (seeds 40000+).

  1. ablation   : energy-only vs both features, for burst AND DE-CuSum (val; DE cost re-matched)
  2. same_look  : burst with DE-CuSum's selected look (val)
  3. fa_curve   : P_fail vs FA at 5 % by sweeping A, LLR clip 8 (as registered) and 20 (val)
  4. budget     : fixed look 16x4, h=2; tuned on val, evaluated on a FRESH split (seeds 60000+)

    PYTHONPATH=. python scripts/analysis_posthoc.py --root data --prereg results/preregistration.json \
        --out analysis --workers 16 [--only ablation,same_look,fa_curve,budget] [--quick]
"""
import argparse
import dataclasses
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdsense import Config
from mdsense.pipeline import (make_split, Split, factory, run_split, collect_windows,
                              calibrate_threshold, calibrate_mu, limit_blas_threads)
from mdsense.evaluate import aggregate
from mdsense.score_model import GaussianLLR, SliceLLR

FEATS = {"both": (0, 1), "energy": (0,)}
SEED_CURVE = 60_000


def llr_from(X0, X1, idx, clip):
    return SliceLLR(GaussianLLR(clip=clip).fit(X0[:, list(idx)], X1[:, list(idx)]), idx)


def tune_de(val, cfg, b, L, d, h, llr, A0, target, fa_max):
    """Same alternation as run_pilot: cost match <-> FA threshold."""
    mu, _, feas = calibrate_mu(val, cfg, b, L, h, llr, A0, target, d)
    if not feas:
        return None
    for _ in range(2):
        mk = factory("decusum", cfg, b, burst_len=L, spacing=d, mu=mu, h=h)
        A, _, _ = calibrate_threshold(val, mk, llr, cfg, fa_max)
        mu, _, _ = calibrate_mu(val, cfg, b, L, h, llr, A, target, d)
    mk = factory("decusum", cfg, b, burst_len=L, spacing=d, mu=mu, h=h)
    A, agg, ok = calibrate_threshold(val, mk, llr, cfg, fa_max)
    return {"mu": mu, "A": A, "ok": ok, **agg}


def row(agg, **extra):
    keys = ("p_fail", "fa_per_min", "cost_monitor", "cost_total", "median_delay", "A", "ok", "mu")
    return {**{k: agg[k] for k in keys if k in agg}, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--prereg", default="results/preregistration.json")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="ablation,same_look,fa_curve,budget")
    ap.add_argument("--quick", action="store_true", help="tiny splits, plumbing check only")
    args = ap.parse_args()
    os.environ["MDSENSE_WORKERS"] = str(max(1, args.workers))
    limit_blas_threads(1)
    todo = set(args.only.split(","))
    os.makedirs(args.out, exist_ok=True)

    with open(args.prereg) as f:
        prereg = json.load(f)
    cfg = Config()
    norm = lambda x: json.loads(json.dumps(x, default=float))
    if norm(dataclasses.asdict(cfg)) != norm(prereg["config"]):
        raise SystemExit("Config differs from the pre-registered one; analyses would not be comparable.")
    P = cfg.pilot
    n_tr, n_va, n_cu = (40, 40, 60) if args.quick else (P.n_train, P.n_val, 300)
    fa_max = P.fa_max_per_min
    sel = prereg["selected"]
    res = {"note": "post-hoc; test split (seeds 40000+) never loaded", "started": time.ctime()}

    def log(m):
        print(m, flush=True)

    train = Split(make_split(cfg, "train", n_tr, P.seed_train, args.root)).preload(cfg)
    val = Split(make_split(cfg, "val", n_va, P.seed_val, args.root)).preload(cfg)
    go_b = list(P.go_budgets)[:1] if args.quick else list(P.go_budgets)

    # ---------------------------------------------------------------- 1 + 2
    if todo & {"ablation", "same_look"}:
        res["ablation"], res["same_look"] = {}, {}
        for b in go_b:
            sb, sd = sel[str(b)]["burst"], sel[str(b)]["decusum"]
            target = sb["val"]["cost_monitor"]       # DE is matched to the tuned burst, as registered
            Lb, db = sb["params"]["burst_len"], sb["params"]["spacing"]
            Ld, dd, h = sd["params"]["burst_len"], sd["params"]["spacing"], sd["params"]["h"]
            mk_b = factory("burst", cfg, b, burst_len=Lb, spacing=db)
            mk_dlook = factory("burst", cfg, b, burst_len=Ld, spacing=dd)
            Xb = collect_windows(train, mk_b, cfg)
            Xd = Xb if (Ld, dd) == (Lb, db) else collect_windows(train, mk_dlook, cfg)
            out = {}
            for name, idx in FEATS.items():
                llr_b = llr_from(*Xb, idx, cfg.det.llr_clip)
                A, agg, ok = calibrate_threshold(val, mk_b, llr_b, cfg, fa_max)
                out[f"burst_{name}"] = row(agg, A=A, ok=ok)
                llr_d = llr_from(*Xd, idx, cfg.det.llr_clip)
                A0, agg0, ok0 = calibrate_threshold(val, mk_dlook, llr_d, cfg, fa_max)
                if name == "both":
                    res["same_look"][str(b)] = {"look": f"{Ld}x{dd}", "burst": row(agg0, A=A0, ok=ok0),
                                                "decusum_registered_val": sd["val"]}
                    log(f"[same_look] b={b} look {Ld}x{dd}: burst P_fail={agg0['p_fail']:.3f} "
                        f"vs DE {sd['val']['p_fail']:.3f}")
                de = tune_de(val, cfg, b, Ld, dd, h, llr_d, A0, target, fa_max)
                out[f"decusum_{name}"] = None if de is None else row(de)
                de_txt = "infeasible" if de is None else f"{de['p_fail']:.3f}"
                log(f"[ablation] b={b} {name:6s} burst P_fail={agg['p_fail']:.3f}  DE P_fail={de_txt}")
            res["ablation"][str(b)] = out

    # ---------------------------------------------------------------- 3
    if "fa_curve" in todo:
        b = 0.05 if 0.05 in go_b else go_b[0]
        sb, sd = sel[str(b)]["burst"], sel[str(b)]["decusum"]
        curves = {}
        for clip in (cfg.det.llr_clip, 20.0):
            for kind, s in (("burst", sb), ("decusum", sd)):
                p = s["params"]
                look = factory("burst", cfg, b, burst_len=p["burst_len"], spacing=p["spacing"])
                X0, X1 = collect_windows(train, look, cfg)
                llr = llr_from(X0, X1, (0, 1), clip)
                mk = look if kind == "burst" else factory("decusum", cfg, b, **p)
                pts = []
                for A in np.geomspace(2.0, 60.0, 12):
                    agg = aggregate(run_split(val, mk, cfg, llr=llr, threshold=float(A)))
                    pts.append(row(agg, A=float(A)))
                curves[f"{kind}_clip{clip:g}"] = pts
                log(f"[fa_curve] {kind} clip={clip:g} done")
        res["fa_curve"] = {"budget": b, "note": "DE mu fixed at its registered value", "curves": curves}
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, pts in curves.items():
            ax.plot([q["fa_per_min"] for q in pts], [q["p_fail"] for q in pts],
                    "o-" if "clip8" in name else "s--", label=name, ms=3)
        ax.axvline(fa_max, color="k", lw=0.8, ls=":")
        ax.set(xscale="symlog", xlabel="false alarms per minute (val)", ylabel="P_fail (val)",
               title=f"Threshold sweep, budget {b:.3f}")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fa_curve.png"), dpi=150)

    # ---------------------------------------------------------------- 4
    if "budget" in todo:
        curve = Split(make_split(cfg, "curve", n_cu, SEED_CURVE, args.root)).preload(cfg)
        budgets = (0.05,) if args.quick else (0.025, 0.05, 0.1, 0.15, 0.2)
        L, d, h = 16, 4, 2.0
        out = {}
        for b in budgets:
            mk_u = factory("uniform", cfg, b, n_window=8)
            mk_b = factory("burst", cfg, b, burst_len=L, spacing=d)
            llr_u = llr_from(*collect_windows(train, mk_u, cfg), (0, 1), cfg.det.llr_clip)
            llr_b = llr_from(*collect_windows(train, mk_b, cfg), (0, 1), cfg.det.llr_clip)
            Au, _, _ = calibrate_threshold(val, mk_u, llr_u, cfg, fa_max)
            Ab, agg_b, _ = calibrate_threshold(val, mk_b, llr_b, cfg, fa_max)
            de = tune_de(val, cfg, b, L, d, h, llr_b, Ab, agg_b["cost_monitor"], fa_max)
            r = {"uniform": row(aggregate(run_split(curve, mk_u, cfg, llr=llr_u, threshold=Au)), A=Au),
                 "burst": row(aggregate(run_split(curve, mk_b, cfg, llr=llr_b, threshold=Ab)), A=Ab)}
            if de is not None:
                mk_d = factory("decusum", cfg, b, burst_len=L, spacing=d, mu=de["mu"], h=h)
                r["decusum"] = row(aggregate(run_split(curve, mk_d, cfg, llr=llr_b, threshold=de["A"])),
                                   A=de["A"], mu=de["mu"])
            out[str(b)] = r
            log(f"[budget] b={b}: " + "  ".join(f"{k} P_fail={v['p_fail']:.3f}" for k, v in r.items()))
        res["budget"] = {"look": f"{L}x{d}", "h": h, "eval_split": f"curve seeds {SEED_CURVE}+",
                         "points": out}
        fig, ax = plt.subplots(figsize=(5, 4))
        for k, mk in (("uniform", "^:"), ("burst", "s--"), ("decusum", "o-")):
            pts = [(out[b][k]["cost_monitor"] / cfg.radio.num_re, out[b][k]["p_fail"])
                   for b in out if k in out[b]]
            ax.plot(*zip(*pts), mk, label=k)
        ax.set(xlabel="realised monitoring probes / s", ylabel="P_fail (deadline 200 ms)",
               title="Fresh split, FA <= 1/min calibrated on val")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "budget_curve.png"), dpi=150)

    res["finished"] = time.ctime()
    with open(os.path.join(args.out, "posthoc.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
    log(f"[done] {os.path.join(args.out, 'posthoc.json')}")


if __name__ == "__main__":
    main()
