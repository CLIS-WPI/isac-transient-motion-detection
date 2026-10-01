"""Validation-only CPU worker benchmark. Does not generate or evaluate the test split.

    PYTHONPATH=. python3 scripts/bench_workers.py --root /tmp/mdsense_bench --n 80
"""
import argparse
import os
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "1"

from mdsense import Config
from mdsense.evaluate import by_seed
from mdsense.pipeline import (Split, close_workers, factory, limit_blas_threads,
                              make_split, run_split)


class ZeroLLR:
    """Picklable stand-in so validation replay does not fit an LLR."""
    def __call__(self, x):
        return 0.0


def _same(a, b):
    sa, sb = by_seed(a), by_seed(b)
    if set(sa) != set(sb):
        return False
    for seed, oa in sa.items():
        ob = sb[seed]
        if oa["n_fa"] != ob["n_fa"] or oa["cost_monitor"] != ob["cost_monitor"]:
            return False
        if oa.get("fail", None) != ob.get("fail", None):
            return False
    return True


def _time_run(split, mk, cfg, workers, repeats, llr):
    kw = dict(llr=llr, threshold=1e9)
    run_split(split, mk, cfg, workers=workers, **kw)
    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = run_split(split, mk, cfg, workers=workers, **kw)
        times.append(time.perf_counter() - t0)
    return last, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/mdsense_bench")
    ap.add_argument("--n", type=int, default=80, help="validation episodes (not the test split)")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    limit_blas_threads(1)

    cfg = Config()
    d = make_split(cfg, "val", args.n, cfg.pilot.seed_val, args.root)
    split = Split(d).preload(cfg)
    mk = factory("uniform", cfg, 0.05, n_window=16)
    llr = ZeroLLR()
    print(f"[bench] n={args.n} BLAS threads=1 policy=uniform nw=16 (validation only)")

    try:
        serial, t1 = _time_run(split, mk, cfg, 1, args.repeats, llr)
        print(f"  workers= 1  {min(t1):.3f}–{max(t1):.3f}s  mean={sum(t1)/len(t1):.3f}s")
        out8, t8 = _time_run(split, mk, cfg, 8, args.repeats, llr)
        print(f"  workers= 8  {min(t8):.3f}–{max(t8):.3f}s  mean={sum(t8)/len(t8):.3f}s  "
              f"match_serial={_same(serial, out8)}")
        out16, t16 = _time_run(split, mk, cfg, 16, args.repeats, llr)
        print(f"  workers=16  {min(t16):.3f}–{max(t16):.3f}s  mean={sum(t16)/len(t16):.3f}s  "
              f"match_serial={_same(serial, out16)}")
        m8, m16 = sum(t8) / len(t8), sum(t16) / len(t16)
        pick = 8 if m8 <= m16 else 16
        print(f"[pick] workers={pick} for this host (lower mean wall time on validation replay)")
    finally:
        close_workers()


if __name__ == "__main__":
    main()
