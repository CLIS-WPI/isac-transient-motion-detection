"""Dataset generation, training-window collection, and calibration on validation.

Splits: train (LLR fitting only), val (threshold / budget / hyper-parameter tuning),
test (touched once, after the pre-registration file is written).
Channels and labels are stored in separate files; only the evaluator reads labels.
On-disk channels live under `root/<fingerprint>/` where the fingerprint hashes
radio, scenario, backend, snapshot_slots, and force_event.

Episode loops use a fork pool whose workers stay alive across calibration calls.
Caches (channel, busy, noise) are filled in the parent before the first fork.
Do not import Sionna/CUDA in this process; GPU channel writes go through spawn.
"""
from dataclasses import asdict, dataclass, field
import atexit
import hashlib
import json
import multiprocessing
import os
import time
import numpy as np

from .body import sample_scenario
from .channel import dense_channel
from .env import episode_noise, public_info, run_episode
from .evaluate import episode_outcome, aggregate
from .keyed_rng import STREAM_BUSY, keyed_uniform
from .policies import UniformPolicy, BurstPolicy, DECuSumPolicy
from .score_model import GaussianLLR, SliceLLR

# Inherited by forked workers: (split, cfg). Policy kwargs travel with each task.
_MP_STATE = None
_POOL = None
_POOL_KEY = None


def limit_blas_threads(n=1):
    n = str(int(n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[var] = n
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=int(n))
    except ImportError:
        pass


def _worker_init():
    limit_blas_threads(1)


def resolve_workers(workers=None):
    if workers is not None:
        return max(1, int(workers))
    env = os.environ.get("MDSENSE_WORKERS")
    if env is not None and env.strip() != "":
        return max(1, int(env))
    return 1


def close_workers():
    global _POOL, _POOL_KEY, _MP_STATE
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
        _POOL_KEY = None
    _MP_STATE = None


atexit.register(close_workers)


# ------------------------------------------------------------------ data
def cache_spec(cfg, backend="analytic", snapshot_slots=4, force_event=None):
    """Inputs that determine the on-disk noiseless channel. Busy/noise RAM caches are separate."""
    return {
        "radio": asdict(cfg.radio),
        "scenario": asdict(cfg.scenario),
        "backend": backend,
        "snapshot_slots": int(snapshot_slots),
        "force_event": force_event,
        "include_clutter": True,
    }


def cache_fingerprint(cfg, backend="analytic", snapshot_slots=4, force_event=None):
    blob = json.dumps(cache_spec(cfg, backend, snapshot_slots, force_event),
                      sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def cache_root(root, cfg, backend="analytic", snapshot_slots=4, force_event=None):
    """`root/<fingerprint>/`. A radio or scenario change cannot reuse another setting's npy files."""
    fp = cache_fingerprint(cfg, backend, snapshot_slots, force_event)
    d = os.path.join(root, fp)
    os.makedirs(d, exist_ok=True)
    spec = cache_spec(cfg, backend, snapshot_slots, force_event)
    spec["fingerprint"] = fp
    spec_path = os.path.join(d, "cache_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            old = json.load(f)
        if old.get("fingerprint") not in (None, fp):
            raise RuntimeError(f"cache spec mismatch under {d}: {old.get('fingerprint')} != {fp}")
    else:
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2, default=str)
    return d


def _sionna_write_batch(payload):
    """Spawn child: init CUDA, write missing npy files, exit. Must not leak into the fork parent."""
    radio, duration, snapshot_slots, items = payload
    from mdsense.channel import dense_channel as _dense
    for path, scn in items:
        if os.path.exists(path):
            continue
        np.save(path, _dense(scn, radio, duration, backend="sionna",
                             snapshot_slots=snapshot_slots))


def make_split(cfg, name, n, seed0, root, backend="analytic", verbose=True, snapshot_slots=4,
               force_event=None):
    d = os.path.join(cache_root(root, cfg, backend, snapshot_slots, force_event), name)
    os.makedirs(d, exist_ok=True)
    labels = {}
    t0 = time.time()
    sionna_missing = []
    for i in range(n):
        seed = seed0 + i
        scn = sample_scenario(seed, cfg.scenario, cfg.radio, force_event=force_event)
        path = os.path.join(d, f"ep{seed}.npy")
        if not os.path.exists(path):
            if backend == "analytic":
                np.save(path, dense_channel(scn, cfg.radio, cfg.scenario.duration, backend="analytic"))
            elif backend == "sionna":
                sionna_missing.append((path, scn))
            else:
                raise ValueError(backend)
        labels[str(seed)] = {"event": scn.event,
                             "nuisances": [nu["kind"] for nu in scn.nuisances]}
    if sionna_missing:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(target=_sionna_write_batch,
                           args=((cfg.radio, cfg.scenario.duration, snapshot_slots, sionna_missing),))
        proc.start()
        proc.join()
        if proc.exitcode:
            raise RuntimeError(f"Sionna spawn worker failed (exit {proc.exitcode})")
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump(labels, f, default=float)
    if verbose:
        print(f"[data] {name}: {n} episodes in {time.time() - t0:.1f}s -> {d}")
    return d


class Split:
    def __init__(self, d):
        self.d = d
        with open(os.path.join(d, "labels.json")) as f:
            self._labels = json.load(f)
        self.seeds = sorted(int(s) for s in self._labels)
        self._channels = {}
        self._busy = {}
        self._noise = {}

    def channel(self, seed):
        H = self._channels.get(seed)
        if H is None:
            H = np.load(os.path.join(self.d, f"ep{seed}.npy"))
            H.setflags(write=False)
            self._channels[seed] = H
        return H

    def busy(self, seed, n_slots, p_busy):
        key = (int(seed), int(n_slots), float(p_busy))
        m = self._busy.get(key)
        if m is None:
            m = keyed_uniform(seed, STREAM_BUSY, np.arange(n_slots)) < p_busy
            m.setflags(write=False)
            self._busy[key] = m
        return m

    def noise(self, seed, n_slots, num_re, rx=0):
        key = (int(seed), int(n_slots), int(num_re), int(rx))
        N = self._noise.get(key)
        if N is None:
            N = np.ascontiguousarray(episode_noise(seed, n_slots, num_re, rx))
            N.setflags(write=False)
            self._noise[key] = N
        return N

    def preload(self, cfg=None):
        """Fill caches in the parent before fork so workers inherit them via COW."""
        for seed in self.seeds:
            H = self.channel(seed)
            if cfg is not None:
                n, K = H.shape
                self.busy(seed, n, cfg.scenario.comm_busy_prob)
                self.noise(seed, n, K)
        return self

    def event(self, seed):                       # evaluator side only
        return self._labels[str(seed)]["event"]

    def subset(self, max_n):
        s = Split.__new__(Split)
        s.d, s._labels, s.seeds = self.d, self._labels, self.seeds[:max_n]
        s._channels, s._busy, s._noise = self._channels, self._busy, self._noise
        return s


# ------------------------------------------------------------------ policies
@dataclass
class PolicyMaker:
    kind: str
    budget: float
    params: dict = field(default_factory=dict)
    pub: object = None
    du: object = None
    det: object = None

    def __call__(self, llr=None, threshold=np.inf, collect=False):
        kw = dict(llr=llr, threshold=threshold, collect=collect)
        p = self.params
        if self.kind == "uniform":
            return UniformPolicy(self.pub, self.du, self.det, self.budget, p["n_window"], **kw)
        if self.kind == "burst":
            return BurstPolicy(self.pub, self.du, self.det, self.budget, p["burst_len"],
                               p.get("spacing", 1), **kw)
        if self.kind == "decusum":
            return DECuSumPolicy(self.pub, self.du, self.det, p["burst_len"], p["mu"], p["h"],
                                 p.get("spacing", 1), **kw)
        raise ValueError(self.kind)


def factory(kind, cfg, budget, **p):
    pub = public_info(cfg)
    return PolicyMaker(kind, budget, dict(p), pub, cfg.du, cfg.det)


def _caches(split, seed, cfg, n_slots, num_re):
    return (split.busy(seed, n_slots, cfg.scenario.comm_busy_prob),
            split.noise(seed, n_slots, num_re))


def _mp_run_item(item):
    seed, maker, make_kw, keep_logs = item
    split, cfg = _MP_STATE
    H = split.channel(seed)
    busy, noise = _caches(split, seed, cfg, H.shape[0], H.shape[1])
    pol = maker(**make_kw)
    log = run_episode(H, seed, pol, cfg, busy=busy, noise=noise)
    o = episode_outcome(log, split.event(seed), cfg)
    o["seed"] = seed
    if keep_logs:
        o["log"] = log
    return o


def _label_records(records, ev, cfg):
    x0, x1 = [], []
    for x, ts in records:
        if ts[0] < cfg.scenario.warmup:
            continue
        if ev is None:
            x0.append(x)
            continue
        t0, t1 = ev["t0"], ev["t0"] + ev["D"]
        inside = np.mean((ts >= t0) & (ts <= t1))
        near = np.any((ts >= t0 - 0.05) & (ts <= t1 + cfg.det.fa_grace))
        if inside >= 0.5:
            x1.append(x)
        elif not near:
            x0.append(x)
    return x0, x1


def _mp_windows_item(item):
    seed, maker, make_kw = item
    split, cfg = _MP_STATE
    H = split.channel(seed)
    busy, noise = _caches(split, seed, cfg, H.shape[0], H.shape[1])
    pol = maker(**make_kw)
    run_episode(H, seed, pol, cfg, busy=busy, noise=noise)
    return _label_records(pol.records, split.event(seed), cfg)


def _map_items(fn, items, split, cfg, workers):
    """Parent caches must already be warm. Fork pool is reused while (split, workers, cfg) match;
    workers inherit _MP_STATE at fork time, so any cfg change must recreate the pool."""
    global _MP_STATE, _POOL, _POOL_KEY
    workers = resolve_workers(workers)
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        workers = 1
    _MP_STATE = (split, cfg)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    key = (id(split), workers, repr(cfg))
    if _POOL is None or _POOL_KEY != key:
        close_workers()
        _MP_STATE = (split, cfg)
        ctx = multiprocessing.get_context("fork")
        _POOL = ctx.Pool(processes=workers, initializer=_worker_init)
        _POOL_KEY = key
    chunk = max(1, len(items) // (workers * 4))
    return _POOL.map(fn, items, chunksize=chunk)


def run_split(split, make_policy, cfg, keep_logs=False, workers=None, **make_kw):
    split.preload(cfg)
    seeds = list(split.seeds)
    items = [(seed, make_policy, make_kw, keep_logs) for seed in seeds]
    outs = _map_items(_mp_run_item, items, split, cfg, workers)
    outs.sort(key=lambda o: o["seed"])
    return outs


def collect_windows(split, make_policy, cfg, workers=None):
    """Label windows from the motion scenario: event if >= 50% of the window's samples lie
    inside the event; null if none lie within [t0 - 50 ms, t_end + grace]; else dropped."""
    split.preload(cfg)
    kw = {"collect": True}
    items = [(seed, make_policy, kw) for seed in split.seeds]
    parts = _map_items(_mp_windows_item, items, split, cfg, workers)
    X0, X1 = [], []
    for x0, x1 in parts:
        X0.extend(x0)
        X1.extend(x1)
    return np.array(X0), np.array(X1)


def fit_llr(split, make_policy, cfg):
    X0, X1 = collect_windows(split, make_policy, cfg)
    return GaussianLLR(clip=cfg.det.llr_clip).fit(X0, X1)


def feature_ablation(train, val, make_policy, cfg, fa_max):
    """Val-only energy / Doppler / both. Does not touch the test split or retune the primary A."""
    X0, X1 = collect_windows(train, make_policy, cfg)
    out = {}
    for name, idx in (("both", (0, 1)), ("energy", (0,)), ("doppler", (1,))):
        llr = GaussianLLR(clip=cfg.det.llr_clip).fit(X0[:, list(idx)], X1[:, list(idx)])
        A, agg, ok = calibrate_threshold(val, make_policy, SliceLLR(llr, idx), cfg, fa_max)
        out[name] = {"A": float(A), "ok": bool(ok), "p_fail": float(agg["p_fail"]),
                     "fa_per_min": float(agg["fa_per_min"]), "n0": int(len(X0)), "n1": int(len(X1))}
    return out


# ------------------------------------------------------------------ calibration
def calibrate_threshold(split, make_policy, llr, cfg, fa_max, lo=0.5, hi=500.0, iters=10):
    """Smallest threshold whose validation FA rate is <= fa_max (bisection in log A)."""
    def fa(A):
        agg = aggregate(run_split(split, make_policy, cfg, llr=llr, threshold=A))
        return agg["fa_per_min"], agg
    f_hi, agg_hi = fa(hi)
    if f_hi > fa_max:
        return hi, agg_hi, False
    best = (hi, agg_hi)
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        f_m, agg_m = fa(np.exp(m))
        if f_m <= fa_max:
            b, best = m, (np.exp(m), agg_m)
        else:
            a = m
    return best[0], best[1], True


def calibrate_mu(split, cfg, budget, L, h, llr, A, target, spacing=1, lo=1e-3, hi=30.0, iters=10, tol=0.03):
    """mu such that DE-CuSum's realised monitoring cost on validation equals `target`
    (REs/s), i.e. the realised cost of the tuned burst baseline, not its nominal budget:
    busy slots make realised cost lower than requested for every policy."""

    def cost(mu):
        mk = factory("decusum", cfg, budget, burst_len=L, mu=mu, h=h, spacing=spacing)
        agg = aggregate(run_split(split, mk, cfg, llr=llr, threshold=A))
        return agg["cost_monitor"]
    c_lo, c_hi = cost(lo), cost(hi)
    if c_lo > target * (1 + tol):
        return lo, c_lo, False              # cannot sleep enough: infeasible at this budget
    if c_hi < target * (1 - tol):
        return hi, c_hi, True               # budget not binding
    a, b = np.log(lo), np.log(hi)
    mu, c = hi, c_hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        c = cost(np.exp(m))
        mu = np.exp(m)
        if abs(c - target) <= tol * target:
            break
        if c > target:
            b = m
        else:
            a = m
    return mu, c, True
