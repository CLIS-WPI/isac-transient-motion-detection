"""The evaluator is the only module that reads event labels.

Failure : no alarm in [t0, min(t0 + deadline, t_end)]. Detections after the event
          ended, or after the deadline, are failures. An alarm before t0 is not a
          detection (it is a false alarm, and its refractory period may blind the policy).
False alarm: any alarm after warm-up outside [t0, t_end + grace]; rate per minute of
          event-free time.
Cost    : distinct transmitted probe slots x REs, counted at transmission, split into
          monitoring and total (monitoring + post-alarm service).
"""
import numpy as np


def episode_outcome(log, event, cfg):
    sc, det, r = cfg.scenario, cfg.det, cfg.radio
    T, warm = sc.duration, sc.warmup
    alarms = np.array([a for a in log.alarm_times if a >= warm])
    out = {"has_event": event is not None}
    excl = 0.0
    if event is not None:
        t0, t1 = event["t0"], event["t0"] + event["D"]
        dl = min(t0 + det.deadline, t1)
        hit = alarms[(alarms >= t0) & (alarms <= dl)]
        out["fail"] = hit.size == 0
        late = alarms[(alarms >= t0) & (alarms <= t1 + det.fa_grace)]
        out["delay"] = float(late[0] - t0) if late.size else np.nan
        out["event_D"] = event["D"]
        fa = alarms[(alarms < t0) | (alarms > t1 + det.fa_grace)]
        excl = min(t1 + det.fa_grace, T) - t0
    else:
        fa = alarms
    out["n_fa"] = int(fa.size)
    out["null_time"] = (T - warm) - excl
    span = T - warm
    k = r.num_re
    out["cost_monitor"] = len(log.applied_slots("monitor", warm)) * k / span     # REs per second
    out["cost_total"] = len(log.applied_slots(None, warm)) * k / span
    out["n_reject"] = len(log.rejects)
    return out


def by_seed(outs):
    """Map seed -> outcome. Pair burst/DE on seed, not on worker finish order."""
    return {o["seed"]: o for o in outs}


def paired_event_fails(burst_outs, decusum_outs, max_event_duration):
    """Aligned fail bits for episodes that have an event no longer than the go cutoff."""
    b, d = by_seed(burst_outs), by_seed(decusum_outs)
    fb, fd = [], []
    for seed in sorted(set(b) & set(d)):
        ob = b[seed]
        if ob.get("has_event") and ob.get("event_D", np.inf) <= max_event_duration:
            fb.append(ob["fail"])
            fd.append(d[seed]["fail"])
    return fb, fd


def aggregate(outs):
    ev = [o for o in outs if o["has_event"]]
    null_min = sum(o["null_time"] for o in outs) / 60.0
    return {"p_fail": float(np.mean([o["fail"] for o in ev])) if ev else np.nan,
            "n_event": len(ev),
            "fa_per_min": sum(o["n_fa"] for o in outs) / max(null_min, 1e-9),
            "cost_monitor": float(np.mean([o["cost_monitor"] for o in outs])),
            "cost_total": float(np.mean([o["cost_total"] for o in outs])),
            "median_delay": (float(np.nanmedian(d)) if ev and np.isfinite(d := np.array(
                [o["delay"] for o in ev], float)).any() else np.nan)}


def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """CI of mean(a - b) over paired episodes."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fa_rate_ci(outs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = np.array([o["n_fa"] for o in outs], float)
    t = np.array([o["null_time"] for o in outs], float) / 60.0
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    r = n[idx].sum(1) / t[idx].sum(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
