"""Counter-based randomness: every draw is a pure function of its key, e.g.
(scenario_seed, stream, slot, re, rx). Two policies that observe the same slot see
the same noise, whatever they requested before (common random numbers)."""
import numpy as np

STREAM_NOISE = 1
STREAM_BUSY = 2

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(x):
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def keyed_uint64(*keys):
    h = _GOLDEN
    with np.errstate(over="ignore"):
        for k in keys:
            k = np.asarray(k, dtype=np.int64).astype(np.uint64)
            h = _mix(h ^ _mix(k + _GOLDEN))
    return h


def keyed_uniform(*keys):
    """Uniform in (0, 1)."""
    return ((keyed_uint64(*keys) >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53


def keyed_cnormal(seed, slot, re, rx=0, stream=STREAM_NOISE):
    """Unit-variance circular complex Gaussian (Box-Muller)."""
    u1 = keyed_uniform(seed, stream, 0, slot, re, rx)
    u2 = keyed_uniform(seed, stream, 1, slot, re, rx)
    r = np.sqrt(-np.log(u1))                # |.|^2 ~ Exp(1) -> unit variance
    return r * np.exp(2j * np.pi * u2)
