import numpy as np

from mdsense import Config
from mdsense.env import public_info
from mdsense.policies import TriggerPolicy


def _look_starts(pol, s0, s1):
    """Drive the scheduler tick by tick; return the first slot of each new look."""
    starts = {}
    for slot in range(s0, s1, pol.pub.tick_slots):
        for r in pol._monitor_requests(slot):
            starts.setdefault(r.group, r.target_slot)
    return sorted(starts.values())


def test_trigger_switches_to_fast_and_returns_to_slow():
    cfg = Config()
    pol = TriggerPolicy(public_info(cfg), cfg.du, cfg.det, burst_len=8, p_slow=400, tau=2.0,
                        n_hold=3, spacing=4, llr=lambda x: 0.0, threshold=np.inf)
    span = pol.span                                         # 32 slots
    slow = _look_starts(pol, 0, 1000)
    assert np.all(np.diff(slow) == 400)

    assert not pol._update(1.0) and pol.fast_left == 0      # llr <= tau: stays slow
    now = 1000
    pol._update(5.0)                                        # llr > tau: fast mode
    assert pol.fast_left == 3
    fast = _look_starts(pol, now, now + 1000)
    gaps = np.diff(fast)
    assert fast[0] <= now + pol.first_ok + pol.pub.tick_slots  # reacts at the next tick
    assert list(gaps[:3]) == [span, span, 400]              # 3 back-to-back looks, then slow
    assert np.all(gaps[3:] == 400) and pol.fast_left == 0
