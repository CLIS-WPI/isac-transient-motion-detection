import dataclasses
import numpy as np
import pytest

from mdsense import Config
from mdsense.body import sample_scenario
from mdsense.channel import dense_channel
from mdsense.env import run_episode, PolicyInput, PolicyOutput, ProbeRequest, Observation, Ack
from mdsense.keyed_rng import keyed_cnormal, keyed_uniform, STREAM_BUSY

CFG = Config()
SEED = 4242


@pytest.fixture(scope="module")
def H():
    scn = sample_scenario(SEED, CFG.scenario, CFG.radio, force_event=True)
    return dense_channel(scn, CFG.radio, 1.0)


class FixedSlots:
    """Requests a fixed set of slots as early as allowed; records everything it sees."""
    def __init__(self, slots):
        self.todo = sorted(slots)
        self.seen, self.inputs = {}, []

    def step(self, inp):
        self.inputs.append(inp)
        for ob in inp.observations:
            assert ob.t_deliver <= inp.t + 1e-12, "delivered in the future"
            assert ob.t_acquire <= ob.t_deliver
            self.seen[ob.slot] = ob.y
        lead = CFG.env.ctrl_delay_slots + CFG.env.lead_slots
        now = [s for s in self.todo if s < inp.slot + lead + CFG.env.tick_slots and s >= inp.slot + lead]
        self.todo = [s for s in self.todo if s not in now]
        return PolicyOutput([ProbeRequest(s) for s in now])


def free_slots(n, start=20):
    busy = keyed_uniform(SEED, STREAM_BUSY, np.arange(2000)) < CFG.scenario.comm_busy_prob
    return [s for s in range(start, 2000) if not busy[s]][:n]


def test_policy_input_is_isolated(H):
    pol = FixedSlots(free_slots(5))
    run_episode(H, SEED, pol, CFG)
    inp = next(i for i in pol.inputs if i.observations)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.slot = 0
    for f in dataclasses.fields(inp):
        v = getattr(inp, f.name)
        assert isinstance(v, (int, float, tuple))
        if isinstance(v, tuple):
            assert all(isinstance(e, (Observation, Ack)) for e in v)
    y = inp.observations[0].y
    with pytest.raises(ValueError):
        y[0] = 0


def test_common_noise_independent_of_request_history(H):
    s = free_slots(40)
    a, b = FixedSlots(s[:30]), FixedSlots([s[25]] + s[31:])
    run_episode(H, SEED, a, CFG)
    run_episode(H, SEED, b, CFG)
    assert np.array_equal(a.seen[s[25]], b.seen[s[25]])


def test_keyed_noise_is_order_free():
    k = np.arange(32)
    full = keyed_cnormal(1, 777, k)
    assert np.allclose(full[5], keyed_cnormal(1, 777, 5))
    assert np.allclose(full[::-1], keyed_cnormal(1, 777, k[::-1]))
    z = keyed_cnormal(3, np.arange(20000)[:, None], k[None])
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 0.02


def test_no_future_information(H):
    s0 = 800
    H2 = H.copy()
    H2[s0 + 1:] = 0
    slots = free_slots(60, 700)
    a, b = FixedSlots(slots), FixedSlots(slots)
    run_episode(H, SEED, a, CFG)
    run_episode(H2, SEED, b, CFG)
    for s in slots:
        if s <= s0:
            assert np.array_equal(a.seen[s], b.seen[s])


def test_timestamps_and_cost_booking(H):
    busy = keyed_uniform(SEED, STREAM_BUSY, np.arange(2000)) < CFG.scenario.comm_busy_prob
    busy_slot = int(np.flatnonzero(busy[100:])[0] + 100)
    ok = free_slots(10, 300)
    pol = FixedSlots(ok + [busy_slot])
    log = run_episode(H, SEED, pol, CFG)
    applied = {p["slot"] for p in log.probes}
    assert busy_slot not in applied and set(ok) <= applied
    assert any(r["slot"] == busy_slot and r["reason"] == "busy" for r in log.rejects)
    for p in log.probes:
        assert p["t_request"] <= p["t_arrival"] <= p["t_apply"] == p["t_acquire"] < p["t_deliver"]
