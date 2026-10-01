"""Pilot policies. All run on the same contract, DU processor, LLR family, alarm logic,
refractory period and post-alarm service sensing; they differ only in when they ask
for measurements and how they accumulate evidence.

Adapted DE-CuSum (Banerjee & Veeravalli, IEEE T-IT 2013), one "observation" = one
burst look of L probes:
    W >= 0 : take a look,  W <- max(W + LLR, -h)
    W <  0 : skip a look,  W <- min(W + mu, 0)
    alarm when W > A
No optimality is claimed for this adaptation (looks are not i.i.d., looks can be
partially rejected, and the background estimate goes stale while sleeping).
"""
import numpy as np

from .env import PolicyOutput, ProbeRequest
from .features import DUProcessor


class BasePolicy:
    name = "base"

    def __init__(self, pub, du_cfg, det_cfg, llr=None, threshold=np.inf, collect=False):
        self.pub, self.det = pub, det_cfg
        self.du = DUProcessor(pub, du_cfg)
        self.llr, self.A, self.collect = llr, threshold, collect
        self.W = 0.0
        self.records = []                       # collect mode: (x, sample_times)
        self._ready = []
        self.pause_until = -1
        self.gid = 0
        self.first_ok = pub.ctrl_delay_slots + pub.lead_slots
        self.refr_slots = int(round(det_cfg.refractory / pub.slot_duration))
        self.serv_slots = int(round(det_cfg.service_duration / pub.slot_duration))

    # -- hooks
    def _on_ack(self, ack): pass
    def _on_obs(self, ob): pass
    def _update(self, s): raise NotImplementedError
    def _monitor_requests(self, slot): raise NotImplementedError
    def _after_alarm(self): pass

    def step(self, inp):
        out = PolicyOutput()
        for ack in inp.acks:
            if ack.tag == "monitor":
                self._on_ack(ack)
        for ob in inp.observations:
            if ob.tag == "service":
                self.du.track(ob)
            else:
                self.du.ingest(ob)
                self._on_obs(ob)
        ready, self._ready = self._ready, []
        paused = inp.slot < self.pause_until
        for x, ts in ready:
            if self.collect:
                self.records.append((x, ts))
                continue
            if paused:
                continue
            if self._update(self.llr(x)):
                out.alarm = True
                paused = True
                self.W = 0.0
                self.pause_until = inp.slot + self.refr_slots
                self._after_alarm()
                s0 = inp.slot + self.first_ok
                out.requests += [ProbeRequest(s0 + i, "service") for i in range(self.serv_slots)]
        if not paused:
            out.requests += self._monitor_requests(inp.slot)
        return out


class UniformPolicy(BasePolicy):
    name = "uniform"

    def __init__(self, pub, du_cfg, det_cfg, budget, n_window, **kw):
        super().__init__(pub, du_cfg, det_cfg, **kw)
        self.period = max(1, int(round(1.0 / budget)))
        self.nw, self.hop = n_window, max(1, n_window // 2)
        self.next_slot, self.new = 0, 0

    def _on_obs(self, ob):
        self.new += 1
        if self.new >= self.hop:
            self.new = 0
            st = self.du.statistic(self.nw)
            if st is not None:
                self._ready.append(st)

    def _update(self, s):
        self.W = max(0.0, self.W + s)
        return self.W > self.A

    def _monitor_requests(self, slot):
        lo, hi = slot + self.first_ok, slot + self.first_ok + self.pub.tick_slots
        if self.next_slot < lo:
            self.next_slot = int(np.ceil(lo / self.period)) * self.period
        reqs = []
        while self.next_slot < hi:
            reqs.append(ProbeRequest(self.next_slot))
            self.next_slot += self.period
        return reqs


class _LookPolicy(BasePolicy):
    """Measurements come in looks of L probes spaced `spacing` slots apart (span sets the
    Doppler resolution, spacing sets the Nyquist band); one statistic per completed look."""

    def __init__(self, pub, du_cfg, det_cfg, burst_len, spacing=1, **kw):
        super().__init__(pub, du_cfg, det_cfg, **kw)
        self.L, self.d = burst_len, spacing
        self.span = burst_len * spacing
        self.looks = {}                         # gid -> [acked, accepted, delivered]

    def _new_look(self, start):
        g = self.gid
        self.gid += 1
        self.looks[g] = [0, 0, 0]
        return g, [ProbeRequest(start + i * self.d, "monitor", g) for i in range(self.L)]

    def _check(self, g):
        a = self.looks.get(g)
        if a is None or a[0] < self.L or a[2] < a[1]:
            return
        del self.looks[g]
        if a[1] >= 1:
            # Last accepted residuals of this look; busy/dropped slots are omitted, not filled.
            st = self.du.statistic(a[1])
            if st is not None:
                self._ready.append(st)
        self._look_done(g)

    def _look_done(self, g): pass

    def _on_ack(self, ack):
        if ack.group in self.looks:
            self.looks[ack.group][0] += 1
            self.looks[ack.group][1] += int(ack.accepted)
            self._check(ack.group)

    def _on_obs(self, ob):
        if ob.group in self.looks:
            self.looks[ob.group][2] += 1
            self._check(ob.group)


class BurstPolicy(_LookPolicy):
    name = "burst"

    def __init__(self, pub, du_cfg, det_cfg, budget, burst_len, spacing=1, **kw):
        super().__init__(pub, du_cfg, det_cfg, burst_len, spacing, **kw)
        self.period = max(self.span, int(round(burst_len / budget)))
        self.next_start = 0

    def _update(self, s):
        self.W = max(0.0, self.W + s)
        return self.W > self.A

    def _monitor_requests(self, slot):
        lo, hi = slot + self.first_ok, slot + self.first_ok + self.pub.tick_slots
        if self.next_start < lo:
            self.next_start = int(np.ceil(lo / self.period)) * self.period
        reqs = []
        while self.next_start < hi:
            reqs += self._new_look(self.next_start)[1]
            self.next_start += self.period
        return reqs


class TriggerPolicy(_LookPolicy):
    """Two-rate schedule (post-hoc baseline): one look every p_slow slots; a look with
    llr > tau switches to back-to-back looks for n_hold looks (re-armed by further
    llr > tau), then back to slow. Same CuSum and alarm logic as the burst."""
    name = "trigger"

    def __init__(self, pub, du_cfg, det_cfg, burst_len, p_slow, tau, n_hold, spacing=1, **kw):
        super().__init__(pub, du_cfg, det_cfg, burst_len, spacing, **kw)
        self.p_slow = max(self.span, int(round(p_slow)))
        self.tau, self.n_hold = tau, n_hold
        self.fast_left = 0                      # fast looks still to schedule
        self.next_start = 0
        self.last_end = 0                       # end of the last scheduled look

    def _update(self, s):
        self.W = max(0.0, self.W + s)
        if s > self.tau:
            self.fast_left = self.n_hold
            self.next_start = min(self.next_start, self.last_end)
        return self.W > self.A

    def _after_alarm(self):
        self.fast_left = 0

    def _monitor_requests(self, slot):
        lo, hi = slot + self.first_ok, slot + self.first_ok + self.pub.tick_slots
        self.next_start = max(self.next_start, lo, self.last_end)
        reqs = []
        while self.next_start < hi:
            reqs += self._new_look(self.next_start)[1]
            self.last_end = self.next_start + self.span
            if self.fast_left > 0:
                self.fast_left -= 1
            self.next_start += self.span if self.fast_left > 0 else self.p_slow
        return reqs


class DECuSumPolicy(_LookPolicy):
    name = "decusum"

    def __init__(self, pub, du_cfg, det_cfg, burst_len, mu, h, spacing=1, epoch_slots=None, **kw):
        super().__init__(pub, du_cfg, det_cfg, burst_len, spacing, **kw)
        self.mu, self.h = mu, h
        self.E = epoch_slots or self.span      # awake: back-to-back looks, so any budget is reachable
        self.next_epoch, self.in_flight = 0, None

    def _update(self, s):
        self.W = max(self.W + s, -self.h)
        return self.W > self.A

    def _look_done(self, g):
        if g == self.in_flight:
            self.in_flight = None

    def _after_alarm(self):
        self.in_flight = None

    def _monitor_requests(self, slot):
        if self.in_flight is not None or slot < self.next_epoch:
            return []
        self.next_epoch = slot + self.E
        if self.W >= 0.0:
            g, reqs = self._new_look(slot + self.first_ok)
            self.in_flight = g
            return reqs
        self.W = min(self.W + self.mu, 0.0)
        return []
