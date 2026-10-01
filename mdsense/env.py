"""The observation contract.

run_episode() owns the environment; the policy never receives a reference to it.
Each tick the policy gets a frozen PolicyInput (current time, newly delivered
observations, acks) and returns a PolicyOutput (probe requests, alarm flag).
Observation arrays are read-only copies. The dense channel, the noise seed and the
event labels never cross this boundary.

Timeline of one probe (all recorded):
  t_request  policy emits the request (tick time)
  t_arrival  request reaches the DU scheduler (t_request + ctrl_delay); accept/reject here
  t_apply    slot in which the probe symbol is transmitted (cost is booked here)
  t_acquire  echo measured (same slot for a monostatic radar)
  t_deliver  measurement handed to the policy (t_acquire + proc_delay)
"""
from dataclasses import dataclass, field
import heapq
import numpy as np

from .keyed_rng import keyed_cnormal, keyed_uniform, STREAM_BUSY


def episode_noise(seed, n_slots, num_re, rx=0):
    """Unit-variance keyed noise for every (slot, RE); same values as per-probe draws."""
    return keyed_cnormal(seed, np.arange(n_slots)[:, None], np.arange(num_re)[None, :], rx)


@dataclass(frozen=True)
class PublicInfo:
    """What a DU-side policy is allowed to know at construction."""
    slot_duration: float
    num_re: int
    bandwidth: float
    noise_var: float
    tick_slots: int
    ctrl_delay_slots: int
    lead_slots: int
    proc_delay_slots: int


@dataclass(frozen=True)
class ProbeRequest:
    target_slot: int
    tag: str = "monitor"            # "monitor" or "service"
    group: int = -1                 # policy-defined look id


@dataclass(frozen=True)
class Ack:
    target_slot: int
    tag: str
    group: int
    accepted: bool
    reason: str
    slot: int                       # when the ack reaches the policy


@dataclass(frozen=True)
class Observation:
    slot: int                       # acquisition slot
    t_acquire: float
    t_deliver: float
    tag: str
    group: int
    y: np.ndarray                   # read-only complex [K]


@dataclass(frozen=True)
class PolicyInput:
    slot: int
    t: float
    observations: tuple
    acks: tuple


@dataclass
class PolicyOutput:
    requests: list = field(default_factory=list)
    alarm: bool = False


@dataclass
class EpisodeLog:
    alarm_times: list
    probes: list                    # dicts with the five timestamps, tag, group
    rejects: list
    n_slots: int
    slot_duration: float

    def applied_slots(self, tag=None, t_from=0.0):
        return {p["slot"] for p in self.probes
                if (tag is None or tag in p["tags"]) and p["slot"] * self.slot_duration >= t_from}


def public_info(cfg):
    r, e = cfg.radio, cfg.env
    return PublicInfo(r.slot_duration, r.num_re, r.bandwidth, r.noise_var,
                      e.tick_slots, e.ctrl_delay_slots, e.lead_slots, e.proc_delay_slots)


def run_episode(H, seed, policy, cfg, busy=None, noise=None):
    """H: dense noiseless channel [n_slots, K] (never passed to the policy).

    `busy` and `noise` are optional caches. If omitted they are keyed from `seed`
    (same values as a cache hit). Precomputation does not change causality: both
    are functions of (seed, slot, RE), not of the policy's request history.
    """
    r, e = cfg.radio, cfg.env
    n_slots, K = H.shape
    Ts = r.slot_duration
    if busy is None:
        busy = keyed_uniform(seed, STREAM_BUSY, np.arange(n_slots)) < cfg.scenario.comm_busy_prob
    if noise is None:
        noise = episode_noise(seed, n_slots, K)
    sigma_n = np.sqrt(r.noise_var)

    in_transit = []                 # heap of (arrival_slot, seq, request, t_request)
    booked = {}                     # slot -> dict(tags=set, groups=list, t_request, t_arrival)
    to_apply = []                   # heap of booked slots
    deliveries = []                 # heap of (deliver_slot, seq, Observation)
    acks = []                       # heap of (ack_slot, seq, Ack)
    seq = 0
    log = EpisodeLog([], [], [], n_slots, Ts)

    for s in range(0, n_slots, e.tick_slots):
        # 1) scheduler: requests arriving by now, in arrival order
        while in_transit and in_transit[0][0] <= s:
            arr, _, req, t_req = heapq.heappop(in_transit)
            ts = req.target_slot
            if ts < arr + e.lead_slots:
                ok, why = False, "too_late"
            elif ts >= n_slots:
                ok, why = False, "out_of_episode"
            elif busy[ts] and ts not in booked:
                ok, why = False, "busy"
            else:
                ok, why = True, "ok"
                if ts not in booked:
                    booked[ts] = {"tags": set(), "groups": [], "t_request": t_req, "t_arrival": arr * Ts}
                    heapq.heappush(to_apply, ts)
                booked[ts]["tags"].add(req.tag)
                booked[ts]["groups"].append((req.tag, req.group))
            ack_slot = arr + e.ctrl_delay_slots
            heapq.heappush(acks, (ack_slot, seq, Ack(ts, req.tag, req.group, ok, why, ack_slot)))
            seq += 1
            if not ok:
                log.rejects.append({"slot": ts, "reason": why, "tag": req.tag})
        # 2) transmit / acquire booked slots up to now
        while to_apply and to_apply[0] <= s:
            ts = heapq.heappop(to_apply)
            b = booked.pop(ts)
            y = np.asarray(H[ts] + sigma_n * noise[ts], np.complex64)
            y.setflags(write=False)
            d_slot = ts + e.proc_delay_slots
            log.probes.append({"slot": ts, "tags": sorted(b["tags"]), "t_request": b["t_request"],
                               "t_arrival": b["t_arrival"], "t_apply": ts * Ts, "t_acquire": ts * Ts,
                               "t_deliver": d_slot * Ts})
            for tag, grp in b["groups"]:
                heapq.heappush(deliveries, (d_slot, seq, Observation(ts, ts * Ts, d_slot * Ts, tag, grp, y)))
                seq += 1
        # 3) hand over what has arrived
        obs = []
        while deliveries and deliveries[0][0] <= s:
            obs.append(heapq.heappop(deliveries)[2])
        aks = []
        while acks and acks[0][0] <= s:
            aks.append(heapq.heappop(acks)[2])
        out = policy.step(PolicyInput(s, s * Ts, tuple(obs), tuple(aks)))
        if out.alarm:
            log.alarm_times.append(s * Ts)
        for req in out.requests:
            heapq.heappush(in_transit, (s + e.ctrl_delay_slots, seq, req, s * Ts))
            seq += 1
    return log
