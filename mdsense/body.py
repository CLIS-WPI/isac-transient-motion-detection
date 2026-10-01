"""Articulated human made of 12 ellipsoidal segments.

Labels come from the motion scenario itself (the event dict), never from a detector.
A target event is a fast limb motion as defined in ScenarioConfig. Nuisance motions
(slow arm adjustment, weight shift, head turn) and breathing/sway/jitter belong to
the negative class.
"""
from dataclasses import dataclass
import numpy as np

SEGMENTS = ["torso", "head",
            "uarm_L", "farm_L", "hand_L", "uarm_R", "farm_R", "hand_R",
            "thigh_L", "shin_L", "thigh_R", "shin_R"]
# (radius, half-length) of each ellipsoid for a 1.75 m person, scaled with height
_DIMS = {"torso": (0.15, 0.30), "head": (0.09, 0.12),
         "uarm": (0.045, 0.15), "farm": (0.035, 0.135), "hand": (0.03, 0.09),
         "thigh": (0.07, 0.225), "shin": (0.05, 0.225)}
L_UARM, L_FARM, L_HAND, L_THIGH, L_SHIN = 0.30, 0.27, 0.18, 0.45, 0.45
JOINTS = ["shf_L", "sha_L", "elb_L", "shf_R", "sha_R", "elb_R", "hip_L", "knee_L", "hip_R", "knee_R"]


@dataclass
class Scenario:
    seed: int
    scale: float
    ground: np.ndarray          # (x, y, 0) under the pelvis
    yaw: float
    breathing: tuple            # (amp, freq, phase)
    sway: np.ndarray            # [2 axes, 2 comps, (amp, freq, phase)]
    rest: dict                  # joint -> rest angle [rad]
    jitter: dict                # joint -> [3, (amp, freq, phase)]
    nuisances: list
    event: dict | None
    clutter_pos: np.ndarray     # [Q, 3]
    clutter_rcs: np.ndarray     # [Q]
    drift: np.ndarray           # [2 (gain, phase), 3 comps, (amp, freq, phase)]


def _u(rng, lo_hi):
    return rng.uniform(*lo_hi)


def sample_scenario(seed, scfg, radio, force_event=None):
    rng = np.random.default_rng(seed)
    scale = _u(rng, scfg.height) / 1.75
    r_xy = _u(rng, scfg.range_xy)
    az = np.deg2rad(_u(rng, scfg.azimuth_deg))
    ground = np.array([r_xy * np.cos(az), r_xy * np.sin(az), 0.0])
    yaw = az + np.pi + np.deg2rad(_u(rng, scfg.facing_deg))   # body +x axis toward radar when facing=0

    breathing = (_u(rng, scfg.breathing_amp), _u(rng, scfg.breathing_freq), rng.uniform(0, 2 * np.pi))
    sway = np.stack([np.stack([[_u(rng, scfg.sway_amp) / np.sqrt(2), rng.uniform(0.08, 0.4),
                                rng.uniform(0, 2 * np.pi)] for _ in range(2)]) for _ in range(2)])
    rest = {"shf_L": np.deg2rad(rng.uniform(0, 10)), "shf_R": np.deg2rad(rng.uniform(0, 10)),
            "sha_L": np.deg2rad(rng.uniform(3, 10)), "sha_R": np.deg2rad(rng.uniform(3, 10)),
            "elb_L": np.deg2rad(rng.uniform(5, 20)), "elb_R": np.deg2rad(rng.uniform(5, 20)),
            "hip_L": 0.0, "hip_R": 0.0,
            "knee_L": np.deg2rad(rng.uniform(0, 5)), "knee_R": np.deg2rad(rng.uniform(0, 5))}
    ja = np.deg2rad(scfg.jitter_amp_deg) / np.sqrt(3)
    jitter = {j: np.stack([[ja * rng.uniform(0.5, 1.5), rng.uniform(0.1, 1.5), rng.uniform(0, 2 * np.pi)]
                           for _ in range(3)]) for j in JOINTS}

    t_lo, t_hi = scfg.warmup + 0.5, scfg.duration - 0.2
    nuisances = []
    for _ in range(rng.poisson(scfg.nuisance_rate)):
        kind = rng.choice(["arm_adjust", "weight_shift", "head_turn"])
        D = rng.uniform(1.5, 3.0) if kind == "arm_adjust" else rng.uniform(0.6, 1.2)
        nuisances.append({"kind": str(kind), "t0": rng.uniform(scfg.warmup, t_hi - D / 2), "D": D,
                          "side": str(rng.choice(["L", "R"])),
                          "amp": (np.deg2rad(rng.uniform(15, 30)) if kind == "arm_adjust"
                                  else rng.uniform(0.03, 0.06) if kind == "weight_shift"
                                  else rng.uniform(0.02, 0.04))})

    has_event = (rng.uniform() < scfg.p_event) if force_event is None else force_event
    event = None
    if has_event:
        D = _u(rng, scfg.event_duration)
        kind = "arm" if rng.uniform() < scfg.p_arm_event else "leg"
        n = int(rng.integers(1, 3)) if kind == "arm" else 1
        v_target = _u(rng, scfg.event_peak_speed)
        lever = scale * ((L_UARM + L_FARM + L_HAND / 2) if kind == "arm" else (L_THIGH + L_SHIN))
        amp = v_target * D / (np.pi * n * lever)
        amp = min(amp, np.deg2rad(120 if kind == "arm" else 60))
        event = {"kind": kind, "side": str(rng.choice(["L", "R"])), "t0": rng.uniform(t_lo, t_hi - D),
                 "D": D, "n": n, "amp": amp, "peak_speed": amp * np.pi * n * lever / D}

    q = scfg.num_clutter
    cr = rng.uniform(*scfg.clutter_range, q)
    ca = rng.uniform(-np.pi / 2, np.pi / 2, q)
    clutter_pos = np.stack([cr * np.cos(ca), cr * np.sin(ca), rng.uniform(0.0, 2.5, q)], 1)
    clutter_rcs = rng.uniform(*scfg.clutter_rcs, q)
    drift = np.stack([np.stack([[amp / np.sqrt(3), rng.uniform(0.05, 1.0), rng.uniform(0, 2 * np.pi)]
                                for _ in range(3)]) for amp in (scfg.drift_amp, scfg.drift_phase)])
    return Scenario(seed, scale, ground, yaw, breathing, sway, rest, jitter, nuisances, event,
                    clutter_pos, clutter_rcs, drift)


# ---------------------------------------------------------------- motion primitives
def _sines(t, comps):
    return sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in comps)


def _bump(t, t0, D, n=1):
    """Out-and-back raised cosine with zero velocity at both ends."""
    x = np.clip((t - t0) / D, 0.0, 1.0)
    return 0.5 * (1 - np.cos(2 * np.pi * n * x))


def _step(t, t0, D):
    x = np.clip((t - t0) / D, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * x))


def joint_angles(scn, t):
    ang = {j: scn.rest[j] + _sines(t, scn.jitter[j]) for j in JOINTS}
    for nu in scn.nuisances:
        if nu["kind"] == "arm_adjust":
            ang["shf_" + nu["side"]] = ang["shf_" + nu["side"]] + nu["amp"] * _bump(t, nu["t0"], nu["D"])
    ev = scn.event
    if ev is not None:
        b = ev["amp"] * _bump(t, ev["t0"], ev["D"], ev["n"])
        s = ev["side"]
        if ev["kind"] == "arm":
            ang["shf_" + s] = ang["shf_" + s] + b
            ang["elb_" + s] = ang["elb_" + s] + 0.4 * b
        else:
            ang["hip_" + s] = ang["hip_" + s] + b
            ang["knee_" + s] = ang["knee_" + s] + 0.6 * b
    return ang


def _sagittal(theta, abd, side):
    """Unit vector: pointing down, flexed forward by theta, abducted outward by abd."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = (1.0 if side == "L" else -1.0) * np.asarray(abd, dtype=np.float64) + 0.0 * theta
    return np.stack([np.sin(theta), np.cos(theta) * np.sin(phi), -np.cos(theta) * np.cos(phi)], -1)


def segment_states(scn, t):
    """Centers [T,S,3], unit axes [T,S,3] in world coordinates, and dims [S,2] (radius, half-length)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    s = scn.scale
    ang = joint_angles(scn, t)
    T = t.shape[0]
    hip_h = 0.92 * s
    zeros, ones = np.zeros(T), np.ones(T)
    ez = np.stack([zeros, zeros, ones], -1)

    centers, axes, dims = [], [], []

    def add(name, c, a):
        centers.append(c)
        axes.append(a / np.linalg.norm(a, axis=-1, keepdims=True))
        dims.append(np.array(_DIMS[name.split("_")[0]]) * s)

    b_amp, b_f, b_p = scn.breathing
    torso_c = np.stack([b_amp * np.sin(2 * np.pi * b_f * t + b_p), zeros, zeros + hip_h + 0.30 * s], -1)
    add("torso", torso_c, ez)
    head_c = np.stack([zeros, zeros, zeros + hip_h + 0.73 * s], -1)
    for nu in scn.nuisances:
        if nu["kind"] == "head_turn":
            head_c = head_c + np.stack([zeros, nu["amp"] * _bump(t, nu["t0"], nu["D"]), zeros], -1)
    add("head", head_c, ez)

    for side, sg in (("L", 1.0), ("R", -1.0)):
        sh = np.array([0.0, sg * 0.19 * s, hip_h + 0.53 * s])
        d_u = _sagittal(ang["shf_" + side], ang["sha_" + side], side)
        d_f = _sagittal(ang["shf_" + side] + ang["elb_" + side], ang["sha_" + side], side)
        elbow = sh + L_UARM * s * d_u
        wrist = elbow + L_FARM * s * d_f
        add("uarm_" + side, sh + 0.5 * L_UARM * s * d_u, d_u)
        add("farm_" + side, elbow + 0.5 * L_FARM * s * d_f, d_f)
        add("hand_" + side, wrist + 0.5 * L_HAND * s * d_f, d_f)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        hip = np.array([0.0, sg * 0.09 * s, hip_h])
        d_t = _sagittal(ang["hip_" + side], 0.0, side)
        d_s = _sagittal(ang["hip_" + side] - ang["knee_" + side], 0.0, side)
        knee = hip + L_THIGH * s * d_t
        add("thigh_" + side, hip + 0.5 * L_THIGH * s * d_t, d_t)
        add("shin_" + side, knee + 0.5 * L_SHIN * s * d_s, d_s)

    C = np.stack(centers, 1)
    A = np.stack(axes, 1)
    # whole-body translation: sway + weight shifts (body frame x, y)
    off = np.zeros((T, 3))
    for ax in range(2):
        off[:, ax] = _sines(t, scn.sway[ax])
    for nu in scn.nuisances:
        if nu["kind"] == "weight_shift":
            sg = 1.0 if nu["side"] == "L" else -1.0
            off[:, 1] += sg * nu["amp"] * _step(t, nu["t0"], nu["D"])
    C = C + off[:, None, :]
    cy, sy = np.cos(scn.yaw), np.sin(scn.yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    C = C @ Rz.T + scn.ground
    A = A @ Rz.T
    return C, A, np.stack(dims)


def segment_velocities(scn, t, dt=1e-4):
    Cp, _, _ = segment_states(scn, np.asarray(t) + dt)
    Cm, _, _ = segment_states(scn, np.asarray(t) - dt)
    return (Cp - Cm) / (2 * dt)


def ellipsoid_rcs(k_hat, axes, dims):
    """Monostatic RCS of prolate ellipsoids [m^2]; k_hat, axes: [...,S,3], dims: [S,2]."""
    r, c = dims[:, 0], dims[:, 1]
    cos2 = np.clip(np.sum(k_hat * axes, -1) ** 2, 0.0, 1.0)
    sin2 = 1.0 - cos2
    return np.pi * r ** 4 * c ** 2 / (r ** 2 * sin2 + c ** 2 * cos2) ** 2


def common_drift(scn, t):
    """Slow common complex gain g(t) = (1 + eps) exp(j phi): makes clutter removal imperfect."""
    eps = _sines(t, scn.drift[0])
    phi = _sines(t, scn.drift[1])
    return (1.0 + eps) * np.exp(1j * phi)
