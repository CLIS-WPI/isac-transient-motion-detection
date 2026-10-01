"""Physical truth: the noiseless monostatic channel H[slot, RE] on the dense slot grid.

Two backends produce the same quantity:
- "analytic": exact radar equation per point scatterer, evaluated at every slot. In an
  empty scene with max_depth=1 and constant RCS this is exactly what Sionna's
  RCSSolver computes (validated in tests/test_sionna_equivalence.py), and it is
  ~1000x faster, so it is used for the Monte Carlo pilot.
- "sionna": Sionna RT 2.2 RCSSolver, one solve per snapshot, Doppler inside a snapshot
  from per-target velocities. Needed once multipath / room geometry is added.

Both apply the same common drift g(t) (imperfect clutter removal). Noise is NOT added
here; the environment adds keyed noise only to acquired measurements.
"""
import os
import numpy as np
from scipy.constants import speed_of_light as C

from .body import segment_states, segment_velocities, ellipsoid_rcs, common_drift


def select_mitsuba_variant(mi):
    """Prefer CUDA (GPU1 appears as device 0 when NVIDIA_VISIBLE_DEVICES=1), else LLVM."""
    variants = list(mi.variants())
    cuda_wanted = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() != ""
    order = []
    if cuda_wanted:
        order.append("cuda_ad_mono_polarized")
        order.extend(v for v in variants if v.startswith("cuda_") and v not in order)
    order.append("llvm_ad_mono_polarized")
    order.extend(v for v in variants if v not in order)
    last_err = None
    for name in order:
        if name not in variants:
            continue
        try:
            mi.set_variant(name)
            return mi.variant()
        except Exception as exc:  # GPU listed but not usable (no driver, bad fork, ...)
            last_err = exc
    if last_err is not None and not variants:
        raise last_err
    return mi.variant() if hasattr(mi, "variant") else None


def _point_response(pos, sigma, radio, radar):
    """pos: [..., P, 3], sigma: [..., P] -> H [..., K]."""
    d = pos - radar
    R = np.linalg.norm(d, axis=-1)
    amp = np.sqrt(sigma) * radio.wavelength / ((4 * np.pi) ** 1.5 * R ** 2)
    tau = 2 * R / C
    f = radio.fc + radio.re_offsets
    return np.einsum("...p,...pk->...k", amp, np.exp(-2j * np.pi * tau[..., None] * f))


def human_channel_analytic(scn, radio, t):
    radar = np.asarray(radio.radar_position)
    Cn, Ax, dims = segment_states(scn, t)
    k_hat = (Cn - radar) / np.linalg.norm(Cn - radar, axis=-1, keepdims=True)
    sigma = ellipsoid_rcs(k_hat, Ax, dims)
    return _point_response(Cn, sigma, radio, radar)


def clutter_channel_analytic(scn, radio):
    return _point_response(scn.clutter_pos, scn.clutter_rcs, radio, np.asarray(radio.radar_position))


def dense_channel(scn, radio, duration, backend="analytic", snapshot_slots=4, include_clutter=True):
    n_slots = int(round(duration / radio.slot_duration))
    t = np.arange(n_slots) * radio.slot_duration
    if backend == "analytic":
        H = np.zeros((n_slots, radio.num_re), np.complex128)
        for i0 in range(0, n_slots, 2000):              # chunk to bound memory
            H[i0:i0 + 2000] = human_channel_analytic(scn, radio, t[i0:i0 + 2000])
        if include_clutter:
            H += clutter_channel_analytic(scn, radio)[None]
    elif backend == "sionna":
        H = SionnaBackend(radio, snapshot_slots).dense(scn, n_slots, include_clutter)
    else:
        raise ValueError(backend)
    return (H * common_drift(scn, t)[:, None]).astype(np.complex64)


class SionnaBackend:
    """Each body segment and clutter point is a ConstantRCSSensingTarget (2 cm box, so
    that artificial inter-segment shadowing is negligible). Per snapshot: set position,
    velocity and sigma (ellipsoid RCS for the current aspect), solve, expand in time
    with the Doppler shifts, convert to the sensing REs."""

    def __init__(self, radio, snapshot_slots=4):
        import mitsuba as mi
        self.variant = select_mitsuba_variant(mi)
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
        from sionna.rt.rcs import RCSSolver, ConstantRCSSensingTarget
        self._mods = (load_scene, Transmitter, Receiver, PlanarArray, ConstantRCSSensingTarget)
        self.solver = RCSSolver(deterministic=True)
        self.radio, self.S = radio, snapshot_slots

    def _scene(self, n_targets):
        load_scene, Transmitter, Receiver, PlanarArray, CT = self._mods
        scene = load_scene()
        scene.frequency = self.radio.fc
        pos = list(self.radio.radar_position)
        scene.add(Transmitter("tx", position=pos))
        scene.add(Receiver("rx", position=pos))
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V", pattern="iso")
        scene.rx_array = scene.tx_array
        targets = [CT(f"st{i}", sigma=1.0, length=0.02, width=0.02, height=0.02,
                      position=(10.0 + i, 10.0, 1.0)) for i in range(n_targets)]
        scene.add(targets)
        return scene, targets

    def solve_points(self, scene, targets, pos, vel, sigma, num_steps):
        import mitsuba as mi
        for tg, p, v, s in zip(targets, pos, vel, sigma):
            tg.position = mi.Point3f(*map(float, p))
            tg.velocity = mi.Vector3f(*map(float, v))
            tg.sigma = float(s)
        paths = self.solver(scene, max_depth=1, seed=0)
        a, tau = paths.cir(sampling_frequency=1.0 / self.radio.slot_duration, num_time_steps=num_steps,
                           normalize_delays=False, out_type="numpy")
        a = np.asarray(a).reshape(-1, num_steps)          # [paths, steps]
        tau = np.asarray(tau).reshape(-1)                 # [paths]
        # Sionna's coefficients already carry exp(-j 2 pi fc tau); add the baseband offsets.
        return a.T @ np.exp(-2j * np.pi * tau[:, None] * self.radio.re_offsets[None])

    def dense(self, scn, n_slots, include_clutter=True):
        radar = np.asarray(self.radio.radar_position)
        q = scn.clutter_pos.shape[0] if include_clutter else 0
        n_seg = segment_states(scn, [0.0])[0].shape[1]
        scene, targets = self._scene(n_seg + q)
        H = np.zeros((n_slots, self.radio.num_re), np.complex128)
        for n0 in range(0, n_slots, self.S):
            steps = min(self.S, n_slots - n0)
            t0 = n0 * self.radio.slot_duration
            Cn, Ax, dims = segment_states(scn, [t0])
            V = segment_velocities(scn, [t0])
            k_hat = (Cn[0] - radar) / np.linalg.norm(Cn[0] - radar, axis=-1, keepdims=True)
            sig = ellipsoid_rcs(k_hat, Ax[0], dims)
            pos, vel = Cn[0], V[0]
            if q:
                pos = np.vstack([pos, scn.clutter_pos])
                vel = np.vstack([vel, np.zeros((q, 3))])
                sig = np.concatenate([sig, scn.clutter_rcs])
            H[n0:n0 + steps] = self.solve_points(scene, targets, pos, vel, sig, steps)
        return H


def tr38901_human_channel(scn, radio, duration, snapshot_slots=4, model_type=1):
    """Sionna TR 38.901 'human' (single scattering point) placed at the torso and moved
    with the torso velocity. Used to show that the standard model has no limb micro-Doppler."""
    import mitsuba as mi
    from sionna.rt.rcs import TR38901SensingTarget
    be = SionnaBackend(radio, snapshot_slots)
    load_scene, Transmitter, Receiver, PlanarArray, _ = be._mods
    scene = load_scene()
    scene.frequency = radio.fc
    scene.add(Transmitter("tx", position=list(radio.radar_position)))
    scene.add(Receiver("rx", position=list(radio.radar_position)))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V", pattern="iso")
    scene.rx_array = scene.tx_array
    tg = TR38901SensingTarget("human", object_type="human", model_type=model_type,
                              position=(10.0, 10.0, 1.0))
    scene.add(tg)
    n_slots = int(round(duration / radio.slot_duration))
    H = np.zeros((n_slots, radio.num_re), np.complex128)
    for n0 in range(0, n_slots, snapshot_slots):
        steps = min(snapshot_slots, n_slots - n0)
        t0 = n0 * radio.slot_duration
        Cn, _, _ = segment_states(scn, [t0])
        V = segment_velocities(scn, [t0])
        tg.position = mi.Point3f(*map(float, Cn[0, 0]))
        tg.velocity = mi.Vector3f(*map(float, V[0, 0]))
        tg.orientation = mi.Point3f(float(scn.yaw), 0.0, 0.0)
        paths = be.solver(scene, max_depth=1, seed=0)
        a, tau = paths.cir(sampling_frequency=1.0 / radio.slot_duration, num_time_steps=steps,
                           normalize_delays=False, out_type="numpy")
        a = np.asarray(a).reshape(-1, steps)
        tau = np.asarray(tau).reshape(-1)
        H[n0:n0 + steps] = a.T @ np.exp(-2j * np.pi * tau[:, None] * radio.re_offsets[None])
    return H
