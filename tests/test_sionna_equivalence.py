"""Requires sionna-rt >= 2.2 (skipped otherwise). Slow-ish (~1 min on one CPU core)."""
import numpy as np
import pytest

sionna = pytest.importorskip("sionna.rt.rcs")

from mdsense import Config
from mdsense.body import sample_scenario
from mdsense.channel import SionnaBackend, dense_channel

CFG = Config()


def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_snapshot_stitching_is_phase_continuous():
    """Constant radial velocity: snapshots of 8 slots + Doppler vs one solve per slot."""
    r = CFG.radio
    be = SionnaBackend(r, snapshot_slots=8)
    scene, targets = be._scene(1)
    radar = np.asarray(r.radar_position)
    p0, v = radar + np.array([0.0, 6.0, -1.0]), np.array([0.0, -3.0, 0.0])
    n = 160
    per_slot = np.vstack([be.solve_points(scene, targets, [p0 + v * k * r.slot_duration], [v], [1.0], 1)
                          for k in range(n)])
    stitched = np.vstack([be.solve_points(scene, targets, [p0 + v * k * r.slot_duration], [v], [1.0], 8)
                          for k in range(0, n, 8)])
    print(f"\n[stitching] corr={_corr(per_slot, stitched):.6f}")
    assert _corr(per_slot, stitched) > 0.999
    f = np.fft.fftfreq(n, r.slot_duration)
    peak = lambda h: f[np.argmax(np.abs(np.fft.fft(h[:, 0] * np.hanning(n))))]
    assert abs(peak(per_slot) - peak(stitched)) < 1e-9
    assert abs(peak(stitched) - 2 * 3.0 * np.dot(-v / 3.0, (p0 - radar) / np.linalg.norm(p0 - radar))
               / r.wavelength) < 2 / (n * r.slot_duration)


def test_analytic_backend_matches_rcssolver():
    """Articulated body with an arm event, 0.3 s around the event, clutter included."""
    scn = sample_scenario(102, CFG.scenario, CFG.radio, force_event=True)
    t0 = scn.event["t0"]
    dur = t0 + 0.3
    Ha = dense_channel(scn, CFG.radio, dur, backend="analytic")
    n0 = int(t0 / CFG.radio.slot_duration)
    # Sionna is run on the event window only (solving from t = 0 would be slow)
    be = SionnaBackend(CFG.radio, snapshot_slots=4)
    from mdsense import body
    Hs = []
    scene, targets = be._scene(len(body.SEGMENTS) + scn.clutter_pos.shape[0])
    radar = np.asarray(CFG.radio.radar_position)
    for n in range(n0, Ha.shape[0], 4):
        t = n * CFG.radio.slot_duration
        Cn, Ax, dims = body.segment_states(scn, [t])
        V = body.segment_velocities(scn, [t])
        k_hat = (Cn[0] - radar) / np.linalg.norm(Cn[0] - radar, axis=-1, keepdims=True)
        sig = np.concatenate([body.ellipsoid_rcs(k_hat, Ax[0], dims), scn.clutter_rcs])
        pos = np.vstack([Cn[0], scn.clutter_pos])
        vel = np.vstack([V[0], np.zeros_like(scn.clutter_pos)])
        steps = min(4, Ha.shape[0] - n)
        Hs.append(be.solve_points(scene, targets, pos, vel, sig, steps))
    Hs = np.vstack(Hs) * body.common_drift(scn, np.arange(n0, Ha.shape[0]) * CFG.radio.slot_duration)[:, None]
    Ha_w = Ha[n0:]
    # compare the dynamic (human) part: remove the static mean so clutter does not dominate
    da, ds = Ha_w - Ha_w.mean(0), Hs - Hs.mean(0)
    print(f"\n[equivalence] corr(dynamic)={_corr(da, ds):.5f}  power ratio={np.sum(np.abs(Hs) ** 2) / np.sum(np.abs(Ha_w) ** 2):.5f}")
    assert _corr(da, ds) > 0.98
    assert abs(np.sum(np.abs(Hs) ** 2) / np.sum(np.abs(Ha_w) ** 2) - 1) < 0.02
