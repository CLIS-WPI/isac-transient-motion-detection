"""Motivating figure: the TR 38.901 'human' (one scattering point, rigid translation) has
no limb micro-Doppler; the articulated 12-segment body does. Requires sionna-rt >= 2.2.

    PYTHONPATH=. python scripts/fig_3gpp_vs_articulated.py --out figures
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

from mdsense import Config
from mdsense.body import sample_scenario
from mdsense.channel import dense_channel, tr38901_human_channel


def spectrogram(H, r, rng):
    w = np.hanning(r.num_re + 2)[1:-1]
    Z = np.fft.ifft(H * w, axis=1)
    b = int(round(rng / r.range_bin))
    x = Z[:, max(b - 1, 0):b + 2]
    x = x - x.mean(0)
    f, t, S = stft(x.T, fs=1 / r.slot_duration, nperseg=256, noverlap=224,
                   return_onesided=False, axis=-1)
    P = np.fft.fftshift((np.abs(S) ** 2).sum(0), axes=0)
    return t, np.fft.fftshift(f) * r.wavelength / 2, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed", type=int, default=102)
    args = ap.parse_args()
    cfg = Config()
    r = cfg.radio
    scn = sample_scenario(args.seed, cfg.scenario, r, force_event=True)
    ev = scn.event
    dur = min(cfg.scenario.duration, ev["t0"] + ev["D"] + 0.7)
    rng = np.linalg.norm(np.array([*scn.ground[:2], 1.3]) - np.asarray(r.radar_position))
    H_art = dense_channel(scn, r, dur, include_clutter=False)
    H_3gpp = tr38901_human_channel(scn, r, dur)
    noise = np.sqrt(r.noise_var / 2) * np.random.default_rng(0).standard_normal((2,) + H_art.shape)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, H, title in ((axs[0], H_3gpp, "TR 38.901 human (Sionna RCSSolver, single point)"),
                         (axs[1], H_art, "Articulated 12-segment body")):
        t, v, P = spectrogram(H + noise[0] + 1j * noise[1], r, rng)
        ax.pcolormesh(t, v, 10 * np.log10(P + 1e-30), shading="auto", vmin=10 * np.log10(P.max()) - 45)
        for x in (ev["t0"], ev["t0"] + ev["D"]):
            ax.axvline(x, color="w", ls="--", lw=1)
        ax.set(title=title, xlabel="time [s]", ylim=(-5, 5), xlim=(ev["t0"] - 1.0, dur))
    axs[0].set_ylabel("radial velocity [m/s]")
    fig.suptitle(f"{ev['kind']} event, peak limb speed {ev['peak_speed']:.1f} m/s, torso static "
                 f"(3.5 GHz, 2 kHz slow time)")
    plt.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "fig_3gpp_vs_articulated.png")
    plt.savefig(path, dpi=120)
    print(path)


if __name__ == "__main__":
    main()
