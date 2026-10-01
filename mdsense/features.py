"""DU processing shared by all policies. It only ever sees delivered observations.

Nyquist is not imposed on the data: the complex samples are used as they are. The
Doppler grid of a window is limited to that window's own unambiguous band from
delivered sample times (1 / (2 * median successive interval)). Dropped/busy probes
are not reconstructed; aliased energy stays in.
"""
from collections import deque
import numpy as np
from scipy.constants import speed_of_light as C


def delivered_nyquist(t, du):
    """Unambiguous Doppler [Hz] from delivered sample times only (dropped slots unused)."""
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        return du.doppler_max
    dt = max(np.median(np.diff(t)), 1e-9)
    return min(du.doppler_max, 0.5 / dt)


class DUProcessor:
    def __init__(self, pub, du, max_buffer=128):
        K = pub.num_re
        self.w = np.hanning(K + 2)[1:-1]
        self.w /= np.sqrt(np.mean(self.w ** 2))
        rb = C / (2 * pub.bandwidth)
        self.bins = np.arange(int(np.ceil(du.zone[0] / rb)), int(np.floor(du.zone[1] / rb)) + 1)
        self.noise_bin = pub.noise_var * np.sum(self.w ** 2) / K ** 2
        self.du = du
        self.m = None
        self.t_last = None
        self.t_first = None
        self.buf = deque(maxlen=max_buffer)     # (t, residual[bins])

    def _range_gate(self, y):
        return np.fft.ifft(y * self.w)[self.bins]

    def track(self, obs):
        """Update the background estimate only (e.g. from service probes)."""
        self._update(obs.t_acquire, self._range_gate(obs.y))

    def ingest(self, obs):
        z = self._range_gate(obs.y)
        res = np.zeros_like(z) if self.m is None else z - self.m
        self._update(obs.t_acquire, z)
        self.buf.append((obs.t_acquire, res))
        return res

    def _update(self, t, z):
        if self.m is None:
            self.m, self.t_last, self.t_first = z.copy(), t, t
            return
        a = 1.0 - np.exp(-max(t - self.t_last, 0.0) / self.du.clutter_tau)
        self.m = self.m + a * (z - self.m)
        self.t_last = t

    def statistic(self, n):
        """Features of the last n residuals, or None. Returns (x[2], sample_times).

        x[0]: log energy left after a complex linear detrend over the window (slow motion,
              breathing, sway and drift are ~linear over a short window), noise-normalised,
              in the strongest range bin of the zone
        x[1]: logit of the Hann-weighted detrended spectrum above the fast cutoff; the cutoff
              is at least 1.5/span (beyond the main lobe at DC) and the grid stops at the
              window's own Nyquist from delivered times
        """
        if n < self.du.min_samples or len(self.buf) < n:
            return None
        if self.t_last - self.t_first < self.du.clutter_tau:   # background not converged yet
            return None
        items = list(self.buf)[-n:]
        t = np.array([it[0] for it in items])
        E = np.array([it[1] for it in items])                  # [n, bins]
        tc = t - t.mean()
        A = np.stack([np.ones_like(tc), tc / max(np.ptp(t), 1e-9)], 1)
        coef, *_ = np.linalg.lstsq(A, E, rcond=None)
        Rz = E - A @ coef
        eb = np.mean(np.abs(Rz) ** 2, axis=0) / self.noise_bin * n / (n - 2)
        b = int(np.argmax(eb))                                  # strongest range bin
        energy, Rz = eb[b], Rz[:, b]
        dt = max(np.median(np.diff(t)), 1e-9)
        span = np.ptp(t) + dt
        f_ny = delivered_nyquist(t, self.du)
        f_cut = min(max(self.du.f_cut, 1.5 / span), 0.8 * f_ny)
        f = np.arange(-f_ny, f_ny + 1e-9, min(self.du.doppler_step, f_ny / 8))
        w = np.sin(np.pi * (t - t[0] + dt / 2) / span) ** 2
        ph = np.exp(-2j * np.pi * np.outer(f, t - t[-1])) * w[None]
        S = np.abs(ph @ Rz) ** 2
        frac = np.clip(S[np.abs(f) > f_cut].sum() / max(S.sum(), 1e-30), 1e-3, 1 - 1e-3)
        x = np.array([np.log(energy + 1e-3), np.log(frac / (1 - frac))])
        return x, t
