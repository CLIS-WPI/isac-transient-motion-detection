import numpy as np

from mdsense import Config
from mdsense.features import delivered_nyquist
from mdsense.score_model import GaussianLLR, SliceLLR


def test_delivered_nyquist_uses_median_gap_not_max_gap():
    du = Config().du
    dt = 0.002
    t_reg = np.arange(8) * dt
    assert np.isclose(delivered_nyquist(t_reg, du), min(du.doppler_max, 0.5 / dt))
    t_drop = np.array([0.0, dt, 4 * dt, 5 * dt, 6 * dt])  # one 3-slot hole
    med = np.median(np.diff(t_drop))
    assert np.isclose(delivered_nyquist(t_drop, du), min(du.doppler_max, 0.5 / med))
    assert delivered_nyquist(t_drop, du) > 0.5 / np.max(np.diff(t_drop))


def test_slice_llr_energy_vs_both():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(40, 2))
    X1 = rng.normal(loc=[2.0, 0.0], size=(40, 2))
    both = GaussianLLR().fit(X0, X1)
    energy = GaussianLLR().fit(X0[:, :1], X1[:, :1])
    x = np.array([2.0, 0.0])
    assert SliceLLR(energy, (0,))(x) == energy(x[:1])
    assert both(x) != 0.0
