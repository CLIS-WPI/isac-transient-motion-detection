import numpy as np

from mdsense import Config
from mdsense.channel import _point_response
from mdsense.body import ellipsoid_rcs

R = Config().radio


def test_radar_equation():
    radar = np.asarray(R.radar_position)
    pos = radar + np.array([[0.0, 5.0, 0.0]])
    H = _point_response(pos, np.array([2.0]), R, radar)
    expected = 2.0 * R.wavelength ** 2 / ((4 * np.pi) ** 3 * 5.0 ** 4)
    assert np.allclose(np.abs(H) ** 2, expected, rtol=1e-9)


def test_doppler_sign_and_value():
    radar = np.asarray(R.radar_position)
    v = 2.0                                           # approaching at 2 m/s
    t = np.arange(2000) * R.slot_duration
    pos = radar + np.stack([np.zeros_like(t), 6.0 - v * t, np.zeros_like(t)], 1)[:, None, :]
    h = _point_response(pos, np.ones((len(t), 1)), R, radar)[:, 0]
    f = np.fft.fftfreq(len(t), R.slot_duration)
    f_peak = f[np.argmax(np.abs(np.fft.fft(h * np.hanning(len(t)))))]
    assert abs(f_peak - 2 * v / R.wavelength) < 1.5    # approaching -> positive Doppler


def test_ellipsoid_rcs_limits():
    dims = np.array([[0.15, 0.30]])
    axis = np.array([[0.0, 0.0, 1.0]])
    broad = ellipsoid_rcs(np.array([[1.0, 0.0, 0.0]]), axis, dims)
    end = ellipsoid_rcs(np.array([[0.0, 0.0, 1.0]]), axis, dims)
    assert np.isclose(broad, np.pi * 0.30 ** 2)
    assert np.isclose(end, np.pi * 0.15 ** 4 / 0.30 ** 2)
