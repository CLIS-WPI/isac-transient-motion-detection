import dataclasses
import json
import os
import numpy as np

from mdsense import Config
from mdsense.env import episode_noise, run_episode
from mdsense.evaluate import by_seed
from mdsense.keyed_rng import STREAM_BUSY, keyed_cnormal, keyed_uniform
from mdsense.pipeline import (
    Split, cache_fingerprint, close_workers, factory, make_split, run_split,
)


def _toy_split(tmp_path, seeds=(11, 12, 13, 14, 15, 16, 17, 18)):
    cfg = Config()
    n_slots = 400
    d = tmp_path / "toy"
    d.mkdir()
    labels = {}
    rng = np.random.default_rng(0)
    for seed in seeds:
        H = (rng.standard_normal((n_slots, cfg.radio.num_re))
             + 1j * rng.standard_normal((n_slots, cfg.radio.num_re))).astype(np.complex64)
        np.save(d / f"ep{seed}.npy", H)
        labels[str(seed)] = {"event": None, "nuisances": []}
    with open(d / "labels.json", "w") as f:
        json.dump(labels, f)
    split = Split(str(d))
    cfg = dataclasses.replace(
        cfg, scenario=dataclasses.replace(cfg.scenario, duration=n_slots * cfg.radio.slot_duration,
                                         warmup=0.0))
    return split, cfg


def test_channel_busy_noise_are_cached(tmp_path):
    split, cfg = _toy_split(tmp_path, seeds=(7, 8))
    a = split.channel(7)
    assert a is split.channel(7)
    p = cfg.scenario.comm_busy_prob
    m = split.busy(7, a.shape[0], p)
    assert m is split.busy(7, a.shape[0], p)
    assert np.array_equal(m, keyed_uniform(7, STREAM_BUSY, np.arange(a.shape[0])) < p)
    N = split.noise(7, a.shape[0], a.shape[1])
    assert N is split.noise(7, a.shape[0], a.shape[1])
    k = np.arange(a.shape[1])
    assert np.allclose(N[17], keyed_cnormal(7, 17, k))
    assert np.allclose(N, episode_noise(7, a.shape[0], a.shape[1]))


def test_cached_noise_matches_uncached_episode(tmp_path):
    split, cfg = _toy_split(tmp_path, seeds=(9,))
    H = split.channel(9)
    mk = factory("uniform", cfg, 0.05, n_window=8)
    split.preload(cfg)
    log_a = run_episode(H, 9, mk(), cfg,
                        busy=split.busy(9, H.shape[0], cfg.scenario.comm_busy_prob),
                        noise=split.noise(9, H.shape[0], H.shape[1]))
    log_b = run_episode(H, 9, mk(), cfg)
    assert {p["slot"] for p in log_a.probes} == {p["slot"] for p in log_b.probes}


def test_run_split_parallel_matches_serial_by_seed(tmp_path):
    split, cfg = _toy_split(tmp_path)
    mk = factory("uniform", cfg, 0.05, n_window=8)
    try:
        serial = run_split(split, mk, cfg, workers=1)
        parallel = run_split(split, mk, cfg, workers=8)
        assert [o["seed"] for o in serial] == [o["seed"] for o in parallel] == list(split.seeds)
        s, p = by_seed(serial), by_seed(parallel)
        assert set(s) == set(p) == set(split.seeds)
        for seed in split.seeds:
            assert s[seed]["n_fa"] == p[seed]["n_fa"]
            assert s[seed]["cost_monitor"] == p[seed]["cost_monitor"]
            assert s[seed]["cost_total"] == p[seed]["cost_total"]
    finally:
        close_workers()


def test_cache_fingerprint_keys_radio_and_scenario():
    cfg = Config()
    a = cache_fingerprint(cfg)
    assert a != cache_fingerprint(dataclasses.replace(
        cfg, radio=dataclasses.replace(cfg.radio, snr_ref_db=10.0)))
    assert a != cache_fingerprint(dataclasses.replace(
        cfg, scenario=dataclasses.replace(cfg.scenario, duration=4.0)))
    assert a != cache_fingerprint(cfg, backend="sionna")
    assert a != cache_fingerprint(cfg, force_event=False)
    assert a == cache_fingerprint(cfg)


def test_make_split_writes_under_settings_hash(tmp_path):
    base = Config()
    cfg = dataclasses.replace(
        base, scenario=dataclasses.replace(base.scenario, duration=0.01, warmup=0.0))
    d1 = make_split(cfg, "toy", 1, 0, str(tmp_path), verbose=False)
    fp1 = cache_fingerprint(cfg)
    npy = os.path.join(d1, "ep0.npy")
    assert os.path.basename(os.path.dirname(d1)) == fp1
    assert os.path.isfile(npy)
    assert os.path.isfile(os.path.join(tmp_path, fp1, "cache_spec.json"))
    mtime = os.path.getmtime(npy)
    make_split(cfg, "toy", 1, 0, str(tmp_path), verbose=False)
    assert os.path.getmtime(npy) == mtime
    cfg2 = dataclasses.replace(cfg, radio=dataclasses.replace(cfg.radio, fc=28e9))
    d2 = make_split(cfg2, "toy", 1, 0, str(tmp_path), verbose=False)
    assert os.path.dirname(d2) != os.path.dirname(d1)
    assert os.path.isfile(os.path.join(d2, "ep0.npy"))
