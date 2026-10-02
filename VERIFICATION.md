# Sionna validation — verification run (2026-09-30)

## Environment (primary run: Docker)
- Image `mdsense:phase1` built from this repo's `Dockerfile` (`nvidia/cuda:12.6.3-runtime-ubuntu24.04`).
  It was run with `--gpus device=1`, so the container sees host GPU 1 as device 0
  (NVIDIA H100 NVL, `GPU-509d8ab3-0c8e-8b98-fbf8-0c55f6ba9058`, driver 580.126.09).
  Host: AMD EPYC 9354, Linux 6.8.0-124.
- Interpreter: `/usr/bin/python3` (3.12.3) inside the container. sionna-rt was missing for that
  interpreter, so it was installed with `pip install sionna-rt==2.2.0 pytest`. Resolved versions:
  sionna-rt 2.2.0, mitsuba 3.9.1, drjit 1.5.0, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
  pytest 9.1.1.
- Mitsuba variant from `mdsense.channel.select_mitsuba_variant()`: **`cuda_ad_mono_polarized`**.
- **Image gap:** in the image as built, `import drjit` fails (`libatomic.so.1: cannot open shared
  object file`) for both `/opt/venv` and `/usr/bin/python3`. The fix is `apt-get install libatomic1`.
  It was applied only inside the throwaway `--rm` container; the `Dockerfile` is unchanged.

## Tests
`PYTHONPATH=. /usr/bin/python3 -m pytest -q -s tests` → **17 passed, 0 skipped, 0 failed** (34.5 s).

| Check | Result | README value | Test bound | Runtime |
|---|---|---|---|---|
| `[stitching]` per-slot vs snapshot-stitched, corr | **0.999971** | 0.99997 | > 0.999 | 2.2 s |
| `[equivalence]` analytic twin vs RCSSolver, corr (dynamic) | **0.99998** | 0.99998 | > 0.98 | 30.7 s |
| `[equivalence]` power ratio | **1.00005** | 1.0001 | \|r − 1\| < 0.02 | (same test) |

Cross-check: the same suite was also run on the host in a Python 3.12.9 venv with the same package
versions (host `/usr/bin/python3` is 3.10 and cannot import sionna-rt 2.2). That run gave
identical numbers in 34.2 s.

## Figure
`scripts/fig_3gpp_vs_articulated.py --out figures` regenerated
`figures/fig_3gpp_vs_articulated.png` in the container (1440×480 PNG, 52.4 s). It renders
correctly and is byte-identical to the host-venv regeneration. It shows an arm event at 3.4 m/s
peak limb speed: the TR 38.901 single-point human has no limb micro-Doppler, and the articulated
body shows clear limb micro-Doppler. Compared with the figure committed in `9de2e0e`, it is
visually the same but not byte-identical (about 15 % of pixels differ; not investigated,
plausibly the newer plotting stack).

No code, `Dockerfile`, config, `results/`, `analysis/`, or test-split data was changed or loaded.
The container mounted only `figures/`. `run_pilot.py` was not run.

## Addendum (2026-10-02)
The image gap above is fixed: `libatomic1` is in the `Dockerfile` (commit `6e2e1f3`), and Python
packages are pinned in `requirements.txt`. A fresh image with no manual installs passes
`python -m pytest -q tests`: 18 passed, 0 skipped (17 above + `test_trigger.py`).
