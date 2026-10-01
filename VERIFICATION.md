# Sionna validation — verification run (2026-09-30)

## Environment
- Host (not a Docker container): Linux 6.8.0-124, AMD EPYC 9354 (64 threads), 2× NVIDIA H100 NVL,
  driver 580.126.09. Run pinned to host GPU 1 (`CUDA_VISIBLE_DEVICES=1`), the card used in
  `docker-compose.yml`. GPU 0 was busy with an unrelated job.
- Python 3.12.9 in an isolated venv created from miniconda base. This is **not** `/usr/bin/python3`
  (3.10.12): sionna-rt 2.2.0 imports `typing.Self`, which needs Python ≥ 3.11, so it cannot
  import there. The user-site `sionna 1.2.1` / `sionna-rt 1.2.1` install (no `sionna.rt.rcs`)
  was left untouched.
- sionna-rt 2.2.0, mitsuba 3.9.1, drjit 1.5.0, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
  pytest 9.1.1.
- Mitsuba variant from `mdsense.channel.select_mitsuba_variant()`: **`cuda_ad_mono_polarized`**.
  The LLVM fallback is unavailable on this host (system LLVM 13; Dr.Jit needs ≥ 15).

## Tests
`PYTHONPATH=. python -m pytest -q -s tests` → **17 passed, 0 skipped, 0 failed** (34.2 s).

| Check | Result | README value | Test bound | Runtime |
|---|---|---|---|---|
| `[stitching]` per-slot vs snapshot-stitched, corr | **0.999971** | 0.99997 | > 0.999 | 2.7 s |
| `[equivalence]` analytic twin vs RCSSolver, corr (dynamic) | **0.99998** | 0.99998 | > 0.98 | 29.8 s |
| `[equivalence]` power ratio | **1.00005** | 1.0001 | \|r − 1\| < 0.02 | (same test) |

## Figure
`scripts/fig_3gpp_vs_articulated.py --out figures` regenerated
`figures/fig_3gpp_vs_articulated.png` (1440×480 PNG, 50.6 s). It renders correctly and matches
the previous version visually: arm event, 3.4 m/s peak limb speed, no limb micro-Doppler for the
TR 38.901 single-point human, clear limb micro-Doppler for the articulated body. The file is not
byte-identical: about 15 % of pixels differ, which was not investigated further and is plausibly
caused by the newer plotting stack.

No code, config, `results/`, `analysis/`, or test-split data was changed or loaded; `run_pilot.py` was not run.
