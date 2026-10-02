# mdsense — sensing-schedule pilot for transient-motion detection in ISAC

**Question.** At approximately matched monitoring cost and a common false-alarm cap, does an
adapted DE-CuSum observation policy detect the onset of fast limb motion (static torso) within a
deadline more often than a tuned periodic burst? Uniform equal-budget sampling is a third
reference. No optimality is claimed for the DE-CuSum adaptation.

**Answer (pre-registered test split, `results/`).** GO at both pre-registered budgets.
The gain is P_fail(burst) − P_fail(DE-CuSum) over test events no longer than 0.6 s, with paired
bootstrap 95 % CIs (n = 93 events):

| budget (probes / slot) | gain | 95 % CI | DE-CuSum FA/min (test / event-free) | cost ratio DE/burst |
|---|---|---|---|---|
| 0.025 | +0.108 | [+0.022, +0.194] | 0.17 / 0.08 | 1.02 |
| 0.05  | +0.247 | [+0.161, +0.344] | 0.40 / 0.30 | 1.04 |

The result also holds without retuning in a multipath concrete room simulated with Sionna RT
(`analysis/room.json`; see "Post-hoc analyses").

---

## Where each result lives

| Result | File(s) | Produced by | Status |
|---|---|---|---|
| Main comparison, GO verdict | `results/results.json`, `results/log.txt` | `scripts/run_pilot.py` | pre-registered, test split used once |
| Frozen selection (configs, thresholds, LLRs) | `results/preregistration.json`, `results/selected_models.pkl` | `scripts/run_pilot.py` | written before the test split was generated |
| Feature ablation, same-look burst, threshold sweep, budget curve | `analysis/posthoc.json`, `analysis/fa_curve.png`, `analysis/budget_curve.png` | `scripts/analysis_posthoc.py` | post-hoc, val and fresh split (60000+) |
| Two-rate trigger baseline | `analysis/trigger.json` | `scripts/trigger_baseline.py` | post-hoc, pre-registered 2026-10-01, fresh split (70000+) |
| Multipath room robustness (Sionna RT) | `analysis/room.json` | `scripts/room_writer.py` + `scripts/room_robustness.py` | post-hoc, pre-registered 2026-10-01, room split (80000+) |
| Motivating figure (TR 38.901 vs articulated body) | `figures/fig_3gpp_vs_articulated.png` | `scripts/fig_3gpp_vs_articulated.py` | — |
| Sionna validation record | `VERIFICATION.md` | `tests/test_sionna_equivalence.py` | — |
| Smoke run (plumbing check, not evidence) | `results_smoke/` | `run_pilot.py --quick` | — |

All decision rules, and every deviation from them, are written in **`SCIENTIFIC_DECISIONS.md`**.
Each post-hoc experiment has a dated section there, and that section was committed before its
data was generated (see `git log`).

## Environment

The Docker image is the reference environment: CUDA 12.6.3, Ubuntu 24.04, Python 3.12 and the
pinned versions in `requirements.txt` (sionna-rt 2.2.0, mitsuba 3.9.1, drjit 1.5.0).

```bash
docker compose build
docker compose run --rm mdsense python -m pytest -q tests      # 18 passed, 0 skipped (~35 s)
```

`docker-compose.yml` pins host GPU 1 (`device_ids: ["1"]`); change it for your machine. It mounts
`data/`, `results/`, `analysis/` and `figures/`. The GPU is needed only for the Sionna parts
(figure, Sionna tests, room experiment); everything else runs on CPU.

Without Docker, use Python ≥ 3.11 (sionna-rt 2.2 needs it) and run
`pip install -r requirements.txt`. drjit also needs the system library `libatomic1`. Without
sionna-rt the two Sionna tests are skipped and everything else works.

## Reproducing the paper

Run from the repository root (`PYTHONPATH=.` is set in the image). The runtimes were measured
with 16 CPU workers on an AMD EPYC 9354 and one H100 NVL.

| Step | Command | Output | Runtime |
|---|---|---|---|
| Tests | `python -m pytest -q tests` | — | 35 s |
| Main pilot (**generates and evaluates the test split; re-running it is a replication, not part of the pre-registered run**) | `python scripts/run_pilot.py --root data --out <new_dir> --workers 16` | `results.json`, `preregistration.json`, `selected_models.pkl`, `log.txt` | 15 min |
| Post-hoc analyses | `python scripts/analysis_posthoc.py --root data --prereg results/preregistration.json --out analysis --workers 16` | `analysis/posthoc.json`, two PNGs | 6 min |
| Trigger baseline | `python scripts/trigger_baseline.py --root data --out analysis --workers 16` | `analysis/trigger.json` | 18 min |
| Room channels (GPU) | `for j in 0 1 2 3 4 5; do python scripts/room_writer.py room 80000 80200 $j 6 & done; wait`, then the same with `room_val 81000 81100` | `data/24e1cdbd44be/{room,room_val}/` | 6.3 h + 3 h (one H100) |
| Room evaluation | `python scripts/room_robustness.py --n 200 --workers 16` | `analysis/room.json` | 1 min (after the channels exist) |
| Figure (GPU) | `python scripts/fig_3gpp_vs_articulated.py --out figures` | `figures/fig_3gpp_vs_articulated.png` | 1 min |

Free-space channels are regenerated from the seeds on first use (about 0.1 s per episode). The
post-hoc scripts read the frozen selection from `results/` and never load the test split.
`room_robustness.py` can also generate missing room episodes on its own, but serially
(about 6 min per episode). `room_writer.py` runs several writers on one GPU and produces
identical files.

**Reproducibility check.** The committed `analysis/*.json` were produced on the host with
Python 3.10 and numpy 1.26. Re-running `analysis_posthoc.py`, `trigger_baseline.py` and
`room_robustness.py` in the Docker image (Python 3.12, numpy 2.5) reproduces all three files
exactly: zero numeric differences at 1e-9 relative tolerance, excluding timestamps and runtimes.
The PNGs can differ at the pixel level across matplotlib versions.

## Data, seeds and caching

Channels are written to `data/<12-hex fingerprint>/<split>/ep<seed>.npy`, with labels in a
separate `labels.json`. The fingerprint hashes radio, scenario, backend, snapshot length,
`force_event` and (for the room) scene name, `max_depth` and background flag. A config change
therefore cannot reuse stale files. `data/` is gitignored: every array is a deterministic
function of its seed, and all randomness is counter-based (`mdsense/keyed_rng.py`).

| Split | Seeds | n | Used for |
|---|---|---|---|
| train | 10000+ | 200 | LLR fitting only |
| val | 20000+ | 200 | thresholds, cost matching, hyper-parameters, ablation |
| smoke test | 30000–30079 | 80 | smoke run only; never reused |
| **test** | **40000+** | 300 | main result, evaluated once after pre-registration |
| neg | 50000+ | 200 | event-free FA check (does not retune) |
| curve | 60000+ | 300 | post-hoc budget curve |
| fresh2 | 70000+ | 300 | post-hoc trigger comparison |
| room | 80000+ | 200 | multipath room (Sionna RT) |
| room_val | 81000+ | 100 | room threshold-only recalibration (secondary) |

## Post-hoc analyses (descriptive, not part of the GO decision)

- **Ablation, same look, threshold sweep, budget curve** (`analysis/posthoc.json`). DE-CuSum's
  advantage is not explained by its look design alone: a burst using DE-CuSum's look is
  still worse.
- **Two-rate trigger baseline** (`analysis/trigger.json`), tuned on val with the same cost and FA
  matching, evaluated on fresh2. It beats the burst but not DE-CuSum. P_fail(trigger) −
  P_fail(DE-CuSum): +0.014 [−0.057, +0.086] at b = 0.025 and +0.086 [+0.000, +0.179] at b = 0.05.
- **Multipath room** (`analysis/room.json`). Closed concrete room, 8 × 10 × 3 m, simulated with
  Sionna RT on a GPU: RCSSolver with max_depth 3 plus the static wall echo; registered policies
  without retuning; n = 200. Burst − DE-CuSum gain: +0.136 [+0.017, +0.254] at b = 0.025,
  +0.305 [+0.169, +0.441] at b = 0.05, +0.220 [+0.102, +0.339] at b = 0.10. In the room the
  uniform and burst policies exceed 1.25 FA/min, so a pre-registered secondary threshold-only
  recalibration on `room_val` is also reported. Caveat: at b = 0.025, DE-CuSum's room
  monitoring cost is 1.28× the burst's.

## Method notes

**Observation contract (enforced in code, tested in `tests/test_contract.py`).**
- `run_episode()` owns the environment. The policy only receives a frozen `PolicyInput` (time,
  delivered `Observation`s with read-only arrays, `Ack`s) and returns requests and an alarm. The
  dense channel, the noise seed and the labels never cross that boundary.
- Each probe records five timestamps: request, arrival at the scheduler, transmission (cost is
  booked here), acquisition and delivery. Busy communication slots (30 %) reject probes.
- Noise is keyed by `(scenario_seed, slot, RE, rx)`, so all policies see identical noise in a
  given slot.
- Nyquist is not imposed, and dropped probes are not reconstructed.
- Labels come from the motion scenario. Only `evaluate.py` reads them.

**Code layout.**
```
mdsense/config.py      all constants, budgets, deadline, FA limit, GO criteria, seeds
mdsense/body.py        12-segment ellipsoid body; event, nuisance, breathing, sway, jitter
mdsense/channel.py     dense channel: analytic twin (default) and Sionna RT backend (free space or room)
mdsense/env.py         causal environment and message types
mdsense/features.py    DU processing: range gate, EMA background, window features
mdsense/policies.py    uniform, burst, adapted DE-CuSum, two-rate trigger (post-hoc)
mdsense/pipeline.py    splits and cache, window collection, LLR fit, threshold and cost calibration
mdsense/evaluate.py    failure / FA / cost, paired bootstrap
```

**Validation.** Sionna RT 2.2 `RCSSolver` vs the analytic twin on the articulated body:
dynamic correlation 0.99998, power ratio 1.00005. Snapshot stitching vs a per-slot solve:
0.999971 (`VERIFICATION.md`). In the room, snapshot 8 vs 4 slots gives 0.999992. The analytic
twin is used for the free-space Monte Carlo (about 0.1 s per episode, against about 6 min per
room episode with Sionna).

## Known limitations

- Free space for the main result: LoS only, single isotropic antenna, constant ellipsoid RCS
  per segment, no ground reflection.
- The room experiment adds specular wall reflections and the static wall echo. It ignores
  diffraction, transmission and person shadowing of the walls, and models a single room.
- Busy slots are i.i.d., and E2 delay is not modelled: all policies run in the DU.
- The drift, SNR, deadline and FA limit are assumptions to be justified by the application.
