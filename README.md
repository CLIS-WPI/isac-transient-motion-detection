# mdsense — phase-1 pilot

Question: at approximately matched monitoring cost and a common false-alarm cap, does an
adapted DE-CuSum observation policy detect the onset of fast limb motion (static torso) within a
deadline more often than a tuned periodic burst? Uniform equal-budget sampling is a third
reference. No optimality is claimed for the DE-CuSum adaptation.

## Observation contract (enforced in code, tested)
- `run_episode()` owns the environment. The policy only receives a frozen `PolicyInput`
  (time, delivered `Observation`s with read-only arrays, `Ack`s) and returns requests + alarm.
  The dense channel, noise seed and labels never cross the boundary.
- Five timestamps per probe: request, arrival at scheduler, transmission (cost booked),
  acquisition, delivery. Busy communication slots (30 %) reject probes.
- Noise keyed by `(scenario_seed, slot, RE, rx)` → identical for all policies at a given slot.
- Nyquist is not imposed: policies get complex samples; aliasing arises from the data.
- Labels come from the motion scenario; only `evaluate.py` reads them.
- Order: train (LLR fit) → val (thresholds, cost match, hyper-parameters, feature ablation) →
  `preregistration.json` → event-free FA split → test once → verdict.

See `SCIENTIFIC_DECISIONS.md` for the locked cost / Nyquist / FA / ablation rules.

## Layout
```
mdsense/config.py      all constants, budgets, deadline, FA limit, go criteria
mdsense/body.py        12-segment ellipsoid body, event/nuisance/breathing/sway/jitter
mdsense/channel.py     dense channel: analytic twin (default) and Sionna RT RCSSolver
mdsense/env.py         causal environment and message types
mdsense/features.py    DU processing: range gate, EMA background, detrended window features
mdsense/policies.py    uniform, burst (look = L probes x spacing), adapted DE-CuSum
mdsense/pipeline.py    data, window collection, LLR fit, threshold and cost calibration
mdsense/evaluate.py    failure / FA / cost, paired bootstrap
scripts/run_pilot.py   full pipeline
scripts/fig_3gpp_vs_articulated.py   motivating figure (needs sionna-rt)
tests/                 contract, physics, Sionna equivalence
```

## Run
```bash
pip install numpy scipy matplotlib pytest sionna-rt==2.2.0   # sionna only for its backend/figure/tests
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python scripts/run_pilot.py --quick --root /tmp/d --out /tmp/r   # ~8 min smoke test
PYTHONPATH=. python scripts/run_pilot.py --workers 16 --root data --out results   # full pilot
```
`--root` is a parent directory. Channels are written under
`data/<12-hex fingerprint of radio+scenario+backend>/`, so a config change cannot
silently reuse another setting's `.npy` files. The arrays are determined by the
episode seeds and are gitignored; delete the fingerprint folder to regenerate.
The full run (3 budgets, 4 look designs, 3 h values, 200/200/300 episodes) is
roughly 2–4 h on one core; budgets are independent and can be run in parallel.

## Validation status
- All tests pass. Sionna RT 2.2 `RCSSolver` vs the analytic twin on the articulated body:
  correlation 0.99998, power ratio 1.0001; snapshot stitching vs per-slot solve: 0.99997.
  The analytic twin is used for Monte Carlo (~0.1 s/episode vs ~1 min with Sionna).
- Smoke run (`results_smoke/`, 60/60/80 episodes, budget 5 %, one config each) is a
  plumbing check, not evidence: 24 test events, wide CI.
- Test seeds 30000–30079 were seen in the smoke run; the full run uses 40000+.

## Known limitations (state them in the paper)
Free space, LoS only, single antenna, constant ellipsoid RCS per segment, no ground
reflection, i.i.d. busy slots, E2 delay not modelled (all three policies run in the DU),
drift/SNR/deadline/FA limit are assumptions to be justified by the application.
