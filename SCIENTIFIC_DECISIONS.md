# Pre-registered scientific decisions (lock before the one test split)

These are fixed in `mdsense/config.py` as `SCIENTIFIC_DECISIONS` and copied into
`preregistration.json`. Deadline 0.20 s, FA limit 1/min, SNR 5 dB, and go budgets
(0.025, 0.05) are unchanged.

## Total cost

GO matches **monitoring** cost (`go_cost_basis = monitor`, ratio 0.9–1.1 vs the
tuned burst). Post-alarm service is part of `cost_total` and is **reported**
(`cost_total_ratio`) but is not a GO gate: service is the operational response
after an alarm, so using it as an equality constraint would penalise a detector
that alarms more often when it should.

## Dropped samples and Doppler band

Busy or late probes are **not reconstructed**. A look uses only delivered
timestamps. The window Nyquist is `1 / (2 * median successive delivered
interval)`, then capped by `doppler_max`. Aliased energy remains in the
spectrum. `min_samples = 4`; looks with fewer accepted probes yield no statistic.

## Long-negative FA

After selection is frozen, FA is also measured on `n_neg = 200` event-free
episodes (`seed_neg = 50000`, `force_event=False`), duration and radio unchanged.
This split does **not** retune A. GO requires DE-CuSum FA on that split
`<= go_fa_neg_slack * fa_max` (slack 1.25), in addition to mixed-test FA.

## Energy / Doppler ablation

The primary detector uses both window features. On **validation only**, the
selected burst look is re-scored with energy-only, Doppler-only, and both,
each recailbrated to the FA limit. Ablation is descriptive, not a GO gate, and
does not touch the test split.

## Test

Test seeds start at 40000. Smoke seeds 30000–30079 are never reused. The test
split is generated and evaluated once after preregistration.

## 2026-10-01 — Post-hoc baseline: two-rate trigger schedule (not part of the registered GO)

Registered results, selections and the test split (seeds 40000+) are unchanged and not used.

**Policy (`TriggerPolicy`).** Looks of L probes spaced d slots. Slow mode: one look every
`P_slow` slots. After each completed look, CuSum `W <- max(0, W + llr)`; alarm when `W > A`,
with the same alarm, refractory and post-alarm service logic as `BasePolicy`. If a look's
llr > tau, the policy enters fast mode: looks back to back (period = look span L*d) for `N_hold`
looks, then slow mode again. A look with llr > tau while already in fast mode resets the
counter to `N_hold`. An alarm clears fast mode.

**Grid.** (L, d) in the registered `look_grid` {(8,4), (16,4), (16,8), (32,4)};
tau in {0, 2, 4}; N_hold in {2, 4, 8}. Budgets 0.025 and 0.05 only.

**LLR.** For each look, the Gaussian LLR fitted on train for the burst at that look and budget
(`fit_llr(train, burst(L, d))`, identical to the registered fit). No new model.

**Calibration (val only, seeds 20000+).** Cost target = the registered burst's val
`cost_monitor` at that budget. Alternation as in `run_pilot.py`: A from the FA cap ->
`P_slow` by bisection in log `P_slow` (integer slots, `P_slow >= span`; stop within 3 %) ->
twice more -> final A. A = smallest value with val FA <= 1/min (`calibrate_threshold`).
A config is eligible if its FA cap is met and its final val cost is within 5 % of the target.
If `P_slow` cannot bring cost down to the target, the config is infeasible.
**Selection:** lowest val P_fail among eligible configs, per budget.

**Evaluation split `fresh2`.** n = 300, seeds 70000–70299, same Config, events as in the
registered splits (p_event 0.5). On fresh2 we run the selected trigger and the **registered**
burst and DE-CuSum (`results/selected_models.pkl`: params, A, LLR, unchanged).

**Metrics per budget and policy.** P_fail; events detected in time (n_event − fails);
FA/min; monitoring cost; total cost; median delay. Paired bootstrap of
P_fail(trigger) − P_fail(DE) over all fresh2 event episodes paired by seed (no duration
cutoff), 5000 resamples, seed 0, 95 % percentile CI. Output: `analysis/trigger.json`.
Descriptive only; no GO decision depends on it.

## 2026-10-01 — Robustness check: multipath indoor room with Sionna RT (GPU)

Robustness only: no retuning; registered configs, features, LLRs, calibration and the
test split (seeds 40000+) are unchanged and not used.

**Scene.** Closed concrete room built from one box mesh (ITU `concrete`, 0.2 m), scene
name `room_8x10x3`: x in [−0.5, 7.5], y in [−5, 5], z in [0, 3] m. The radar at (0, 0, 2.5)
is 0.5 m from the back wall. No built-in Sionna scene fits (the closed `box` scene is metal).
*Deviation from the brief ("about 8 × 6 × 3 m"):* the width is 10 m because the registered
person geometry (range 3–7 m, azimuth ±40°) reaches |y| = 4.5 m and must stay unchanged.
The scenario's constant-RCS clutter points are kept as sampled. With refraction off, points
outside the walls receive no paths, so the walls act as the static clutter.

**Channel.** `SionnaBackend(scene="room_8x10x3", max_depth=3, background=True)`,
`snapshot_slots = 8`. Sensing paths: `RCSSolver(deterministic=True)`, max_depth 3, LoS +
specular reflection, refraction off (no diffraction exists in the RCS solver), seed 0,
default sample counts. Static background: `PathSolver`, max_depth 3, `los=False`
(monostatic), specular only, refraction off, seed 0. It is solved once per episode in the
walls-only scene and added to every slot, so person shadowing of wall echoes is not
modelled. Common drift and keyed noise are as in free space. Scene name, max_depth and the
background flag are part of the cache fingerprint.

**Pre-checks (before the full split).** (i) Snapshot check on seed 80000:
corr(dynamic, mean removed) between snapshot_slots 8 and 4 must be > 0.999; otherwise stop
and report. (ii) Time 3 episodes (seeds 80000–80002) and report the estimate.
n_room = 300, or 200 if 300 episodes would take > 8 h.

**Split `room`.** Seeds 80000+, n_room, same Config (p_event 0.5).

**Policies, no retuning.** Registered uniform, burst and DE-CuSum from
`results/selected_models.pkl` (params, A, LLR) at budgets 0.025, 0.05, 0.10. Selected
trigger configs from `analysis/trigger.json` at 0.025 and 0.05 (its A and P_slow; LLR = the
train-fitted burst LLR for its look and budget, as in that run).

**Metrics.** Per budget and policy: P_fail, events in time, FA/min, monitoring cost, total
cost, median delay. DE vs burst, as in the paper: gain = P_fail(burst) − P_fail(DE) over
event episodes with D ≤ 0.6 s, paired by seed. Trigger − DE over all event episodes, as in
the trigger analysis. Both use a paired bootstrap with 5000 resamples, seed 0, 95 %
percentile CI. **"Main result holds in the room"** if, at both 0.025 and 0.05: gain ≥ 0.10,
CI lower bound > 0, and DE FA ≤ 1.25/min.

**Secondary (only if any registered uniform, burst or DE-CuSum exceeds 1.25 FA/min in the
room at any budget).** Separate split `room_val`, seeds 81000+, n = 100, same room settings.
For every policy and budget (trigger included), re-calibrate only A (smallest A with
room_val FA ≤ 1/min, `calibrate_threshold`); mu, P_slow and looks are unchanged. Then
re-evaluate on `room` with the same metrics. Labelled **secondary**: costs are no longer
re-matched.

Output: `analysis/room.json` (new file).
