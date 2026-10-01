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
