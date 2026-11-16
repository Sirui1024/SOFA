# Objective units and PHV conventions

Raw objective order is `[utility loss (%), cost (USD), grid carbon (kg), water (L)]`;
all four are minimized. The first coordinate uses percentage points: 100 means
100% loss, not 100% achieved utility. Water-treatment carbon is not included in
the carbon objective.

There are three PHV paths with different scaling conventions:

| Path | Convention |
| --- | --- |
| Framework search history | Framework-specific scaling and reference; not a shared cross-framework metric |
| Simulator `calculate_final_phv` | Raw hypervolume with reference `[100, 100, 100, 100]`; points outside that box contribute zero |
| Seed summary | Raw objectives divided by a common reference, then hypervolume against `[1, 1, 1, 1]` |

The simulator's reference is often too small for real cost, carbon, and
water values. Use the seed summary with an explicit common reference for
comparisons, not the simulator's printed PHV. For the *same* objective set and
reference, `raw_HV = normalized_HV * product(reference)`. Changing the reference
requires recomputing hypervolume; it is not a unit conversion.

All fixed-routing policies are evaluated by [ENV_V3.py](../ENV_V3.py).

For a completed six-framework, ten-seed batch, use
[summarize_seed_runs.py](../summarize_seed_runs.py) with a justified common
reference in raw objective units. For example:

```sh
python summarize_seed_runs.py results --reference 20 1000 1500 15000
```

These bounds are illustrative, not prescribed. Use `--allow-partial` only for
partial-batch diagnostics and report the actual sample counts.
