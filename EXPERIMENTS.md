# Experiment log

Run date: 2026-11-17

These results use the official synthetic sandbox generator. They verify the
pipeline and demonstrate behavior under the generator's invented dependencies.
They do **not** estimate performance on the unreleased survey microdata.

## Official schema-shaped sandboxes

| Instrument | Development skill | Development runtime | Final skill | Final runtime |
|---|---:|---:|---:|---:|
| UNICEF | 0.3912 | 2.2 s | 0.3801 | 3.1 s |
| World Bank | 0.2513 | 4.6 s | 0.2621 | 6.1 s |
| UNHCR | 0.2124 | 19.0 s | 0.2145 | 29.6 s |

The official crowd-marginal reference has skill 0. All six rehearsals passed
the output contract and completed far below the 900-second development and
3600-second final budgets.

## Small sample schema

- Development skill before privacy post-processing: 0.1647
- Reported development skill after the simulated privacy mechanism: 0.1605
- Runtime: 1.3 s

## Verification

The table above describes the original statistical model. The follow-up experiment compares it with the organizer's hard-gate model, calibrated gates and a per-item mixture. On a second synthetic draw made after the weight grid was fixed (generator seed 17), the direct predictor averaged 0.2844 skill and the mixture averaged 0.2930. These are equal-weight means over instruments with a uniform question prior. Individual runs are in [local logs](report/local-results/). The [reproduction instructions](report/reproduce.md) include the separate paired stress experiment. The competition's displayed leaderboard scores do not identify which methods other teams used.

- Direct unit test of output count, width, finiteness, non-negativity,
  normalization, and contextual direction: passed.
- Official `score.py`: passed for all three instruments in both phases.
- Upload archive validator: see the latest packaging run before submission.

## Offline Qwen rehearsal on the RTX 5060 Laptop GPU

The official phase-two scorer ran on full synthetic seed-0 data. A local adapter called the competition predictor and raised an error on missing, invalid or entirely uniform model priors. All three runs passed with the pinned checkpoint loaded.

| Instrument | Masked cells | Skill | Grader time |
|---|---:|---:|---:|
| UNICEF | 41,676 | 0.388161 | 155.83 s |
| World Bank | 92,112 | 0.272311 | 162.23 s |
| UNHCR | 308,637 | 0.215183 | 185.62 s |

The arithmetic mean skill was 0.291885. The same candidate with a uniform prior matched these three scores to four decimals. All runtimes include a model load in a fresh local process. The total was 503.68 seconds. In a separate small-frame check, Qwen produced finite, normalized priors for all 301 questions and PyTorch reserved at most 3,474 MiB of GPU memory. The measurements are from the local RTX GPU. [Run logs](report/local-results/) and [commands and versions](report/reproduce.md).
