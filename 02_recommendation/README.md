# Recommendation experiments

This stage contains the finalized recommendation notebooks for the three internal evaluation scenarios:

- DMF-Whole network
- CC-DMF
- DC-DMF

for both Douban Movie and Last.fm.

Training seeds: `42`, `52`, `62`.

Within each configuration, `SplitSeed = 42` and `EvaluationSeed = 42`.

Raw per-user outputs for CC-DMF and DC-DMF are stored under `outputs/` and organized by seed and community. `RAW_OUTPUT_MANIFEST.csv` gives the exact mapping. DMF-Whole network is supported by the finalized multi-seed regeneration notebooks.
