# Per-run experimental results

Raw per-run records behind the tables and figures in *DupliMend: Online
Detection and Refinement of Imprecise Activity Labels*. Every row is one
(log, method, seed) observation — these are the inputs to the reported means and
standard deviations, not the aggregates themselves.

The `.csv.gz` files are plain gzipped CSV: `gunzip -c <file> | head`, or in
pandas `pd.read_csv(path)` reads them directly.

## Files

| File | Rows | Contents |
|---|---|---|
| `baselines_per_run_best_params.csv.gz` | 39,120 | Label-Refinement, PM-Label-Splitting and Unrefined, one row per (log, seed) after per-log best-parameter selection. 1,304 logs x 10 seeds x 3 methods. |
| `duplimend_per_checkpoint_all_seeds.csv.gz` | 247,760 | DupliMend at every streaming checkpoint (10%…100%) for each log and seed. Filter `Checkpoint_Percentage == 100` for the final-checkpoint values the paper reports. |
| `duplimend_streaming_per_checkpoint_seeds1-5.csv.gz` | 32,625 | Streaming-performance records per checkpoint. **Seeds 1–5 only.** |
| `cybersec_per_log_seeds1-5.csv.gz` | 100,340 | Large-scale cybersecurity/IoT collection, per test log. **Seeds 1–5 only.** |
| `aggregated_duplimend_results.csv` | 50 | DupliMend aggregated by dataset type and checkpoint. |
| `aggregated_baseline_results.csv` | 30 | Baselines aggregated by dataset type and method. |
| `aggregated_cybersec_results.csv` | 3 | Cybersecurity/IoT aggregates. |

## Columns

Metric columns are shared across files: `ARI`, `NMI`, `Silhouette_Score`,
`Log_Precision`, `Log_Fitness`, `F-score`, `Expected_Entropy_Clusters` and
`Expected_Entropy_Labels`. Identifying columns are `Dataset`, `Dataset_Type`,
`Method` and `Seed`; the baseline sweep additionally carries the parameter
columns `variant_threshold`, `unfolding_threshold`, `similarity_threshold`,
`context_size` and `distance_metric`.

`Seed` identifies the independent run. Coverage differs by file and is stated
in the table above: the two main result files
(`baselines_per_run_best_params.csv.gz` and
`duplimend_per_checkpoint_all_seeds.csv.gz`) carry the full ten runs, seeds
1–10. The streaming and cybersecurity files carry seeds 1–5. See
`config/paper_configs/seeds.json`.

## Selection protocol

The baselines were swept exhaustively over their full grids
(`config/paper_configs/baseline_grids.json`) and the best-scoring parameter
combination was retained for each (log, seed) pair. DupliMend used one
configuration per dataset rather than per-log selection, so the comparison is
deliberately favourable to the baselines. The complete unselected sweep
(every parameter combination x log x seed, ~700 MB) is not included here for
size reasons and is available from the authors on request.

## Note on file naming

Files whose names end in `_seeds1-5` contain five runs rather than ten; the
suffix is there so the coverage is visible without opening the file. The two
main result files carry all ten seeds.
