#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman
from itertools import combinations
import os
import sys

parser = argparse.ArgumentParser(
    description='Friedman + Nemenyi significance tests over per-seed evaluation results.')
parser.add_argument('--results-dir', type=str,
                    default=os.environ.get('DUPLIMEND_SIGNIFICANCE_DIR', 'statistical_significance_output'),
                    help='Directory holding the per-seed result CSVs. Overrides '
                         'DUPLIMEND_SIGNIFICANCE_DIR; defaults to ./statistical_significance_output')
parser.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Explicit list of seeds to test, e.g. --seeds 1 2 3 4 5. '
                         'Defaults to every seed present in the input files.')
parser.add_argument('--skip-incomplete', action='store_true',
                    help='Exclude any dataset/method missing a seed instead of aborting. '
                         'Skipped entries are reported and written to skipped_entries.csv. '
                         'Values are never imputed under either mode.')
args = parser.parse_args()

OUTPUT_DIR = args.results_dir

SKIPPED_ENTRIES = []


def collect_seed_values(rows, metric, seeds, dataset, method, single_value=False):
    """Return one value per seed, in seed order.

    Every seed must be present. A missing seed is a data problem, not something to
    impute: under the default strict mode it aborts, and with --skip-incomplete the
    dataset/method is dropped from the test and recorded in SKIPPED_ENTRIES. This
    function never invents a value.
    """
    values = []
    missing = []

    for seed in seeds:
        seed_data = rows[rows['Seed'] == seed]
        if len(seed_data) == 0:
            missing.append(seed)
            continue
        values.append(seed_data[metric].values[0] if single_value else seed_data[metric].mean())

    if missing:
        detail = (f"{dataset} / {method} / {metric}: missing seed(s) "
                  f"{', '.join(str(s) for s in missing)} "
                  f"(expected {len(seeds)}: {', '.join(str(s) for s in seeds)})")
        if not args.skip_incomplete:
            sys.exit(
                f"\nERROR: incomplete seed matrix — {detail}\n"
                f"Every method must have a result for every seed before significance "
                f"testing. Re-run the missing seeds, or pass --skip-incomplete to exclude "
                f"this entry from the analysis.\n")
        SKIPPED_ENTRIES.append({'Dataset': dataset, 'Method': method,
                                'Metric': metric, 'Missing_Seeds': ';'.join(str(s) for s in missing)})
        return None

    return values


def resolve_seeds(*frames):
    """Determine the seed set to test, either from --seeds or from the input files."""
    if args.seeds:
        return list(args.seeds)

    present = set()
    for frame in frames:
        if frame is not None and 'Seed' in frame.columns:
            present.update(frame['Seed'].dropna().unique().tolist())

    if not present:
        sys.exit("ERROR: no 'Seed' column found in the input files — cannot determine "
                 "the number of independent runs. Pass --seeds explicitly.")
    return sorted(present)


def run_significance_test_for_stream(dataset_name, metric_name, method_data_dict):
    methods = list(method_data_dict.keys())
    n_methods = len(methods)
    n_seeds = len(list(method_data_dict.values())[0])

    if n_methods < 2 or n_seeds < 2:
        return None

    data_matrix = np.array([[method_data_dict[method][seed_idx]
                            for method in methods]
                           for seed_idx in range(n_seeds)])

    try:
        statistic, p_value = friedmanchisquare(*[data_matrix[:, i] for i in range(n_methods)])
    except:
        return None

    df_for_test = pd.DataFrame(data_matrix, columns=methods)

    try:
        nemenyi_results = posthoc_nemenyi_friedman(df_for_test)
    except:
        return None

    ranks_matrix = np.zeros_like(data_matrix)
    for row_idx in range(n_seeds):
        ranks_matrix[row_idx, :] = stats.rankdata(-data_matrix[row_idx, :], method='average')

    mean_ranks = {method: np.mean(ranks_matrix[:, i]) for i, method in enumerate(methods)}
    winner = min(mean_ranks, key=mean_ranks.get)

    significant_over = []
    for method in methods:
        if method != winner:
            try:
                p_val = nemenyi_results.loc[winner, method]
                if p_val < 0.05 and mean_ranks[winner] < mean_ranks[method]:
                    significant_over.append(method)
            except (KeyError, IndexError):
                continue

    return {
        'Dataset': dataset_name,
        'Metric': metric_name,
        'Friedman_Statistic': round(statistic, 4),
        'Friedman_P_Value': round(p_value, 6),
        'Significant': p_value <= 0.05,
        'Winner': winner,
        'Winner_Rank': round(mean_ranks[winner], 3),
        'N_Methods': n_methods,
        'N_Seeds': n_seeds,
        'Winner_Significantly_Better_Than': ','.join(significant_over) if significant_over else 'None'
    }


print("="*80)
print("GENERATING STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

print(f"\nResults directory: {OUTPUT_DIR}")
if not os.path.isdir(OUTPUT_DIR):
    sys.exit(f"ERROR: results directory not found: {OUTPUT_DIR}\n"
             f"Pass --results-dir or set DUPLIMEND_SIGNIFICANCE_DIR.")

print("Loading data files...")
baseline_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'baseline_results_with_all_parameters.csv'))
checkpoint_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'duplimend_baseline_checkpoints.csv'))
cybersec_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cybersec_results_with_parameters.csv'))
cybersec_checkpoint_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'duplimend_cybersec_checkpoints.csv'))

print(f"  Baseline: {len(baseline_df):,} rows")
print(f"  Cybersec: {len(cybersec_df):,} rows")

SEEDS = resolve_seeds(baseline_df, checkpoint_df, cybersec_df, cybersec_checkpoint_df)
print(f"\nSeeds under test ({len(SEEDS)}): {', '.join(str(s) for s in SEEDS)}")
print(f"Missing-seed policy: {'skip incomplete entries' if args.skip_incomplete else 'abort (strict)'}")

all_results = []

print("\n" + "="*80)
print("TESTING BASELINE DATASETS")
print("="*80)

baseline_datasets = baseline_df['Dataset'].unique()
print(f"\nTotal baseline datasets: {len(baseline_datasets)}")

real_life = ['BPIC2012', 'BPIC2013', 'BPIC2017', 'Environmental_Permits', 'Road_Traffic_Fines']

for i, dataset in enumerate(baseline_datasets):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(baseline_datasets)} baseline datasets...")

    if dataset in real_life:
        metrics_to_test = ['F-score']
    else:
        metrics_to_test = ['ARI', 'F-score']

    for metric in metrics_to_test:
        method_data = {}

        for method in ['PM-Label-Splitting', 'Label-Refinement', 'Unrefined']:
            method_rows = baseline_df[
                (baseline_df['Dataset'] == dataset) &
                (baseline_df['Method'] == method)
            ]

            if len(method_rows) == 0 or metric not in method_rows.columns:
                continue

            values = collect_seed_values(method_rows, metric, SEEDS, dataset, method)
            if values is not None:
                method_data[method] = values

        dup_rows = checkpoint_df[
            (checkpoint_df['Dataset'] == dataset) &
            (checkpoint_df['Checkpoint_Percentage'] == 100)
        ]

        if len(dup_rows) > 0 and metric in dup_rows.columns:
            values = collect_seed_values(dup_rows, metric, SEEDS, dataset, 'DupliMend',
                                         single_value=True)
            if values is not None:
                method_data['DupliMend'] = values

        if len(method_data) >= 2:
            result = run_significance_test_for_stream(dataset, metric, method_data)
            if result:
                result['Dataset_Type'] = 'Synthetic' if dataset not in ['BPIC2012', 'BPIC2013', 'BPIC2017', 'Environmental_Permits', 'Road_Traffic_Fines'] else 'Real-life'
                all_results.append(result)

print("\n" + "="*80)
print("TESTING CYBERSECURITY DATASETS")
print("="*80)

if 'Test_Log_ID' not in cybersec_df.columns:
    sys.exit("ERROR: 'Test_Log_ID' column missing from cybersec_results_with_parameters.csv.\n"
             "Log identities must come from the results file; they cannot be synthesised.")

test_log_ids = cybersec_df['Test_Log_ID'].unique()
print(f"\nTotal cybersec test logs: {len(test_log_ids)}")

for i, test_log_id in enumerate(test_log_ids):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(test_log_ids)} cybersec logs...")

    for metric in ['ARI', 'F-score']:
        method_data = {}

        if 'Test_Log_ID' in cybersec_df.columns:
            for method in ['PM-Label-Splitting', 'Label-Refinement', 'Unrefined']:
                method_rows = cybersec_df[
                    (cybersec_df['Test_Log_ID'] == test_log_id) &
                    (cybersec_df['Method'] == method)
                ]

                if len(method_rows) > 0 and metric in method_rows.columns:
                    values = collect_seed_values(method_rows, metric, SEEDS,
                                                 test_log_id, method)
                    if values is not None:
                        method_data[method] = values

            if 'Test_Log_ID' in cybersec_checkpoint_df.columns:
                dup_rows = cybersec_checkpoint_df[
                    (cybersec_checkpoint_df['Test_Log_ID'] == test_log_id) &
                    (cybersec_checkpoint_df['Checkpoint_Percentage'] == 100)
                ]

                if len(dup_rows) > 0 and metric in dup_rows.columns:
                    values = collect_seed_values(dup_rows, metric, SEEDS,
                                                 test_log_id, 'DupliMend',
                                                 single_value=True)
                    if values is not None:
                        method_data['DupliMend'] = values

        if len(method_data) >= 2:
            result = run_significance_test_for_stream(test_log_id, metric, method_data)
            if result:
                result['Dataset_Type'] = 'Cybersecurity'
                all_results.append(result)

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

if SKIPPED_ENTRIES:
    skipped_df = pd.DataFrame(SKIPPED_ENTRIES)
    skipped_file = os.path.join(OUTPUT_DIR, 'skipped_entries.csv')
    skipped_df.to_csv(skipped_file, index=False)
    print(f"\nWARNING: {len(SKIPPED_ENTRIES):,} dataset/method/metric entries were EXCLUDED "
          f"for missing seeds.")
    print(f"  Details: {skipped_file}")
    print(f"  The coverage below is therefore partial — it does not span every log.")
else:
    print(f"\nSeed matrix complete for every tested entry ({len(SEEDS)} seeds).")

results_df = pd.DataFrame(all_results)
output_file = os.path.join(OUTPUT_DIR, 'statistical_significance_results.csv')
results_df.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print(f"  Total rows: {len(results_df):,}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTOTAL TESTS RUN: {len(results_df):,}")
print(f"  Significant (p<0.05): {results_df['Significant'].sum():,} ({results_df['Significant'].sum()/len(results_df)*100:.1f}%)")
print(f"  Non-significant: {(~results_df['Significant']).sum():,}")

print("\n" + "-"*80)
print("BREAKDOWN BY DATASET TYPE AND METRIC")
print("-"*80)

for dtype in ['Synthetic', 'Real-life', 'Cybersecurity']:
    subset = results_df[results_df['Dataset_Type'] == dtype]
    if len(subset) == 0:
        continue

    print(f"\n{dtype}:")

    for metric in ['ARI', 'F-score']:
        metric_subset = subset[subset['Metric'] == metric]
        if len(metric_subset) == 0:
            print(f"  {metric}: NOT TESTED (expected - no ground truth)" if metric == 'ARI' and dtype == 'Real-life' else f"  {metric}: No data")
            continue

        print(f"  {metric}:")
        print(f"    Tests: {len(metric_subset):,}")
        print(f"    Significant (p<0.05): {metric_subset['Significant'].sum():,} ({metric_subset['Significant'].sum()/len(metric_subset)*100:.1f}%)")
        print(f"    Mean p-value: {metric_subset['Friedman_P_Value'].mean():.6f}")

        if len(metric_subset) > 0:
            print(f"    Top winners:")
            for winner, count in metric_subset['Winner'].value_counts().head(4).items():
                print(f"      {winner}: {count:,} ({count/len(metric_subset)*100:.1f}%)")

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE COMPLETE")
print("="*80)
