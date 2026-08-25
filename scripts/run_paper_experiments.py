#!/usr/bin/env python3
"""Run the DupliMend experiments exactly as reported in the paper."""

import argparse
import csv
import datetime
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config.paper_config import (
    build_env,
    evaluation_seeds,
    load_global_config,
)

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "config", "paper_configs", "log_manifest.json")

RESULT_COLUMNS = [
    "Dataset",
    "Dataset_Type",
    "Method",
    "Seed",
    "Event_Count",
    "Expected_Entropy_Clusters",
    "Expected_Entropy_Labels",
    "NMI",
    "ARI",
    "Silhouette_Score",
    "Log_Precision",
    "Log_Fitness",
    "F-score",
    "Log_Precision_Unrefined",
    "Log_Fitness_Unrefined",
    "F-score_Unrefined",
    "Total_Events_Analyzed",
    "Total_Clusters",
    "Runtime_Seconds",
    "Run_Directory",
    "Evaluation_Json",
]


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    logs = manifest.get("logs", [])
    if not logs:
        raise SystemExit("Manifest %s contains no 'logs' entries." % path)
    return manifest, logs


def count_events(csv_path):
    """Number of data rows in a CSV, used to resolve WARMUP_EVENTS."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
        total = sum(1 for _ in handle)
    return max(0, total - 1)


def run_key(log_id, seed):
    return "%s::seed=%d" % (log_id, seed)


def load_ledger(path):
    """Read the resume ledger: run keys already completed successfully."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("status") == "ok" and entry.get("key"):
                done.add(entry["key"])
    return done


def append_ledger(path, entry):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def build_run_env(log, seed, n_events, run_dir, group=None):
    """Environment for one (log, seed) run: paper config + this log's wiring."""
    env = os.environ.copy()
    env.update(build_env(n_events=n_events, group=group))

    env["DUPLIMEND_SEED"] = str(seed)
    env["GROUND_TRUTH_PATH"] = os.path.join(REPO_ROOT, log["ground_truth"])
    env["DEFAULT_ACTIVITY"] = log["activity"]
    env["EVENT_ID_COLUMN"] = log["event_id_column"]
    env["CASE_ID_COLUMN"] = log["case_id_column"]
    env["CONTROL_FLOW_COLUMN"] = log["control_flow_column"]
    env["GROUND_TRUTH_ACTIVITY_COLUMN"] = log["ground_truth_activity_column"]

    env["TIMESTAMP_COLUMN"] = log["timestamp_column"]
    resource_column = log.get("resource_column")
    if resource_column:
        env["RESOURCE_COLUMN"] = resource_column
    else:
        env.pop("RESOURCE_COLUMN", None)

    env["DUPLIMEND_OUTPUT_DIR"] = run_dir
    env["EXPERIMENT_OUTPUT_DIR"] = run_dir
    env["EXPERIMENT_CONFIG_NAME"] = "paper_%s_seed%d" % (log["id"], seed)
    env["PYTHONHASHSEED"] = str(seed)

    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + existing if existing else "")

    for key, value in (log.get("config_overrides") or {}).items():
        env[str(key)] = str(value)
    return env


def find_evaluation_json(run_dir):
    """Locate the evaluation JSON main.py wrote for this run, if any."""
    matches = glob.glob(os.path.join(run_dir, "**", "evaluation_results_*.json"),
                        recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _display_path(path):
    """Repo-relative when inside the repo, absolute otherwise."""
    absolute = os.path.abspath(path)
    if absolute.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(absolute, REPO_ROOT)
    return absolute


def extract_row(log, seed, n_events, run_dir, eval_json_path, runtime_seconds):
    """Build one result row from an evaluation JSON. Returns None if unusable."""
    with open(eval_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    final = payload.get("final_results")
    if not final:
        return None

    refined = final.get("refined_model") or {}
    imprecise = final.get("imprecise_model") or {}

    return {
        "Dataset": log["id"],
        "Dataset_Type": log.get("dataset_type", ""),
        "Method": "DupliMend",
        "Seed": seed,
        "Event_Count": n_events,
        "Expected_Entropy_Clusters": final.get("expected_entropy_clusters_perspective"),
        "Expected_Entropy_Labels": final.get("expected_entropy_labels_perspective"),
        "NMI": final.get("normalized_mutual_info_score"),
        "ARI": final.get("adjusted_rand_score"),
        "Silhouette_Score": final.get("silhouette_score"),
        "Log_Precision": refined.get("precision"),
        "Log_Fitness": refined.get("fitness"),
        "F-score": refined.get("fscore"),
        "Log_Precision_Unrefined": imprecise.get("precision"),
        "Log_Fitness_Unrefined": imprecise.get("fitness"),
        "F-score_Unrefined": imprecise.get("fscore"),
        "Total_Events_Analyzed": final.get("total_events_analyzed"),
        "Total_Clusters": final.get("total_clusters"),
        "Runtime_Seconds": round(runtime_seconds, 3),
        "Run_Directory": _display_path(run_dir),
        "Evaluation_Json": _display_path(eval_json_path),
    }


def append_result(results_path, row):
    is_new = not os.path.exists(results_path)
    with open(results_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def execute_run(log, seed, output_dir, group, timeout, python_exe):
    """Run one (log, seed) pair. Returns (row_or_None, failure_reason_or_None)."""
    input_path = os.path.join(REPO_ROOT, log["input"])
    ground_truth_path = os.path.join(REPO_ROOT, log["ground_truth"])

    for label, path in (("input", input_path), ("ground truth", ground_truth_path)):
        if not os.path.exists(path):
            return None, "missing %s file: %s" % (label, path)

    run_dir = os.path.join(output_dir, "runs", log["id"], "seed_%d" % seed)
    os.makedirs(run_dir, exist_ok=True)

    n_events = count_events(input_path)
    env = build_run_env(log, seed, n_events, run_dir, group=group)

    command = [
        python_exe, os.path.join(REPO_ROOT, "main.py"),
        "--mode", "online_mode",
        "--seed", str(seed),
        "--input", input_path,
        "--output", os.path.join(run_dir, "refined_log.csv"),
    ]

    stdout_path = os.path.join(run_dir, "stdout.log")
    started = datetime.datetime.now()
    with open(stdout_path, "w", encoding="utf-8") as stdout_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                env=env,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "timed out after %ss (see %s)" % (timeout, stdout_path)
    runtime_seconds = (datetime.datetime.now() - started).total_seconds()

    if completed.returncode != 0:
        return None, "main.py exited with code %d (see %s)" % (completed.returncode, stdout_path)

    eval_json_path = find_evaluation_json(run_dir)
    if eval_json_path is None:
        return None, ("no evaluation_results_*.json produced; the run finished but "
                      "evaluation did not (see %s)" % stdout_path)

    row = extract_row(log, seed, n_events, run_dir, eval_json_path, runtime_seconds)
    if row is None:
        return None, "evaluation JSON %s has no final_results block" % eval_json_path
    return row, None


def write_provenance(output_dir, manifest_path, seeds, group, selected_logs):
    """Record what this sweep was, so results can be traced back to a config."""
    config = load_global_config()
    provenance = {
        "generated_at": datetime.datetime.now().isoformat(),
        "config_name": config["name"],
        "config_source": "config/paper_configs/duplimend_global.json",
        "group_override": group,
        "manifest": os.path.relpath(manifest_path, REPO_ROOT),
        "seeds": seeds,
        "logs": [log["id"] for log in selected_logs],
        "n_runs_planned": len(selected_logs) * len(seeds),
        "global_env": build_env(group=group),
        "resolved_env_per_log": {
            log["id"]: dict(build_env(group=group),
                            **{str(k): str(v)
                               for k, v in (log.get("config_overrides") or {}).items()})
            for log in selected_logs
        },
        "python": sys.version,
    }
    path = os.path.join(output_dir, "provenance.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Log manifest JSON (default: config/paper_configs/log_manifest.json).")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "paper_runs"),
                        help="Where run artefacts and the results CSV are written.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: the ten seeds from seeds.json).")
    parser.add_argument("--logs", nargs="+", default=None,
                        help="Restrict to these manifest log ids.")
    parser.add_argument("--group", default=None,
                        help="Use one optimization group's winning parameters instead "
                             "of the global paper configuration. Reported results use "
                             "the global configuration; this is for auditing only.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-run timeout in seconds (default: no timeout).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used for the child runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the runs that would be executed and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run pairs already recorded as completed in the ledger.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _, logs = load_manifest(args.manifest)

    if args.logs:
        wanted = set(args.logs)
        known = {log["id"] for log in logs}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit("Unknown log id(s): %s\nAvailable: %s"
                             % (", ".join(unknown), ", ".join(sorted(known))))
        logs = [log for log in logs if log["id"] in wanted]

    seeds = args.seeds if args.seeds is not None else evaluation_seeds()

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "duplimend_per_run_results.csv")
    ledger_path = os.path.join(args.output_dir, "ledger.jsonl")
    done = set() if args.force else load_ledger(ledger_path)

    pending = [(log, seed) for log in logs for seed in seeds
               if run_key(log["id"], seed) not in done]

    print("Logs:  %d" % len(logs))
    print("Seeds: %s" % ", ".join(str(s) for s in seeds))
    print("Runs:  %d planned, %d already complete, %d to execute"
          % (len(logs) * len(seeds), len(logs) * len(seeds) - len(pending), len(pending)))

    if args.dry_run:
        print("\n--dry-run: nothing executed. Runs that would execute:")
        for log, seed in pending:
            print("  %s  seed=%d" % (log["id"], seed))
        return 0

    provenance_path = write_provenance(args.output_dir, args.manifest, seeds,
                                       args.group, logs)
    print("Provenance: %s" % provenance_path)
    print("Results:    %s\n" % results_path)

    failures = []
    for index, (log, seed) in enumerate(pending, start=1):
        key = run_key(log["id"], seed)
        print("[%d/%d] %s seed=%d ..." % (index, len(pending), log["id"], seed),
              flush=True)

        row, failure = execute_run(log, seed, args.output_dir, args.group,
                                   args.timeout, args.python)

        timestamp = datetime.datetime.now().isoformat()
        if failure is not None:
            failures.append((key, failure))
            append_ledger(ledger_path, {"key": key, "status": "failed",
                                        "reason": failure, "at": timestamp})
            print("       FAILED: %s" % failure, flush=True)
            continue

        append_result(results_path, row)
        append_ledger(ledger_path, {"key": key, "status": "ok", "at": timestamp})
        print("       ok  ARI=%s  NMI=%s  F-score=%s"
              % (row["ARI"], row["NMI"], row["F-score"]), flush=True)

    print("\n%d/%d runs succeeded." % (len(pending) - len(failures), len(pending)))
    if failures:
        failures_path = os.path.join(args.output_dir, "failures.txt")
        with open(failures_path, "w", encoding="utf-8") as handle:
            for key, reason in failures:
                handle.write("%s\t%s\n" % (key, reason))
        print("%d run(s) FAILED. Details: %s" % (len(failures), failures_path))
        print("The results CSV is incomplete; do not aggregate it as if it were a "
              "full sweep. Fix the failures and rerun (completed runs are skipped).")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
