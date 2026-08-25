"""Load the published DupliMend configuration into the environment."""

import argparse
import json
import os
import shlex
import sys

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_configs")

GLOBAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "duplimend_global.json")
PER_GROUP_CONFIG_PATH = os.path.join(CONFIG_DIR, "duplimend_per_group.json")
BASELINE_GRIDS_PATH = os.path.join(CONFIG_DIR, "baseline_grids.json")
SEEDS_PATH = os.path.join(CONFIG_DIR, "seeds.json")


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_global_config():
    """Return the published global DupliMend configuration."""
    return _load(GLOBAL_CONFIG_PATH)


def load_per_group_configs():
    """Return the four group-level Bayesian-optimization winners."""
    return _load(PER_GROUP_CONFIG_PATH)


def load_baseline_grids():
    """Return the Label-Refinement and PM-Label-Splitting sweep grids."""
    return _load(BASELINE_GRIDS_PATH)


def load_seeds():
    """Return the seed manifest."""
    return _load(SEEDS_PATH)


def evaluation_seeds():
    """The ten seeds behind every reported mean/std."""
    return list(load_seeds()["evaluation_runs"]["seeds"])


def build_env(n_events=None, group=None):
    """Build the environment-variable mapping for the paper configuration."""
    config = load_global_config()
    env = dict(config["env_var_mapping"])
    env.pop("_comment", None)

    if group is not None:
        env = _apply_group_overrides(env, group)

    warmup_proportion = config["fixed_operational_params"]["warmup_proportion"]
    if n_events is not None:
        env["WARMUP_EVENTS"] = str(max(1, int(round(warmup_proportion * n_events))))

    return env


def _apply_group_overrides(env, group):
    """Overlay one group's winning parameters onto the global env mapping."""
    groups = load_per_group_configs()["groups"]
    match = next((g for g in groups if g["study_name"] == group), None)
    if match is None:
        names = ", ".join(g["study_name"] for g in groups)
        raise ValueError("Unknown group %r. Available groups: %s" % (group, names))

    params = match["params"]
    layer_size = params["layer_size"]
    if params["n_hidden_layers"] != 2:
        raise ValueError(
            "hidden_dims are only defined for n_hidden_layers == 2; group %r uses %d"
            % (group, params["n_hidden_layers"])
        )

    env = dict(env)
    env.update({
        "HIDDEN_DIMS": "[%d,%d]" % (2 * layer_size, layer_size),
        "LATENT_DIM": str(params["latent_dim"]),
        "BATCH_SIZE": str(params["batch_size"]),
        "DROPOUT_RATE": repr(params["dropout_rate"]),
        "LEARNING_RATE": repr(params["learning_rate"]),
        "NOISE_STD": repr(params["noise_level_sigma"]),
        "SPARSITY_LAMBDA": repr(params["sparsity_weight_lambda_s"]),
        "VARIANCE_THRESHOLD": repr(params["variance_threshold_epsilon_split"]),
        "MERGE_THRESHOLD": repr(params["merge_threshold_epsilon_merge"]),
        "CLUSTER_REG_WEIGHT": repr(params["cluster_reg_weight_lambda_c"]),
        "MEMORY_REGULARIZATION_WEIGHT": repr(params["memory_reg_weight_alpha"]),
        "CONTROL_FLOW_CONTEXT_WINDOW": str(params["context_window_size"]),
    })
    return env


def apply_paper_config(n_events=None, group=None, seed=None, override_existing=True):
    """Apply the paper configuration to os.environ."""
    if "config.config" in sys.modules:
        raise RuntimeError(
            "config.config is already imported; it read the environment at import "
            "time and will not see the paper configuration. Call "
            "apply_paper_config() before importing it, or use the "
            "`eval \"$(python -m config.paper_config --export)\"` shell form."
        )

    env = build_env(n_events=n_events, group=group)
    if seed is not None:
        env["DUPLIMEND_SEED"] = str(seed)

    for key, value in env.items():
        if override_existing or key not in os.environ:
            os.environ[key] = value
    return env


def _describe(env, config):
    lines = ["Published DupliMend configuration (%s)" % config["name"], ""]
    for key in sorted(env):
        lines.append("  %-32s %s" % (key, env[key]))

    if "WARMUP_EVENTS" not in env:
        lines += [
            "",
            "  WARMUP_EVENTS is not set: it depends on log length "
            "(warmup_proportion = %s)." % config["fixed_operational_params"]["warmup_proportion"],
            "  Pass --n-events N to resolve it, or let "
            "scripts/run_paper_experiments.py compute it per log.",
        ]

    diffs = {k: v for k, v in config["differs_from_repo_defaults"].items()
             if not k.startswith("_")}
    lines += ["", "Differs from the committed defaults in config/config.py:"]
    for key in sorted(diffs):
        repo_default, paper_value = diffs[key]
        lines.append("  %-32s %s -> %s" % (key, repo_default, paper_value))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true",
                        help="Emit shell `export` statements for eval.")
    parser.add_argument("--show", action="store_true",
                        help="Print the resolved configuration in readable form (default).")
    parser.add_argument("--json", action="store_true",
                        help="Emit the environment mapping as JSON.")
    parser.add_argument("--n-events", type=int, default=None,
                        help="Log length, used to resolve WARMUP_EVENTS from warmup_proportion.")
    parser.add_argument("--group", type=str, default=None,
                        help="Use one group's optimization winner instead of the global config.")
    parser.add_argument("--seeds", action="store_true",
                        help="Print the ten evaluation seeds, one per line, and exit.")
    args = parser.parse_args(argv)

    if args.seeds:
        for seed in evaluation_seeds():
            print(seed)
        return 0

    config = load_global_config()
    env = build_env(n_events=args.n_events, group=args.group)

    if args.export:
        for key in sorted(env):
            print("export %s=%s" % (key, shlex.quote(env[key])))
    elif args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        print(_describe(env, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
