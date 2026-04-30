"""Benchmark configuration helpers."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import rl_preparation as rp

PACKAGE_DIR = Path(__file__).resolve().parent
BENCHMARK_CONFIGS_PATH = PACKAGE_DIR / "benchmark_configs.json"


def _ensure_argument(parser: argparse.ArgumentParser, *args, **kwargs) -> None:
    option_strings = getattr(parser, "_option_string_actions", {})
    if any(arg in option_strings for arg in args if isinstance(arg, str) and arg.startswith("-")):
        return
    parser.add_argument(*args, **kwargs)


def _normalize_default_keys(defaults: Dict) -> Dict:
    normalized = {}
    for key, value in defaults.items():
        normalized[key.replace("-", "_")] = value
    return normalized


def available_algo_names():
    """Return the algorithm names present in the benchmark configuration table."""
    configs, _ = load_benchmark_configs()
    algo_names = set()
    for task_configs in configs.values():
        algo_names.update(task_configs.keys())
    return tuple(sorted(algo_names))


def load_benchmark_configs(config_path: str = None) -> Tuple[Dict, str]:
    """Load the task-by-algorithm benchmark configuration table."""
    path = Path(config_path) if config_path else BENCHMARK_CONFIGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Benchmark configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs, str(path)


def available_benchmark_tasks(config_path: str = None):
    """Return the canonical task names with benchmark parameter entries."""
    configs, _ = load_benchmark_configs(config_path=config_path)
    return tuple(sorted(configs.keys()))


def load_benchmark_config(
    task: str,
    algo_name: str,
    config_path: str = None,
) -> Tuple[Dict, str]:
    """Load benchmark parameters for one task and one algorithm."""
    task_key = rp.resolve_task_name(task)
    algo_key = algo_name.lower()
    configs, resolved_path = load_benchmark_configs(config_path=config_path)

    if task_key not in configs:
        raise KeyError(f"Unknown benchmark task '{task}'")

    task_configs = configs[task_key]
    if algo_key not in task_configs:
        raise KeyError(f"Algorithm '{algo_name}' has no benchmark config for task '{task_key}'")

    return _normalize_default_keys(task_configs[algo_key]), resolved_path


def apply_benchmark_defaults(
    parser: argparse.ArgumentParser,
    algo_name: str,
    *,
    default_task: str = "temp",
    config_path: str = None,
) -> Tuple[Dict, str]:
    """Load task-specific benchmark defaults and install them on an argparse parser."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--task", type=str, default=default_task)
    bootstrap.add_argument(
        "--config",
        type=str,
        default=config_path or str(BENCHMARK_CONFIGS_PATH),
    )
    known_args, _ = bootstrap.parse_known_args(sys.argv[1:])

    defaults, resolved_path = load_benchmark_config(
        known_args.task,
        algo_name,
        config_path=known_args.config,
    )
    if defaults:
        parser.set_defaults(**defaults)

    _ensure_argument(
        parser,
        "--config",
        type=str,
        default=resolved_path,
        help="Path to the task-by-algorithm benchmark config JSON file",
    )
    return defaults, resolved_path
