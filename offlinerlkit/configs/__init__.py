"""Packaged algorithm configuration helpers."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent

NORMAL_ALGOS = ("combo", "cql", "edac", "mobile", "mopo", "ppo", "td3bc")
BAO_ALGOS = ("iql", "mcq", "gcil", "bambrl", "rambo", "mppi", "rombrl")
ALGO_CONFIGS = {name: PACKAGE_DIR / f"{name}.json" for name in (*NORMAL_ALGOS, *BAO_ALGOS)}


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
    """Return the canonical algorithm names shipped with the package."""
    return tuple(sorted(ALGO_CONFIGS.keys()))


def get_default_algo_config_path(algo_name: str) -> str:
    """Return the packaged config path for one algorithm."""
    algo_key = algo_name.lower()
    if algo_key not in ALGO_CONFIGS:
        raise KeyError(f"Unknown algorithm '{algo_name}'")
    return str(ALGO_CONFIGS[algo_key])


def resolve_config_path(config_path: str = None, algo_name: str = None) -> Path:
    """Resolve an algorithm config path, falling back to the packaged file."""
    if config_path:
        candidate = Path(config_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if algo_name is None:
        raise ValueError("Either config_path or algo_name must be provided")

    algo_key = algo_name.lower()
    if algo_key not in ALGO_CONFIGS:
        raise KeyError(f"Unknown algorithm '{algo_name}'")
    return ALGO_CONFIGS[algo_key]


def load_algo_config(algo_name: str, config_path: str = None) -> Tuple[Dict, str]:
    """Load a single algorithm configuration section and its source path."""
    path = resolve_config_path(config_path, algo_name=algo_name)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("algo_name", algo_name.lower()) != algo_name.lower():
        raise ValueError(f"Configuration file '{path}' does not match algorithm '{algo_name}'")

    return config, str(path)


def apply_algo_defaults(
    parser: argparse.ArgumentParser,
    algo_name: str,
    config_path: str = None,
) -> Tuple[Dict, str]:
    """Load defaults from the algorithm config and install them on an argparse parser."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=config_path or get_default_algo_config_path(algo_name))
    known_args, _ = bootstrap.parse_known_args(sys.argv[1:])

    config, resolved_path = load_algo_config(algo_name, config_path=known_args.config)
    defaults = _normalize_default_keys(config.get("default_params", {}))
    if defaults:
        parser.set_defaults(**defaults)

    _ensure_argument(
        parser,
        "--config",
        type=str,
        default=resolved_path,
        help="Path to the algorithm JSON config file",
    )
    return config, resolved_path
