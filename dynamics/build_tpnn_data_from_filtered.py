"""Build a TPNN-ready dataset from the existing filtered RPNN dataset.

The synthesized transitions are kept unchanged.  The output directory is a
byte-for-byte copy of the source directory except for ``rl_data.h5``:

* the RPNN-specific ``hidden_states`` dataset is omitted;
* compact metadata for reconstructing each transition's complete raw history is
  added.

No Transformer history or KV cache is duplicated on disk.  At runtime, the
model input for transition ``i`` is reconstructed as

    [observation, pre_action, action - pre_action]

from ``history_start_indices[i]`` through ``i`` (inclusive).
"""

import argparse
import filecmp
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_SOURCE_DIR = Path(
    "/data/datasets/noshape_gas_benchmark_synthesized_dymodel_all_filter"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/datasets/noshape_gas_benchmark_synthesized_dymodel_all_filter_TPNN"
)
RL_DATA_FILENAME = "rl_data.h5"

REQUIRED_RL_KEYS = (
    "observations",
    "next_observations",
    "pre_actions",
    "actions",
    "terminals",
    "time_step",
    "traj_start_indices",
)

HISTORY_DATASETS = (
    "history_start_indices",
    "history_lengths",
    "episode_start_indices",
    "episode_end_indices",
)


def _validate_source(source_dir: Path, output_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Source and output directories must be different.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    source_rl_path = source_dir / RL_DATA_FILENAME
    if not source_rl_path.is_file():
        raise FileNotFoundError(f"Source RL data does not exist: {source_rl_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite it: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return source_rl_path


def _copy_non_rl_entries(source_dir: Path, temporary_output_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        if source_path.name == RL_DATA_FILENAME:
            continue
        destination_path = temporary_output_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, symlinks=True)
        elif source_path.is_symlink():
            destination_path.symlink_to(os.readlink(source_path))
        else:
            shutil.copy2(source_path, destination_path)


def _require_keys(hdf: h5py.File, keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in hdf]
    if missing:
        raise KeyError(f"Source rl_data.h5 is missing required datasets: {missing}")


def _build_history_metadata(
    time_steps: np.ndarray,
    terminals: np.ndarray,
    block_size: int,
):
    time_steps = np.asarray(time_steps, dtype=np.int64).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    if time_steps.shape != terminals.shape:
        raise ValueError(
            f"time_step and terminals shapes differ: "
            f"{time_steps.shape} vs {terminals.shape}"
        )
    if time_steps.size == 0:
        raise ValueError("rl_data.h5 contains no transitions.")
    if time_steps[0] != 0:
        raise ValueError("The first transition must have time_step=0.")
    if np.any(time_steps < 0):
        raise ValueError("time_step contains negative values.")

    episode_start_indices = np.flatnonzero(time_steps == 0).astype(np.int64)
    episode_end_indices = np.concatenate(
        [episode_start_indices[1:], np.asarray([time_steps.size], dtype=np.int64)]
    )
    if episode_start_indices.size == 0:
        raise ValueError("No episode starts were found.")
    if not np.all(terminals[episode_end_indices - 1]):
        bad = episode_end_indices[~terminals[episode_end_indices - 1]]
        raise ValueError(
            f"Episode boundaries without terminal=True were found near indices {bad[:10]}"
        )
    if int(terminals.sum()) != episode_start_indices.size:
        raise ValueError(
            f"terminal count ({int(terminals.sum())}) does not match episode count "
            f"({episode_start_indices.size})."
        )
    if np.any(terminals[:-1]) and not np.all(time_steps[1:][terminals[:-1]] == 0):
        raise ValueError("time_step does not reset to 0 after every terminal.")

    transition_indices = np.arange(time_steps.size, dtype=np.int64)
    history_start_indices = transition_indices - time_steps
    history_lengths = time_steps.astype(np.uint16)
    if np.any(history_start_indices < 0):
        raise ValueError("Reconstructed history points before the start of the dataset.")
    if not np.array_equal(
        history_start_indices + history_lengths.astype(np.int64),
        transition_indices,
    ):
        raise ValueError("History start/length metadata is internally inconsistent.")

    max_sequence_length = int(time_steps.max()) + 1
    if max_sequence_length > block_size:
        raise ValueError(
            f"Maximum sequence length {max_sequence_length} exceeds "
            f"TPNN block_size={block_size}."
        )

    return {
        "history_start_indices": history_start_indices,
        "history_lengths": history_lengths,
        "episode_start_indices": episode_start_indices,
        "episode_end_indices": episode_end_indices,
        "max_sequence_length": max_sequence_length,
    }


def _copy_rl_data(
    source_rl_path: Path,
    output_rl_path: Path,
    block_size: int,
) -> dict:
    with h5py.File(source_rl_path, "r") as source_hdf:
        _require_keys(source_hdf, REQUIRED_RL_KEYS)
        time_steps = source_hdf["time_step"][:]
        terminals = source_hdf["terminals"][:]
        metadata = _build_history_metadata(time_steps, terminals, block_size)

        observation_dim = int(source_hdf["observations"].shape[-1])
        next_observation_dim = int(source_hdf["next_observations"].shape[-1])
        action_dim = int(source_hdf["actions"].shape[-1])
        pre_action_dim = int(source_hdf["pre_actions"].shape[-1])
        if action_dim != pre_action_dim:
            raise ValueError(
                f"actions dim {action_dim} does not match pre_actions dim "
                f"{pre_action_dim}."
            )
        input_dim = observation_dim + pre_action_dim + action_dim

        with h5py.File(output_rl_path, "w") as output_hdf:
            for key, value in source_hdf.attrs.items():
                output_hdf.attrs[key] = value

            for key in source_hdf.keys():
                if key == "hidden_states":
                    continue
                source_hdf.copy(key, output_hdf, name=key)

            for key in HISTORY_DATASETS:
                values = metadata[key]
                chunk_size = min(max(1, values.shape[0]), 65536)
                output_hdf.create_dataset(
                    key,
                    data=values,
                    chunks=(chunk_size,),
                    dtype=values.dtype,
                )

            output_hdf.attrs["history_format"] = "full_raw_history_v1"
            output_hdf.attrs["history_includes_current"] = False
            output_hdf.attrs[
                "history_input_layout"
            ] = "observation,pre_action,action_minus_pre_action"
            output_hdf.attrs["source_generator"] = "RPNN"
            output_hdf.attrs["source_rl_data"] = str(source_rl_path.resolve())
            output_hdf.attrs["tpnn_block_size"] = int(block_size)
            output_hdf.attrs["max_sequence_length"] = metadata[
                "max_sequence_length"
            ]
            output_hdf.attrs["history_input_dim"] = input_dim
            output_hdf.attrs["history_output_dim"] = next_observation_dim

    return {
        "num_transitions": int(time_steps.shape[0]),
        "num_episodes": int(metadata["episode_start_indices"].shape[0]),
        "max_sequence_length": int(metadata["max_sequence_length"]),
        "input_dim": input_dim,
        "output_dim": next_observation_dim,
    }


def _files_are_identical(source_path: Path, output_path: Path) -> bool:
    if source_path.is_dir():
        comparison = filecmp.dircmp(source_path, output_path)
        if comparison.left_only or comparison.right_only or comparison.funny_files:
            return False
        if any(
            not filecmp.cmp(
                source_path / filename,
                output_path / filename,
                shallow=False,
            )
            for filename in comparison.common_files
        ):
            return False
        return all(
            _files_are_identical(
                source_path / directory,
                output_path / directory,
            )
            for directory in comparison.common_dirs
        )
    return filecmp.cmp(source_path, output_path, shallow=False)


def _validate_output(source_dir: Path, output_dir: Path, block_size: int) -> dict:
    source_entries = {path.name for path in source_dir.iterdir()}
    output_entries = {path.name for path in output_dir.iterdir()}
    if source_entries != output_entries:
        raise ValueError(
            f"Directory entries differ: source={sorted(source_entries)}, "
            f"output={sorted(output_entries)}"
        )

    for name in sorted(source_entries - {RL_DATA_FILENAME}):
        if not _files_are_identical(source_dir / name, output_dir / name):
            raise ValueError(f"Non-RL entry differs from the source: {name}")

    output_rl_path = output_dir / RL_DATA_FILENAME
    with h5py.File(output_rl_path, "r") as hdf:
        _require_keys(hdf, REQUIRED_RL_KEYS)
        _require_keys(hdf, HISTORY_DATASETS)
        if "hidden_states" in hdf:
            raise ValueError("Output rl_data.h5 still contains hidden_states.")
        metadata = _build_history_metadata(
            hdf["time_step"][:],
            hdf["terminals"][:],
            block_size,
        )
        for key in HISTORY_DATASETS:
            if not np.array_equal(hdf[key][:], metadata[key]):
                raise ValueError(f"Output history dataset failed validation: {key}")

        return {
            "num_transitions": int(hdf["observations"].shape[0]),
            "num_episodes": int(hdf["episode_start_indices"].shape[0]),
            "max_sequence_length": int(hdf.attrs["max_sequence_length"]),
            "input_dim": int(hdf.attrs["history_input_dim"]),
            "output_dim": int(hdf.attrs["history_output_dim"]),
        }


def build_dataset(source_dir: Path, output_dir: Path, block_size: int) -> dict:
    source_rl_path = _validate_source(source_dir, output_dir)
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    temporary_output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )

    try:
        _copy_non_rl_entries(source_dir, temporary_output_dir)
        build_info = _copy_rl_data(
            source_rl_path,
            temporary_output_dir / RL_DATA_FILENAME,
            block_size,
        )
        validation_info = _validate_output(
            source_dir,
            temporary_output_dir,
            block_size,
        )
        if build_info != validation_info:
            raise ValueError(
                f"Build and validation summaries differ: "
                f"{build_info} vs {validation_info}"
            )
        temporary_output_dir.rename(output_dir)
        return validation_info
    except Exception:
        if temporary_output_dir.exists():
            shutil.rmtree(temporary_output_dir)
        raise


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy the filtered RPNN dataset into a new TPNN-ready directory, "
            "replacing hidden_states with full-history metadata."
        )
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block_size", type=int, default=225)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.block_size < 1:
        raise ValueError(f"block_size must be positive, got {args.block_size}")
    summary = build_dataset(args.source_dir, args.output_dir, args.block_size)
    print(f"TPNN-ready dataset: {args.output_dir.resolve()}")
    print(f"Transitions: {summary['num_transitions']}")
    print(f"Episodes: {summary['num_episodes']}")
    print(f"Maximum sequence length: {summary['max_sequence_length']}")
    print(f"TPNN dimensions: {summary['input_dim']} -> {summary['output_dim']}")


if __name__ == "__main__":
    main()
