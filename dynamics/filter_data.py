import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import shutil
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamics.synthesize_rollouts_all import (
    H5SequenceWriter,
    _compute_hidden_states_for_shot,
    _load_models,
)


RAW_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark"
SOURCE_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_dymodel_all"
OUTPUT_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_dymodel_all_filter"
MODEL_HIDDEN_DIR = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_synthesize_step2"

SOURCE_FULL_PATH = os.path.join(SOURCE_DATA_DIR, "full.hdf5")
OUTPUT_FULL_PATH = os.path.join(OUTPUT_DATA_DIR, "full.hdf5")
OUTPUT_RL_PATH = os.path.join(OUTPUT_DATA_DIR, "rl_data.h5")
OUTPUT_IL_PATH = os.path.join(OUTPUT_DATA_DIR, "il_data.h5")
OUTPUT_TRACKING_VAL_PATH = os.path.join(OUTPUT_DATA_DIR, "tracking_val.h5")
OUTPUT_TRACKING_TEST_PATH = os.path.join(OUTPUT_DATA_DIR, "tracking_test.h5")

MAX_SHOT_LENGTH = 205
MIN_SHOT_LENGTH = 155
DROP_HEAD = 5
DROP_TAIL = 20
VAL_SHOT_COUNT = 300
TEST_SHOT_COUNT = 300
SPLIT_SEED = 0

STATIC_KEYS = [
    "action_lower_bounds",
    "action_upper_bounds",
    "action_positions_lower_bounds",
    "action_positions_upper_bounds",
    "action_velocity_lower_bounds",
    "action_velocity_upper_bounds",
    "states_positions_lower_bounds",
    "states_positions_upper_bounds",
    "states_velocity_lower_bounds",
    "states_velocity_upper_bounds",
]


def _collect_shot_slices(shotnum: np.ndarray) -> List[Tuple[int, int, int]]:
    if shotnum.ndim != 1:
        raise ValueError(f"shotnum must be 1D, got shape {shotnum.shape}")

    shot_slices: List[Tuple[int, int, int]] = []
    start = 0
    total = shotnum.shape[0]
    while start < total:
        shot_id = int(shotnum[start])
        end = start + 1
        while end < total and int(shotnum[end]) == shot_id:
            end += 1
        shot_slices.append((shot_id, start, end))
        start = end
    return shot_slices


def _processed_bounds(start: int, end: int) -> Tuple[int, int, int]:
    raw_len = end - start
    truncated_len = min(raw_len, MAX_SHOT_LENGTH)
    if truncated_len < MIN_SHOT_LENGTH:
        return truncated_len, -1, -1

    keep_start = start + DROP_HEAD
    keep_end = start + truncated_len - DROP_TAIL
    if keep_end <= keep_start:
        return truncated_len, -1, -1

    return truncated_len, keep_start, keep_end


def _select_split_shots(valid_shots: List[int]) -> Tuple[List[int], List[int], List[int]]:
    if len(valid_shots) < VAL_SHOT_COUNT + TEST_SHOT_COUNT:
        raise ValueError(
            f"Not enough valid shots for split: {len(valid_shots)} < "
            f"{VAL_SHOT_COUNT + TEST_SHOT_COUNT}"
        )

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = np.array(sorted(valid_shots), dtype=np.int64)
    rng.shuffle(shuffled)

    val_shots = shuffled[:VAL_SHOT_COUNT].tolist()
    test_shots = shuffled[VAL_SHOT_COUNT:VAL_SHOT_COUNT + TEST_SHOT_COUNT].tolist()
    train_shots = shuffled[VAL_SHOT_COUNT + TEST_SHOT_COUNT:].tolist()
    return train_shots, val_shots, test_shots


def _build_processed_shot(hdf: h5py.File, keep_start: int, keep_end: int, shot_id: int) -> Dict[str, np.ndarray]:
    states = hdf["states"][keep_start:keep_end].astype(np.float32)
    pre_actions = hdf["actuators"][keep_start:keep_end].astype(np.float32)
    action_deltas = hdf["next_actuators"][keep_start:keep_end].astype(np.float32)
    state_deltas = hdf["next_states"][keep_start:keep_end].astype(np.float32)
    time = hdf["time"][keep_start:keep_end].astype(np.float32)

    n = states.shape[0]
    time_step = np.arange(n, dtype=np.int64)
    terminals = np.zeros(n, dtype=np.bool_)
    terminals[-1] = True

    action_lb = hdf["action_lower_bounds"][:]
    action_ub = hdf["action_upper_bounds"][:]
    pre_actions = np.clip(pre_actions, action_lb, action_ub)
    actions = np.clip(pre_actions + action_deltas, action_lb, action_ub)
    next_observations = states + state_deltas

    return {
        "states": states,
        "pre_actions": pre_actions,
        "action_deltas": action_deltas,
        "state_deltas": state_deltas,
        "actions": actions.astype(np.float32),
        "next_observations": next_observations.astype(np.float32),
        "time": time,
        "time_step": time_step,
        "terminals": terminals,
        "shotnum": np.full(n, shot_id, dtype=np.int64),
    }


def _write_static_keys(source_hdf: h5py.File, *writers: H5SequenceWriter) -> None:
    for key in STATIC_KEYS:
        value = source_hdf[key][:]
        for writer in writers:
            writer.write_static(key, value)


def _append_full_shot(full_writer: H5SequenceWriter, shot_data: Dict[str, np.ndarray]) -> None:
    full_writer.append("states", shot_data["states"])
    full_writer.append("actuators", shot_data["pre_actions"])
    full_writer.append("next_actuators", shot_data["action_deltas"])
    full_writer.append("next_states", shot_data["state_deltas"])
    full_writer.append("shotnum", shot_data["shotnum"])
    full_writer.append("time", shot_data["time"])


def _append_rl_il_shot(
    writer: H5SequenceWriter,
    shot_data: Dict[str, np.ndarray],
    traj_start_indices: List[int],
    sample_offset: int,
    hidden_states: np.ndarray = None,
) -> int:
    writer.append("observations", shot_data["states"])
    writer.append("pre_actions", shot_data["pre_actions"])
    writer.append("actions", shot_data["actions"])
    writer.append("next_observations", shot_data["next_observations"])
    writer.append("terminals", shot_data["terminals"])
    writer.append("time_step", shot_data["time_step"])
    if hidden_states is not None:
        writer.append("hidden_states", hidden_states.astype(np.float32))

    num_traj_starts = min(10, shot_data["states"].shape[0])
    traj_start_indices.extend(sample_offset + np.arange(num_traj_starts, dtype=np.int64))
    return shot_data["states"].shape[0]


def _write_tracking_group(hdf: h5py.File, shot_id: int, shot_data: Dict[str, np.ndarray]) -> None:
    group = hdf.create_group(str(shot_id))
    group.create_dataset("tracking_states", data=shot_data["states"].astype(np.float32))
    group.create_dataset("tracking_next_states", data=shot_data["next_observations"].astype(np.float32))
    group.create_dataset("tracking_pre_actions", data=shot_data["pre_actions"].astype(np.float32))
    group.create_dataset("tracking_actions", data=shot_data["actions"].astype(np.float32))


def main() -> None:
    if not os.path.exists(SOURCE_FULL_PATH):
        raise FileNotFoundError(f"Source full.hdf5 not found: {SOURCE_FULL_PATH}")

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_models = _load_models(MODEL_HIDDEN_DIR, device)

    with h5py.File(SOURCE_FULL_PATH, "r") as source_hdf:
        shot_slices = _collect_shot_slices(source_hdf["shotnum"][:].astype(np.int64))

        valid_slices: List[Tuple[int, int, int, int, int]] = []
        num_truncated = 0
        num_filtered_short = 0

        for shot_id, start, end in shot_slices:
            raw_len = end - start
            if raw_len > MAX_SHOT_LENGTH:
                num_truncated += 1

            truncated_len, keep_start, keep_end = _processed_bounds(start, end)
            if keep_start < 0:
                num_filtered_short += 1
                continue

            valid_slices.append((shot_id, start, end, keep_start, keep_end))

        valid_shots = [shot_id for shot_id, _, _, _, _ in valid_slices]
        train_shots, val_shots, test_shots = _select_split_shots(valid_shots)
        train_set = set(train_shots)
        val_set = set(val_shots)
        test_set = set(test_shots)

        full_writer = H5SequenceWriter(OUTPUT_FULL_PATH)
        rl_writer = H5SequenceWriter(OUTPUT_RL_PATH)
        il_writer = H5SequenceWriter(OUTPUT_IL_PATH)

        rl_traj_start_indices: List[int] = []
        il_traj_start_indices: List[int] = []
        rl_sample_count = 0
        il_sample_count = 0

        with h5py.File(OUTPUT_TRACKING_VAL_PATH, "w") as tracking_val_hdf, h5py.File(
            OUTPUT_TRACKING_TEST_PATH, "w"
        ) as tracking_test_hdf:
            for shot_id, _, _, keep_start, keep_end in tqdm(valid_slices, desc="Filtering shots"):
                shot_data = _build_processed_shot(source_hdf, keep_start, keep_end, shot_id)
                _append_full_shot(full_writer, shot_data)

                if shot_id in train_set:
                    hidden_states = _compute_hidden_states_for_shot(
                        shot_data["states"],
                        shot_data["pre_actions"],
                        shot_data["actions"],
                        shot_data["terminals"],
                        hidden_models,
                        device,
                    )
                    if hidden_states.shape[0] != shot_data["states"].shape[0]:
                        raise ValueError(
                            f"Hidden-state length mismatch for shot {shot_id}: "
                            f"{hidden_states.shape[0]} vs {shot_data['states'].shape[0]}"
                        )

                    rl_added = _append_rl_il_shot(
                        rl_writer,
                        shot_data,
                        rl_traj_start_indices,
                        rl_sample_count,
                        hidden_states=hidden_states,
                    )
                    il_added = _append_rl_il_shot(
                        il_writer,
                        shot_data,
                        il_traj_start_indices,
                        il_sample_count,
                    )
                    rl_sample_count += rl_added
                    il_sample_count += il_added
                elif shot_id in val_set:
                    _write_tracking_group(tracking_val_hdf, shot_id, shot_data)
                elif shot_id in test_set:
                    _write_tracking_group(tracking_test_hdf, shot_id, shot_data)
                else:
                    raise RuntimeError(f"Shot {shot_id} does not belong to any split.")

        _write_static_keys(source_hdf, full_writer, rl_writer, il_writer)
        rl_writer.write_static("traj_start_indices", np.asarray(rl_traj_start_indices, dtype=np.int64))
        il_writer.write_static("traj_start_indices", np.asarray(il_traj_start_indices, dtype=np.int64))

        full_writer.close()
        rl_writer.close()
        il_writer.close()

    info_src = os.path.join(RAW_DATA_DIR, "info.pkl")
    info_dst = os.path.join(OUTPUT_DATA_DIR, "info.pkl")
    if os.path.exists(info_src):
        shutil.copy2(info_src, info_dst)

    used_shots_info_path = os.path.join(OUTPUT_DATA_DIR, "used_shots_info.txt")
    with open(used_shots_info_path, "w") as f:
        f.write(f"Split seed: {SPLIT_SEED}\n")
        f.write(f"Max shot length before filter: {MAX_SHOT_LENGTH}\n")
        f.write(f"Min shot length before trim: {MIN_SHOT_LENGTH}\n")
        f.write(f"Drop head: {DROP_HEAD}\n")
        f.write(f"Drop tail: {DROP_TAIL}\n")
        f.write(f"Raw shots found: {len(shot_slices)}\n")
        f.write(f"Shots truncated at {MAX_SHOT_LENGTH}: {num_truncated}\n")
        f.write(f"Shots filtered for short length: {num_filtered_short}\n")
        f.write(f"Valid shots kept: {len(valid_slices)}\n")
        f.write(f"Train shots ({len(train_shots)}):\n")
        for shot_id in sorted(train_shots):
            f.write(f"  {shot_id}\n")
        f.write(f"\nTracking val shots ({len(val_shots)}):\n")
        for shot_id in sorted(val_shots):
            f.write(f"  {shot_id}\n")
        f.write(f"\nTracking test shots ({len(test_shots)}):\n")
        for shot_id in sorted(test_shots):
            f.write(f"  {shot_id}\n")

    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print(f"Raw shots found: {len(shot_slices)}")
    print(f"Shots truncated at {MAX_SHOT_LENGTH}: {num_truncated}")
    print(f"Shots filtered for short length: {num_filtered_short}")
    print(f"Valid shots kept: {len(valid_slices)}")
    print(f"Train shots: {len(train_shots)}")
    print(f"Tracking val shots: {len(val_shots)}")
    print(f"Tracking test shots: {len(test_shots)}")
    print(f"RL samples: {rl_sample_count}")
    print(f"IL samples: {il_sample_count}")


if __name__ == "__main__":
    main()
