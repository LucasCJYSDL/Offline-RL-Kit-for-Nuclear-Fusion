import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import shutil
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


RAW_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark"
SOURCE_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_dymodel_all"
OUTPUT_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_dymodel_all_filter"

SOURCE_FULL_PATH = os.path.join(SOURCE_DATA_DIR, "full.hdf5")
OUTPUT_FULL_PATH = os.path.join(OUTPUT_DATA_DIR, "full.hdf5")

MAX_SHOT_LENGTH = 205
MIN_SHOT_LENGTH = 155
DROP_HEAD = 5
DROP_TAIL = 20

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


class H5SequenceWriter:
    """Append-only writer for variable-length first dimension datasets."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.h5 = h5py.File(path, "w")
        self.lengths: Dict[str, int] = {}

    def append(self, key: str, arr: np.ndarray):
        arr = np.asarray(arr)
        if arr.shape[0] == 0:
            return

        if key not in self.h5:
            maxshape = (None,) + arr.shape[1:]
            chunk0 = min(max(1, arr.shape[0]), 4096)
            chunks = (chunk0,) + arr.shape[1:]
            self.h5.create_dataset(
                key,
                data=arr,
                maxshape=maxshape,
                chunks=chunks,
                dtype=arr.dtype,
            )
            self.lengths[key] = arr.shape[0]
        else:
            ds = self.h5[key]
            old_n = self.lengths[key]
            new_n = old_n + arr.shape[0]
            ds.resize((new_n,) + ds.shape[1:])
            ds[old_n:new_n] = arr
            self.lengths[key] = new_n

    def write_static(self, key: str, arr: np.ndarray):
        self.h5.create_dataset(key, data=np.asarray(arr))

    def close(self):
        self.h5.close()


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


def _processed_bounds(
    start: int,
    end: int,
    max_shot_length: int,
    min_shot_length: int,
    drop_head: int,
    drop_tail: int,
) -> Tuple[int, int, int]:
    raw_len = end - start
    truncated_len = min(raw_len, max_shot_length)
    if truncated_len < min_shot_length:
        return truncated_len, -1, -1

    keep_start = start + drop_head
    keep_end = start + truncated_len - drop_tail
    if keep_end <= keep_start:
        return truncated_len, -1, -1

    return truncated_len, keep_start, keep_end


def _build_processed_shot(hdf: h5py.File, keep_start: int, keep_end: int, shot_id: int) -> Dict[str, np.ndarray]:
    states = hdf["states"][keep_start:keep_end].astype(np.float32)
    pre_actions = hdf["actuators"][keep_start:keep_end].astype(np.float32)
    action_deltas = hdf["next_actuators"][keep_start:keep_end].astype(np.float32)
    state_deltas = hdf["next_states"][keep_start:keep_end].astype(np.float32)
    time = hdf["time"][keep_start:keep_end].astype(np.float32)

    action_lb = hdf["action_lower_bounds"][:]
    action_ub = hdf["action_upper_bounds"][:]
    pre_actions = np.clip(pre_actions, action_lb, action_ub)

    return {
        "states": states,
        "pre_actions": pre_actions,
        "action_deltas": action_deltas,
        "state_deltas": state_deltas,
        "time": time,
        "shotnum": np.full(states.shape[0], shot_id, dtype=np.int64),
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_full_path", default=SOURCE_FULL_PATH)
    parser.add_argument("--output_data_dir", default=OUTPUT_DATA_DIR)
    parser.add_argument("--raw_data_dir", default=RAW_DATA_DIR)
    parser.add_argument("--max_shot_length", type=int, default=MAX_SHOT_LENGTH)
    parser.add_argument("--min_shot_length", type=int, default=MIN_SHOT_LENGTH)
    parser.add_argument("--drop_head", type=int, default=DROP_HEAD)
    parser.add_argument("--drop_tail", type=int, default=DROP_TAIL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_full_path = args.source_full_path
    output_data_dir = args.output_data_dir
    output_full_path = os.path.join(output_data_dir, "full.hdf5")

    if not os.path.exists(source_full_path):
        raise FileNotFoundError(f"Source full.hdf5 not found: {source_full_path}")

    os.makedirs(output_data_dir, exist_ok=True)

    with h5py.File(source_full_path, "r") as source_hdf:
        shot_slices = _collect_shot_slices(source_hdf["shotnum"][:].astype(np.int64))

        valid_slices: List[Tuple[int, int, int, int, int]] = []
        num_truncated = 0
        num_filtered_short = 0

        for shot_id, start, end in shot_slices:
            raw_len = end - start
            if raw_len > args.max_shot_length:
                num_truncated += 1

            truncated_len, keep_start, keep_end = _processed_bounds(
                start,
                end,
                args.max_shot_length,
                args.min_shot_length,
                args.drop_head,
                args.drop_tail,
            )
            if keep_start < 0:
                num_filtered_short += 1
                continue

            valid_slices.append((shot_id, start, end, keep_start, keep_end))

        valid_shots = [shot_id for shot_id, _, _, _, _ in valid_slices]
        kept_sample_count = 0
        full_writer = H5SequenceWriter(output_full_path)

        for shot_id, _, _, keep_start, keep_end in tqdm(valid_slices, desc="Filtering full.hdf5"):
            shot_data = _build_processed_shot(source_hdf, keep_start, keep_end, shot_id)
            _append_full_shot(full_writer, shot_data)
            kept_sample_count += shot_data["states"].shape[0]

        _write_static_keys(source_hdf, full_writer)
        full_writer.close()

    source_data_dir = os.path.dirname(source_full_path)
    info_src = os.path.join(source_data_dir, "info.pkl")
    if not os.path.exists(info_src):
        info_src = os.path.join(args.raw_data_dir, "info.pkl")
    info_dst = os.path.join(output_data_dir, "info.pkl")
    if os.path.exists(info_src):
        shutil.copy2(info_src, info_dst)

    used_shots_info_path = os.path.join(output_data_dir, "used_shots_info.txt")
    with open(used_shots_info_path, "w") as f:
        f.write(f"Source full.hdf5: {source_full_path}\n")
        f.write(f"Output full.hdf5: {output_full_path}\n")
        f.write(f"Max shot length before filter: {args.max_shot_length}\n")
        f.write(f"Min shot length before trim: {args.min_shot_length}\n")
        f.write(f"Drop head: {args.drop_head}\n")
        f.write(f"Drop tail: {args.drop_tail}\n")
        f.write(f"Raw shots found: {len(shot_slices)}\n")
        f.write(f"Shots truncated at {args.max_shot_length}: {num_truncated}\n")
        f.write(f"Shots filtered for short length: {num_filtered_short}\n")
        f.write(f"Valid shots kept: {len(valid_slices)}\n")
        f.write(f"Kept samples: {kept_sample_count}\n")
        f.write(f"Kept shots ({len(valid_shots)}):\n")
        for shot_id in sorted(valid_shots):
            f.write(f"  {shot_id}\n")

    print(f"Output directory: {output_data_dir}")
    print(f"Raw shots found: {len(shot_slices)}")
    print(f"Shots truncated at {args.max_shot_length}: {num_truncated}")
    print(f"Shots filtered for short length: {num_filtered_short}")
    print(f"Valid shots kept: {len(valid_slices)}")
    print(f"Kept samples: {kept_sample_count}")


if __name__ == "__main__":
    main()
