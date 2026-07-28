import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
import pickle
from typing import Dict, List

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.utils.data_preprocess import get_raw_data
from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir


RAW_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark"
MODEL_GEN_DIR = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2"
MODEL_HIDDEN_DIR = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_synthesize_step2"
SYNTHESIZED_DATA_DIR = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_dymodel_all"
ACTION_BOUND_FILE = "noshape_gas_flattop.yaml"
STATE_BOUND_FILE = "noshape_gas_flattop.yaml"
WARMUP_STEPS = 5

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

TRACKING_VAL_SHOT_LIST = [140359, 142816, 142898, 144975, 145051, 145180, 145980, 147705, 149617, 152597, 153832, 153937, 154052, 154331, 157073, 157250, 157838, 157985, 158113, 158993, 159004, 159457, 159604, 160496, 160517, 161534, 161540, 161587, 164447, 164481, 164507, 164654, 164659, 164874, 165353, 165720, 166094, 166119, 166585, 166739, 168912, 169434, 169473, 170212, 170255, 170322, 170410, 170445, 170887, 171262, 171989, 174612, 174925, 175348, 176005, 176125, 176140, 176236, 176344, 176445, 176739, 176749, 176925, 176975, 178513, 179027, 179443, 180226, 180391, 180396, 180647, 180721, 182706, 182776, 184457, 184794, 185127, 185140, 185153, 186133, 186459, 186603, 186610, 187106, 188894, 189056, 189204, 189365, 189842, 190299, 190464, 190764, 190786, 191177, 192017, 192768, 193108, 193841, 195751, 200398]
TRACKING_TEST_SHOT_LIST = [140003, 140447, 140475, 140615, 141398, 142638, 142804, 144251, 145179, 147130, 147436, 149115, 152922, 152937, 152996, 153359, 153428, 154380, 155342, 155504, 155590, 157125, 157483, 158603, 158654, 159126, 159313, 159447, 159714, 159732, 160511, 160715, 161096, 161383, 161490, 161584, 162795, 163498, 163544, 164696, 164873, 165014, 165116, 165516, 165679, 167572, 168907, 169502, 170090, 170253, 170276, 170288, 170316, 170438, 170757, 171954, 172536, 174074, 174684, 174718, 174821, 174823, 174936, 174937, 175005, 175281, 175521, 175735, 175851, 176141, 176290, 176826, 178907, 178921, 179448, 180206, 180243, 180631, 180660, 183151, 183981, 184028, 184031, 184460, 185949, 186067, 186458, 186524, 186653, 186804, 186982, 189001, 189381, 190500, 190642, 191471, 192913, 193769, 194237, 203021]


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


def _load_models(parent_dir, device):
    ensemble = load_ensemble_from_parent_dir(parent_dir=parent_dir)
    models = ensemble.members
    for memb in models:
        memb.to(device)
        memb.eval()
    return models


def _get_all_shots_from_raw_data(raw_data_dir):
    tr_path = os.path.join(raw_data_dir, "tr.hdf5")
    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"Cannot find raw file: {tr_path}")

    with h5py.File(tr_path, "r") as hdf:
        if "shotnum" not in hdf:
            raise KeyError("'shotnum' not found in tr.hdf5")
        shots = np.unique(hdf["shotnum"][:].astype(np.int64))

    return sorted(shots.tolist())


def _collect_shot_slices(shotnum: np.ndarray):
    if shotnum.ndim != 1:
        raise ValueError(f"shotnum must be 1D, got shape {shotnum.shape}")

    shot_slices = []
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


def _get_all_shots_from_full(full_data_path):
    with h5py.File(full_data_path, "r") as hdf:
        if "shotnum" not in hdf:
            raise KeyError("'shotnum' not found in full.hdf5")
        shots = np.unique(hdf["shotnum"][:].astype(np.int64))
    return sorted(shots.tolist())


def _write_static_keys_from_hdf(source_hdf, *writers):
    for key in STATIC_KEYS:
        if key not in source_hdf:
            raise KeyError(f"'{key}' not found in source hdf5")
        value = source_hdf[key][:]
        for writer in writers:
            writer.write_static(key, value)


def _write_static_keys_from_dict(source_dict, *writers):
    for key in STATIC_KEYS:
        value = source_dict[key]
        for writer in writers:
            writer.write_static(key, value)


def _copy_info_file(raw_data_dir, data_dir):
    src = os.path.join(raw_data_dir, "info.pkl")
    if not os.path.exists(src):
        return
    with open(src, "rb") as f:
        data_info = pickle.load(f)
    with open(os.path.join(data_dir, "info.pkl"), "wb") as f:
        pickle.dump(data_info, f)


def _resolve_policy_shot_lists(all_shots, tracking_val_shot_list, tracking_test_shot_list):
    all_shot_set = set(all_shots)
    tracking_val_shot_list = [s for s in tracking_val_shot_list if s in all_shot_set]
    tracking_test_shot_list = [s for s in tracking_test_shot_list if s in all_shot_set]

    overlap = set(tracking_val_shot_list) & set(tracking_test_shot_list)
    if overlap:
        raise ValueError("tracking_val_shot_list and tracking_test_shot_list must be disjoint.")

    holdout_shots = set(tracking_val_shot_list) | set(tracking_test_shot_list)
    rl_shot_list = [shot for shot in all_shots if shot not in holdout_shots]
    il_shot_list = [shot for shot in all_shots if shot not in holdout_shots]
    return rl_shot_list, il_shot_list, tracking_val_shot_list, tracking_test_shot_list


def _extract_member_memory(memb, net_input_n, device):
    """
    Compatibility for different dynamics-toolbox versions:
    - Newer/private branch: memb.get_mem_out(...)
    """

    memb_out = memb.get_mem_out(net_input_n)

    if not torch.is_tensor(memb_out):
        memb_out = torch.as_tensor(memb_out, dtype=torch.float32, device=device)
    else:
        memb_out = memb_out.to(device=device, dtype=torch.float32)

    # Normalize shape to [T, 1, H]
    if memb_out.dim() == 1:
        memb_out = memb_out.unsqueeze(0).unsqueeze(0)
    elif memb_out.dim() == 2:
        memb_out = memb_out.unsqueeze(1)

    return memb_out


def _compute_hidden_states_for_shot(
    shot_obs: np.ndarray,
    shot_pre_actions: np.ndarray,
    shot_actions: np.ndarray,
    shot_terminals: np.ndarray,
    hidden_models,
    device,
):
    """
    Compute hidden states segment-wise, resetting model memory at each terminal.
    Returns shape [T, num_models, 1, H] (or equivalent model memory shape after stack).
    """
    n = shot_obs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    terminal_indices = [i for i, t in enumerate(shot_terminals) if t]
    if len(terminal_indices) == 0 or terminal_indices[-1] != n - 1:
        terminal_indices.append(n - 1)

    all_hidden = []
    s = 0
    for e in terminal_indices:
        if e < s:
            continue

        seg_obs = shot_obs[s:e + 1]
        seg_pre = shot_pre_actions[s:e + 1]
        seg_act = shot_actions[s:e + 1]

        hidden_input = torch.cat(
            [
                torch.as_tensor(seg_obs, dtype=torch.float32, device=device),
                torch.as_tensor(seg_pre, dtype=torch.float32, device=device),
                torch.as_tensor(seg_act - seg_pre, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )

        memb_out_list = []
        for memb in hidden_models:
            memb.reset()
            hidden_input_n = memb.normalizer.normalize(hidden_input, 0)
            memb_out = _extract_member_memory(memb, hidden_input_n, device)
            memb_out_list.append(memb_out)

        # [T, num_models, ...]
        seg_hidden = torch.stack(memb_out_list, dim=1).detach().cpu().numpy()
        all_hidden.append(seg_hidden)
        s = e + 1

    if len(all_hidden) == 0:
        raise RuntimeError("Hidden state generation failed: no segments produced.")

    return np.concatenate(all_hidden, axis=0)


def synthesize_rollouts(
    offline_dst,
    raw_data_dir,
    model_gen_dir,
    model_hidden_dir,
    data_dir,
    rl_data_path,
    il_data_path,
    tracking_val_data_path,
    tracking_test_data_path,
    rl_shot_list,
    il_shot_list,
    tracking_val_shot_list,
    tracking_test_shot_list,
    device,
):
    rl_shot_set = set(rl_shot_list)
    il_shot_set = set(il_shot_list)
    tracking_val_shot_set = set(tracking_val_shot_list)
    tracking_test_shot_set = set(tracking_test_shot_list)

    overlap = tracking_val_shot_set & tracking_test_shot_set
    if overlap:
        raise ValueError(f"tracking_val_shot_list and tracking_test_shot_list overlap: {sorted(overlap)}")

    gen_models = _load_models(model_gen_dir, device)
    hidden_models = _load_models(model_hidden_dir, device)

    full_writer = H5SequenceWriter(os.path.join(data_dir, "full.hdf5"))
    rl_writer = H5SequenceWriter(rl_data_path)
    il_writer = H5SequenceWriter(il_data_path)

    # Track only small metadata in memory
    rl_traj_start_indices: List[int] = []
    il_traj_start_indices: List[int] = []
    rl_count = 0
    il_count = 0

    tracking_val_data = {}
    tracking_test_data = {}
    rl_shots_used = set()
    il_shots_used = set()
    tracking_val_shots_used = set()
    tracking_test_shots_used = set()

    action_lb = offline_dst["action_lower_bounds"]
    action_ub = offline_dst["action_upper_bounds"]

    shot_num_list = list(offline_dst["ref_start_index"].keys())
    total_steps = len(offline_dst["shotnum"])

    for cur_shot in tqdm(shot_num_list):
        start_candidates = offline_dst["ref_start_index"].get(cur_shot, [])
        if len(start_candidates) == 0:
            continue

        t = start_candidates[0]
        if t >= total_steps:
            continue

        cur_state = torch.as_tensor(offline_dst["observations"][t], dtype=torch.float32, device=device)
        terminated = False

        # Per-shot buffers (small, avoids huge global memory)
        shot_full_obs = []
        shot_full_pre = []
        shot_full_act_delta = []
        shot_full_obs_delta = []
        shot_full_term = []
        shot_full_shot = []
        shot_full_time = []

        shot_rl_obs = []
        shot_rl_pre = []
        shot_rl_act = []
        shot_rl_next = []
        shot_rl_term = []
        shot_rl_time = []

        shot_il_obs = []
        shot_il_pre = []
        shot_il_act = []
        shot_il_next = []
        shot_il_term = []
        shot_il_time = []

        if cur_shot in tracking_val_shot_set:
            tracking_val_data[cur_shot] = {
                "tracking_states": [],
                "tracking_next_states": [],
                "tracking_pre_actions": [],
                "tracking_actions": [],
            }
            tracking_val_shots_used.add(cur_shot)
        if cur_shot in tracking_test_shot_set:
            tracking_test_data[cur_shot] = {
                "tracking_states": [],
                "tracking_next_states": [],
                "tracking_pre_actions": [],
                "tracking_actions": [],
            }
            tracking_test_shots_used.add(cur_shot)

        while t < total_steps and offline_dst["shotnum"][t] == cur_shot:
            pre_action = torch.as_tensor(offline_dst["pre_actions"][t], dtype=torch.float32, device=device)
            action_delta = torch.as_tensor(offline_dst["action_deltas"][t], dtype=torch.float32, device=device)

            # Model-based next state synthesis
            net_input = torch.cat([cur_state, pre_action, action_delta], dim=-1).unsqueeze(0)
            ensemble_preds = 0.0
            with torch.no_grad():
                for memb in gen_models:
                    net_input_n = memb.normalizer.normalize(net_input, 0)
                    net_output_n, _ = memb.single_sample_output_from_torch(net_input_n)
                    net_output = memb.normalizer.unnormalize(net_output_n, 1)
                    ensemble_preds += net_output

            state_delta = ensemble_preds[0] / float(len(gen_models))
            next_state = cur_state + state_delta
            cur_action = pre_action + action_delta

            # Full synthesized dataset
            shot_full_obs.append(cur_state.detach().cpu().numpy())
            shot_full_pre.append(pre_action.detach().cpu().numpy())
            shot_full_act_delta.append(action_delta.detach().cpu().numpy())
            shot_full_obs_delta.append(state_delta.detach().cpu().numpy())
            shot_full_term.append(bool(offline_dst["terminals"][t]))
            shot_full_shot.append(int(cur_shot))
            shot_full_time.append(float(offline_dst["time"][t]))

            # RL dataset (all shots in this "all" script)
            if cur_shot in rl_shot_set:
                rl_shots_used.add(cur_shot)
                shot_rl_obs.append(cur_state.detach().cpu().numpy())
                shot_rl_pre.append(pre_action.detach().cpu().numpy())
                shot_rl_act.append(cur_action.detach().cpu().numpy())
                shot_rl_next.append(next_state.detach().cpu().numpy())
                shot_rl_term.append(bool(offline_dst["terminals"][t]))
                ts = int(offline_dst["time_step"][t])
                shot_rl_time.append(ts)
                if ts < 10:
                    rl_traj_start_indices.append(rl_count + len(shot_rl_obs) - 1)

            # IL dataset (all shots in this "all" script)
            if cur_shot in il_shot_set:
                il_shots_used.add(cur_shot)
                shot_il_obs.append(cur_state.detach().cpu().numpy())
                shot_il_pre.append(pre_action.detach().cpu().numpy())
                shot_il_act.append(cur_action.detach().cpu().numpy())
                shot_il_next.append(next_state.detach().cpu().numpy())
                shot_il_term.append(bool(offline_dst["terminals"][t]))
                ts = int(offline_dst["time_step"][t])
                shot_il_time.append(ts)
                if ts < 10:
                    il_traj_start_indices.append(il_count + len(shot_il_obs) - 1)

            # Tracking dataset
            if cur_shot in tracking_val_data and not terminated:
                tracking_val_data[cur_shot]["tracking_states"].append(cur_state.detach().cpu().numpy())
                tracking_val_data[cur_shot]["tracking_next_states"].append(next_state.detach().cpu().numpy())
                tracking_val_data[cur_shot]["tracking_pre_actions"].append(pre_action.detach().cpu().numpy())
                tracking_val_data[cur_shot]["tracking_actions"].append(cur_action.detach().cpu().numpy())
            if cur_shot in tracking_test_data and not terminated:
                tracking_test_data[cur_shot]["tracking_states"].append(cur_state.detach().cpu().numpy())
                tracking_test_data[cur_shot]["tracking_next_states"].append(next_state.detach().cpu().numpy())
                tracking_test_data[cur_shot]["tracking_pre_actions"].append(pre_action.detach().cpu().numpy())
                tracking_test_data[cur_shot]["tracking_actions"].append(cur_action.detach().cpu().numpy())

            if offline_dst["terminals"][t]:
                terminated = True

            prev_terminal = bool(offline_dst["terminals"][t])
            t += 1
            if t >= total_steps or offline_dst["shotnum"][t] != cur_shot:
                break

            if prev_terminal:
                cur_state = torch.as_tensor(offline_dst["observations"][t], dtype=torch.float32, device=device)
            else:
                cur_state = next_state

        # ---- Flush this shot to disk ----
        # full.hdf5
        if len(shot_full_obs) > 0:
            full_obs = np.asarray(shot_full_obs, dtype=np.float32)
            full_pre = np.asarray(shot_full_pre, dtype=np.float32)
            full_pre = np.clip(full_pre, action_lb, action_ub)
            full_writer.append("states", full_obs)
            full_writer.append("actuators", full_pre)
            full_writer.append("next_actuators", np.asarray(shot_full_act_delta, dtype=np.float32))
            full_writer.append("next_states", np.asarray(shot_full_obs_delta, dtype=np.float32))
            full_writer.append("shotnum", np.asarray(shot_full_shot, dtype=np.int64))
            full_writer.append("time", np.asarray(shot_full_time, dtype=np.float32))

        # RL (with hidden states)
        if len(shot_rl_obs) > 0:
            rl_obs = np.asarray(shot_rl_obs, dtype=np.float32)
            rl_pre = np.asarray(shot_rl_pre, dtype=np.float32)
            rl_act = np.asarray(shot_rl_act, dtype=np.float32)
            rl_next = np.asarray(shot_rl_next, dtype=np.float32)
            rl_term = np.asarray(shot_rl_term, dtype=np.bool_)
            rl_time = np.asarray(shot_rl_time, dtype=np.int64)

            rl_pre = np.clip(rl_pre, action_lb, action_ub)
            rl_act = np.clip(rl_act, action_lb, action_ub)

            rl_hidden = _compute_hidden_states_for_shot(
                rl_obs, rl_pre, rl_act, rl_term, hidden_models, device
            )
            if rl_hidden.shape[0] != rl_obs.shape[0]:
                raise ValueError(
                    f"RL hidden length mismatch for shot {cur_shot}: "
                    f"{rl_hidden.shape[0]} vs {rl_obs.shape[0]}"
                )

            rl_writer.append("observations", rl_obs)
            rl_writer.append("pre_actions", rl_pre)
            rl_writer.append("actions", rl_act)
            rl_writer.append("next_observations", rl_next)
            rl_writer.append("terminals", rl_term)
            rl_writer.append("time_step", rl_time)
            rl_writer.append("hidden_states", rl_hidden)

            rl_count += rl_obs.shape[0]

        # IL
        if len(shot_il_obs) > 0:
            il_obs = np.asarray(shot_il_obs, dtype=np.float32)
            il_pre = np.asarray(shot_il_pre, dtype=np.float32)
            il_act = np.asarray(shot_il_act, dtype=np.float32)
            il_next = np.asarray(shot_il_next, dtype=np.float32)
            il_term = np.asarray(shot_il_term, dtype=np.bool_)
            il_time = np.asarray(shot_il_time, dtype=np.int64)

            il_pre = np.clip(il_pre, action_lb, action_ub)
            il_act = np.clip(il_act, action_lb, action_ub)

            il_writer.append("observations", il_obs)
            il_writer.append("pre_actions", il_pre)
            il_writer.append("actions", il_act)
            il_writer.append("next_observations", il_next)
            il_writer.append("terminals", il_term)
            il_writer.append("time_step", il_time)

            il_count += il_obs.shape[0]

    # ---- Write static bounds / metadata ----
    # full bounds
    full_writer.write_static("action_lower_bounds", offline_dst["action_lower_bounds"])
    full_writer.write_static("action_upper_bounds", offline_dst["action_upper_bounds"])
    full_writer.write_static("action_positions_lower_bounds", offline_dst["action_positions_lower_bounds"])
    full_writer.write_static("action_positions_upper_bounds", offline_dst["action_positions_upper_bounds"])
    full_writer.write_static("action_velocity_lower_bounds", offline_dst["action_velocity_lower_bounds"])
    full_writer.write_static("action_velocity_upper_bounds", offline_dst["action_velocity_upper_bounds"])
    full_writer.write_static("states_positions_lower_bounds", offline_dst["states_positions_lower_bounds"])
    full_writer.write_static("states_positions_upper_bounds", offline_dst["states_positions_upper_bounds"])
    full_writer.write_static("states_velocity_lower_bounds", offline_dst["states_velocity_lower_bounds"])
    full_writer.write_static("states_velocity_upper_bounds", offline_dst["states_velocity_upper_bounds"])

    # RL/IL bounds + traj_start_indices
    for writer, traj_indices in [(rl_writer, rl_traj_start_indices), (il_writer, il_traj_start_indices)]:
        writer.write_static("action_lower_bounds", offline_dst["action_lower_bounds"])
        writer.write_static("action_upper_bounds", offline_dst["action_upper_bounds"])
        writer.write_static("action_positions_lower_bounds", offline_dst["action_positions_lower_bounds"])
        writer.write_static("action_positions_upper_bounds", offline_dst["action_positions_upper_bounds"])
        writer.write_static("action_velocity_lower_bounds", offline_dst["action_velocity_lower_bounds"])
        writer.write_static("action_velocity_upper_bounds", offline_dst["action_velocity_upper_bounds"])
        writer.write_static("states_positions_lower_bounds", offline_dst["states_positions_lower_bounds"])
        writer.write_static("states_positions_upper_bounds", offline_dst["states_positions_upper_bounds"])
        writer.write_static("states_velocity_lower_bounds", offline_dst["states_velocity_lower_bounds"])
        writer.write_static("states_velocity_upper_bounds", offline_dst["states_velocity_upper_bounds"])
        writer.write_static("traj_start_indices", np.asarray(traj_indices, dtype=np.int64))

    full_writer.close()
    rl_writer.close()
    il_writer.close()

    # Save tracking data
    for tracking_path, tracking_data in [
        (tracking_val_data_path, tracking_val_data),
        (tracking_test_data_path, tracking_test_data),
    ]:
        with h5py.File(tracking_path, "w") as hdf:
            for shot_id, shot_data in tracking_data.items():
                group = hdf.create_group(str(shot_id))
                for key in shot_data:
                    arr = np.asarray(shot_data[key], dtype=np.float32)
                    if key in ("tracking_pre_actions", "tracking_actions"):
                        arr = np.clip(arr, action_lb, action_ub)
                    group.create_dataset(key, data=arr)

    # Copy info.pkl
    with open(os.path.join(raw_data_dir, "info.pkl"), "rb") as f:
        data_info = pickle.load(f)
    with open(os.path.join(data_dir, "info.pkl"), "wb") as f:
        pickle.dump(data_info, f)

    # Save shot usage info
    with open(os.path.join(data_dir, "used_shots_info.txt"), "w") as f:
        f.write(f"RL shots used ({len(rl_shots_used)}):\n")
        for s in sorted(rl_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nIL shots used ({len(il_shots_used)}):\n")
        for s in sorted(il_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nTracking val shots used ({len(tracking_val_shots_used)}):\n")
        for s in sorted(tracking_val_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nTracking test shots used ({len(tracking_test_shots_used)}):\n")
        for s in sorted(tracking_test_shots_used):
            f.write(f"  {s}\n")

    print(f"Total RL shots used: {len(rl_shots_used)}")
    print(f"Total IL shots used: {len(il_shots_used)}")
    print(f"Total tracking val shots used: {len(tracking_val_shots_used)}")
    print(f"Total tracking test shots used: {len(tracking_test_shots_used)}")
    print(f"RL samples: {rl_count}, IL samples: {il_count}")


def synthesize_full_from_model(
    offline_dst,
    raw_data_dir,
    model_gen_dir,
    data_dir,
    device,
):
    """Generate only full.hdf5 for dynamics-model training."""
    gen_models = _load_models(model_gen_dir, device)
    full_writer = H5SequenceWriter(os.path.join(data_dir, "full.hdf5"))

    action_lb = offline_dst["action_lower_bounds"]
    action_ub = offline_dst["action_upper_bounds"]
    shot_num_list = list(offline_dst["ref_start_index"].keys())
    total_steps = len(offline_dst["shotnum"])
    shots_used = set()
    sample_count = 0

    for cur_shot in tqdm(shot_num_list, desc="Synthesizing full.hdf5"):
        start_candidates = offline_dst["ref_start_index"].get(cur_shot, [])
        if len(start_candidates) == 0:
            continue

        t = start_candidates[0]
        if t >= total_steps:
            continue

        cur_state = torch.as_tensor(offline_dst["observations"][t], dtype=torch.float32, device=device)

        shot_full_obs = []
        shot_full_pre = []
        shot_full_act_delta = []
        shot_full_obs_delta = []
        shot_full_shot = []
        shot_full_time = []

        while t < total_steps and offline_dst["shotnum"][t] == cur_shot:
            pre_action = torch.as_tensor(offline_dst["pre_actions"][t], dtype=torch.float32, device=device)
            action_delta = torch.as_tensor(offline_dst["action_deltas"][t], dtype=torch.float32, device=device)

            net_input = torch.cat([cur_state, pre_action, action_delta], dim=-1).unsqueeze(0)
            ensemble_preds = 0.0
            with torch.no_grad():
                for memb in gen_models:
                    net_input_n = memb.normalizer.normalize(net_input, 0)
                    net_output_n, _ = memb.single_sample_output_from_torch(net_input_n)
                    net_output = memb.normalizer.unnormalize(net_output_n, 1)
                    ensemble_preds += net_output

            state_delta = ensemble_preds[0] / float(len(gen_models))
            next_state = cur_state + state_delta

            shot_full_obs.append(cur_state.detach().cpu().numpy())
            shot_full_pre.append(pre_action.detach().cpu().numpy())
            shot_full_act_delta.append(action_delta.detach().cpu().numpy())
            shot_full_obs_delta.append(state_delta.detach().cpu().numpy())
            shot_full_shot.append(int(cur_shot))
            shot_full_time.append(float(offline_dst["time"][t]))

            prev_terminal = bool(offline_dst["terminals"][t])
            t += 1
            if t >= total_steps or offline_dst["shotnum"][t] != cur_shot:
                break

            if prev_terminal:
                cur_state = torch.as_tensor(offline_dst["observations"][t], dtype=torch.float32, device=device)
            else:
                cur_state = next_state

        if len(shot_full_obs) == 0:
            continue

        full_pre = np.asarray(shot_full_pre, dtype=np.float32)
        full_pre = np.clip(full_pre, action_lb, action_ub)
        full_writer.append("states", np.asarray(shot_full_obs, dtype=np.float32))
        full_writer.append("actuators", full_pre)
        full_writer.append("next_actuators", np.asarray(shot_full_act_delta, dtype=np.float32))
        full_writer.append("next_states", np.asarray(shot_full_obs_delta, dtype=np.float32))
        full_writer.append("shotnum", np.asarray(shot_full_shot, dtype=np.int64))
        full_writer.append("time", np.asarray(shot_full_time, dtype=np.float32))

        shots_used.add(cur_shot)
        sample_count += len(shot_full_obs)

    _write_static_keys_from_dict(offline_dst, full_writer)
    full_writer.close()
    _copy_info_file(raw_data_dir, data_dir)

    with open(os.path.join(data_dir, "used_shots_info.txt"), "w") as f:
        f.write(f"Full shots used ({len(shots_used)}):\n")
        for s in sorted(shots_used):
            f.write(f"  {s}\n")

    print(f"Full output directory: {data_dir}")
    print(f"Full shots used: {len(shots_used)}")
    print(f"Full samples: {sample_count}")


def build_policy_data_from_full(
    full_data_path,
    model_hidden_dir,
    data_dir,
    rl_data_path,
    il_data_path,
    tracking_val_data_path,
    tracking_test_data_path,
    rl_shot_list,
    il_shot_list,
    tracking_val_shot_list,
    tracking_test_shot_list,
    device,
):
    """Build RL/IL/tracking files from an existing full.hdf5."""
    hidden_models = _load_models(model_hidden_dir, device)
    os.makedirs(data_dir, exist_ok=True)

    rl_shot_set = set(rl_shot_list)
    il_shot_set = set(il_shot_list)
    tracking_val_shot_set = set(tracking_val_shot_list)
    tracking_test_shot_set = set(tracking_test_shot_list)

    overlap = tracking_val_shot_set & tracking_test_shot_set
    if overlap:
        raise ValueError(f"tracking_val_shot_list and tracking_test_shot_list overlap: {sorted(overlap)}")

    rl_writer = H5SequenceWriter(rl_data_path)
    il_writer = H5SequenceWriter(il_data_path)
    rl_traj_start_indices: List[int] = []
    il_traj_start_indices: List[int] = []
    rl_count = 0
    il_count = 0

    tracking_val_data = {}
    tracking_test_data = {}
    rl_shots_used = set()
    il_shots_used = set()
    tracking_val_shots_used = set()
    tracking_test_shots_used = set()

    with h5py.File(full_data_path, "r") as source_hdf:
        shot_slices = _collect_shot_slices(source_hdf["shotnum"][:].astype(np.int64))
        action_lb = source_hdf["action_lower_bounds"][:]
        action_ub = source_hdf["action_upper_bounds"][:]

        for shot_id, start, end in tqdm(shot_slices, desc="Building policy data"):
            states = source_hdf["states"][start:end].astype(np.float32)
            pre_actions = source_hdf["actuators"][start:end].astype(np.float32)
            action_deltas = source_hdf["next_actuators"][start:end].astype(np.float32)
            state_deltas = source_hdf["next_states"][start:end].astype(np.float32)
            if states.shape[0] == 0:
                continue

            pre_actions = np.clip(pre_actions, action_lb, action_ub)
            actions = np.clip(pre_actions + action_deltas, action_lb, action_ub).astype(np.float32)
            next_observations = (states + state_deltas).astype(np.float32)
            terminals = np.zeros(states.shape[0], dtype=np.bool_)
            terminals[-1] = True
            time_step = np.arange(states.shape[0], dtype=np.int64)

            if shot_id in rl_shot_set:
                hidden_states = _compute_hidden_states_for_shot(
                    states, pre_actions, actions, terminals, hidden_models, device
                )
                if hidden_states.shape[0] != states.shape[0]:
                    raise ValueError(
                        f"RL hidden length mismatch for shot {shot_id}: "
                        f"{hidden_states.shape[0]} vs {states.shape[0]}"
                    )
                rl_writer.append("observations", states)
                rl_writer.append("pre_actions", pre_actions)
                rl_writer.append("actions", actions)
                rl_writer.append("next_observations", next_observations)
                rl_writer.append("terminals", terminals)
                rl_writer.append("time_step", time_step)
                rl_writer.append("hidden_states", hidden_states.astype(np.float32))
                rl_traj_start_indices.extend(rl_count + np.arange(min(10, states.shape[0]), dtype=np.int64))
                rl_count += states.shape[0]
                rl_shots_used.add(shot_id)

            if shot_id in il_shot_set:
                il_writer.append("observations", states)
                il_writer.append("pre_actions", pre_actions)
                il_writer.append("actions", actions)
                il_writer.append("next_observations", next_observations)
                il_writer.append("terminals", terminals)
                il_writer.append("time_step", time_step)
                il_traj_start_indices.extend(il_count + np.arange(min(10, states.shape[0]), dtype=np.int64))
                il_count += states.shape[0]
                il_shots_used.add(shot_id)

            if shot_id in tracking_val_shot_set:
                tracking_val_data[shot_id] = {
                    "tracking_states": states,
                    "tracking_next_states": next_observations,
                    "tracking_pre_actions": pre_actions,
                    "tracking_actions": actions,
                }
                tracking_val_shots_used.add(shot_id)

            if shot_id in tracking_test_shot_set:
                tracking_test_data[shot_id] = {
                    "tracking_states": states,
                    "tracking_next_states": next_observations,
                    "tracking_pre_actions": pre_actions,
                    "tracking_actions": actions,
                }
                tracking_test_shots_used.add(shot_id)

        for writer, traj_indices in [(rl_writer, rl_traj_start_indices), (il_writer, il_traj_start_indices)]:
            _write_static_keys_from_hdf(source_hdf, writer)
            writer.write_static("traj_start_indices", np.asarray(traj_indices, dtype=np.int64))

    rl_writer.close()
    il_writer.close()

    for tracking_path, tracking_data in [
        (tracking_val_data_path, tracking_val_data),
        (tracking_test_data_path, tracking_test_data),
    ]:
        with h5py.File(tracking_path, "w") as hdf:
            for shot_id, shot_data in tracking_data.items():
                group = hdf.create_group(str(shot_id))
                for key, value in shot_data.items():
                    group.create_dataset(key, data=np.asarray(value, dtype=np.float32))

    with open(os.path.join(data_dir, "used_shots_info.txt"), "a") as f:
        f.write(f"\nRL shots used ({len(rl_shots_used)}):\n")
        for s in sorted(rl_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nIL shots used ({len(il_shots_used)}):\n")
        for s in sorted(il_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nTracking val shots used ({len(tracking_val_shots_used)}):\n")
        for s in sorted(tracking_val_shots_used):
            f.write(f"  {s}\n")
        f.write(f"\nTracking test shots used ({len(tracking_test_shots_used)}):\n")
        for s in sorted(tracking_test_shots_used):
            f.write(f"  {s}\n")

    print(f"Policy output directory: {data_dir}")
    print(f"Total RL shots used: {len(rl_shots_used)}")
    print(f"Total IL shots used: {len(il_shots_used)}")
    print(f"Total tracking val shots used: {len(tracking_val_shots_used)}")
    print(f"Total tracking test shots used: {len(tracking_test_shots_used)}")
    print(f"RL samples: {rl_count}, IL samples: {il_count}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "full_only", "policy_from_full"], default="all")
    parser.add_argument("--raw_data_dir", default=RAW_DATA_DIR)
    parser.add_argument("--model_gen_dir", default=MODEL_GEN_DIR)
    parser.add_argument("--model_hidden_dir", default=MODEL_HIDDEN_DIR)
    parser.add_argument("--data_dir", default=SYNTHESIZED_DATA_DIR)
    parser.add_argument("--full_data_path", default=None)
    parser.add_argument("--action_bound_file", default=ACTION_BOUND_FILE)
    parser.add_argument("--state_bound_file", default=STATE_BOUND_FILE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    rl_data_path = os.path.join(data_dir, "rl_data.h5")
    il_data_path = os.path.join(data_dir, "il_data.h5")
    tracking_val_data_path = os.path.join(data_dir, "tracking_val.h5")
    tracking_test_data_path = os.path.join(data_dir, "tracking_test.h5")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.stage == "policy_from_full":
        full_data_path = args.full_data_path or os.path.join(data_dir, "full.hdf5")
        all_shots = _get_all_shots_from_full(full_data_path)
        rl_shot_list, il_shot_list, tracking_val_shot_list, tracking_test_shot_list = _resolve_policy_shot_lists(
            all_shots, TRACKING_VAL_SHOT_LIST, TRACKING_TEST_SHOT_LIST
        )
        print(f"Total full.hdf5 shots found: {len(all_shots)}")
        print(
            f"RL shots: {len(rl_shot_list)} | IL shots: {len(il_shot_list)} | "
            f"Tracking val shots: {len(tracking_val_shot_list)} | Tracking test shots: {len(tracking_test_shot_list)}"
        )
        build_policy_data_from_full(
            full_data_path=full_data_path,
            model_hidden_dir=args.model_hidden_dir,
            data_dir=data_dir,
            rl_data_path=rl_data_path,
            il_data_path=il_data_path,
            tracking_val_data_path=tracking_val_data_path,
            tracking_test_data_path=tracking_test_data_path,
            rl_shot_list=rl_shot_list,
            il_shot_list=il_shot_list,
            tracking_val_shot_list=tracking_val_shot_list,
            tracking_test_shot_list=tracking_test_shot_list,
            device=device,
        )
        return

    all_shots = _get_all_shots_from_raw_data(args.raw_data_dir)
    rl_shot_list, il_shot_list, tracking_val_shot_list, tracking_test_shot_list = _resolve_policy_shot_lists(
        all_shots, TRACKING_VAL_SHOT_LIST, TRACKING_TEST_SHOT_LIST
    )

    print(f"Total raw shots found: {len(all_shots)}")
    print(
        f"RL shots: {len(rl_shot_list)} | IL shots: {len(il_shot_list)} | "
        f"Tracking val shots: {len(tracking_val_shot_list)} | Tracking test shots: {len(tracking_test_shot_list)}"
    )

    offline_dst = get_raw_data(
        args.raw_data_dir,
        args.action_bound_file,
        args.state_bound_file,
        all_shots,
        args.warmup_steps,
    )

    if args.stage == "full_only":
        synthesize_full_from_model(
            offline_dst=offline_dst,
            raw_data_dir=args.raw_data_dir,
            model_gen_dir=args.model_gen_dir,
            data_dir=data_dir,
            device=device,
        )
        return

    synthesize_rollouts(
        offline_dst=offline_dst,
        raw_data_dir=args.raw_data_dir,
        model_gen_dir=args.model_gen_dir,
        model_hidden_dir=args.model_hidden_dir,
        data_dir=data_dir,
        rl_data_path=rl_data_path,
        il_data_path=il_data_path,
        tracking_val_data_path=tracking_val_data_path,
        tracking_test_data_path=tracking_test_data_path,
        rl_shot_list=rl_shot_list,
        il_shot_list=il_shot_list,
        tracking_val_shot_list=tracking_val_shot_list,
        tracking_test_shot_list=tracking_test_shot_list,
        device=device,
    )


if __name__ == "__main__":
    main()
