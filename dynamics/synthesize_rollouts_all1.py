import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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


def synthesize_rollouts_all(
    offline_dst,
    raw_data_dir,
    model_gen_dir,
    model_hidden_dir,
    data_dir,
    rl_data_path,
    il_data_path,
    tracking_data_path,
    rl_shot_list,
    il_shot_list,
    tracking_shot_list,
    device,
):
    rl_shot_set = set(rl_shot_list)
    il_shot_set = set(il_shot_list)
    tracking_shot_set = set(tracking_shot_list)

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

    tracking_data = {}
    rl_shots_used = set()
    il_shots_used = set()
    tracking_shots_used = set()

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

        if cur_shot in tracking_shot_set:
            tracking_data[cur_shot] = {
                "tracking_states": [],
                "tracking_next_states": [],
                "tracking_pre_actions": [],
                "tracking_actions": [],
            }
            tracking_shots_used.add(cur_shot)

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
            if cur_shot in tracking_data and not terminated:
                tracking_data[cur_shot]["tracking_states"].append(cur_state.detach().cpu().numpy())
                tracking_data[cur_shot]["tracking_next_states"].append(next_state.detach().cpu().numpy())
                tracking_data[cur_shot]["tracking_pre_actions"].append(pre_action.detach().cpu().numpy())
                tracking_data[cur_shot]["tracking_actions"].append(cur_action.detach().cpu().numpy())

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
            full_writer.append("observations", full_obs)
            full_writer.append("pre_actions", full_pre)
            full_writer.append("action_deltas", np.asarray(shot_full_act_delta, dtype=np.float32))
            full_writer.append("observation_deltas", np.asarray(shot_full_obs_delta, dtype=np.float32))
            full_writer.append("terminals", np.asarray(shot_full_term, dtype=np.bool_))
            full_writer.append("shotnum", np.asarray(shot_full_shot, dtype=np.int64))

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
    with h5py.File(tracking_data_path, "w") as hdf:
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
        f.write(f"\nTracking shots used ({len(tracking_shots_used)}):\n")
        for s in sorted(tracking_shots_used):
            f.write(f"  {s}\n")

    print(f"Total RL shots used: {len(rl_shots_used)}")
    print(f"Total IL shots used: {len(il_shots_used)}")
    print(f"Total tracking shots used: {len(tracking_shots_used)}")
    print(f"RL samples: {rl_count}, IL samples: {il_count}")


if __name__ == "__main__":
    # Raw dataset and models
    raw_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark"
    model_gen_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2"
    model_hidden_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2"

    # Bounds
    action_bound_file = "noshape_gas_flattop.yaml"
    state_bound_file = "noshape_gas_flattop.yaml"
    warmup_steps = 5

    # Tracking list remains user-selectable
    tracking_shot_list = [161409, 161410, 161412]

    # Output directory
    synthesized_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark_synthesized_all"
    rl_data_path = os.path.join(synthesized_data_dir, "rl_data.h5")
    il_data_path = os.path.join(synthesized_data_dir, "il_data.h5")
    tracking_data_path = os.path.join(synthesized_data_dir, "tracking_data.h5")
    os.makedirs(synthesized_data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use ALL raw shots for RL and IL
    all_shots = _get_all_shots_from_raw_data(raw_data_dir)
    rl_shot_list = all_shots
    il_shot_list = all_shots

    # Safety filter for tracking shots
    all_shot_set = set(all_shots)
    tracking_shot_list = [s for s in tracking_shot_list if s in all_shot_set]

    print(f"Total raw shots found: {len(all_shots)}")
    print(f"RL shots: {len(rl_shot_list)} | IL shots: {len(il_shot_list)} | Tracking shots: {len(tracking_shot_list)}")

    offline_dst = get_raw_data(
        raw_data_dir,
        action_bound_file,
        state_bound_file,
        all_shots,
        warmup_steps,
    )

    synthesize_rollouts_all(
        offline_dst=offline_dst,
        raw_data_dir=raw_data_dir,
        model_gen_dir=model_gen_dir,
        model_hidden_dir=model_hidden_dir,
        data_dir=synthesized_data_dir,
        rl_data_path=rl_data_path,
        il_data_path=il_data_path,
        tracking_data_path=tracking_data_path,
        rl_shot_list=rl_shot_list,
        il_shot_list=il_shot_list,
        tracking_shot_list=tracking_shot_list,
        device=device,
    )
