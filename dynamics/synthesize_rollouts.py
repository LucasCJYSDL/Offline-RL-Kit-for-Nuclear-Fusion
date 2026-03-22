import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import pickle

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.utils.data_preprocess import get_raw_data, _post_process, _save_bounders, _dump_data
from dynamics_toolbox.utils.storage.model_storage import load_ensemble_from_parent_dir


def _load_models(parent_dir, device):
    ensemble = load_ensemble_from_parent_dir(parent_dir=parent_dir)
    models = ensemble.members
    for memb in models:
        memb.to(device)
        memb.eval()
    return models


def synthesize_rollouts(
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
    dataset = {
        "states": [],
        "actuators": [],
        "next_actuators": [],
        "next_states": [],
        "terminals": [],
        "shotnum": [],
        "time": [],
    }
    rl_data = {
        "observations": [],
        "pre_actions": [],
        "actions": [],
        "next_observations": [],
        "terminals": [],
        "time_step": [],
        "hidden_states": [],
        "traj_start_indices": [],
    }
    il_data = {
        "observations": [],
        "pre_actions": [],
        "actions": [],
        "next_observations": [],
        "terminals": [],
        "time_step": [],
        "traj_start_indices": [],
    }
    tracking_data = {}

    rl_shots_used = set()
    il_shots_used = set()
    tracking_shots_used = set()

    gen_models = _load_models(model_gen_dir, device)
    hidden_models = _load_models(model_hidden_dir, device)

    shot_num_list = list(offline_dst["ref_start_index"].keys())
    total_steps = len(offline_dst["shotnum"])

    for cur_shot in tqdm(shot_num_list):
        start_candidates = offline_dst["ref_start_index"].get(cur_shot, [])
        if len(start_candidates) == 0:
            continue

        t = start_candidates[0]
        if t >= total_steps:
            continue

        cur_state = torch.FloatTensor(offline_dst["observations"][t]).to(device)
        terminated = False

        if cur_shot in tracking_shot_list:
            tracking_data[cur_shot] = {
                "tracking_states": [],
                "tracking_next_states": [],
                "tracking_pre_actions": [],
                "tracking_actions": [],
            }
            tracking_shots_used.add(cur_shot)

        while t < total_steps and offline_dst["shotnum"][t] == cur_shot:
            pre_action = torch.FloatTensor(offline_dst["pre_actions"][t]).to(device)
            action_delta = torch.FloatTensor(offline_dst["action_deltas"][t]).to(device)

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

            dataset["states"].append(cur_state.cpu().numpy())
            dataset["actuators"].append(pre_action.cpu().numpy())
            dataset["next_actuators"].append(action_delta.cpu().numpy())
            dataset["next_states"].append(state_delta.cpu().numpy())
            dataset["shotnum"].append(cur_shot)
            dataset["time"].append(offline_dst["time"][t])

            if cur_shot in rl_shot_list:
                rl_shots_used.add(cur_shot)
                rl_data["observations"].append(cur_state.cpu().numpy())
                rl_data["pre_actions"].append(pre_action.cpu().numpy())
                rl_data["actions"].append(cur_action.cpu().numpy())
                rl_data["next_observations"].append(next_state.cpu().numpy())
                rl_data["terminals"].append(offline_dst["terminals"][t])
                rl_data["time_step"].append(offline_dst["time_step"][t])
                current_idx = len(rl_data["observations"]) - 1
                if offline_dst["time_step"][t] < 10:
                    rl_data["traj_start_indices"].append(current_idx)

            if cur_shot in il_shot_list:
                il_shots_used.add(cur_shot)
                il_data["observations"].append(cur_state.cpu().numpy())
                il_data["pre_actions"].append(pre_action.cpu().numpy())
                il_data["actions"].append(cur_action.cpu().numpy())
                il_data["next_observations"].append(next_state.cpu().numpy())
                il_data["terminals"].append(offline_dst["terminals"][t])
                il_data["time_step"].append(offline_dst["time_step"][t])
                current_idx = len(il_data["observations"]) - 1
                if offline_dst["time_step"][t] < 10:
                    il_data["traj_start_indices"].append(current_idx)

            if cur_shot in tracking_data and not terminated:
                tracking_data[cur_shot]["tracking_states"].append(cur_state.cpu().numpy())
                tracking_data[cur_shot]["tracking_next_states"].append(next_state.cpu().numpy())
                tracking_data[cur_shot]["tracking_pre_actions"].append(pre_action.cpu().numpy())
                tracking_data[cur_shot]["tracking_actions"].append(cur_action.cpu().numpy())

            if offline_dst["terminals"][t]:
                terminated = True
                if cur_shot in rl_shot_list:
                    s_id = len(rl_data["hidden_states"])
                    shot_states = np.array(rl_data["observations"][s_id:])
                    shot_pre_actions = np.array(rl_data["pre_actions"][s_id:])
                    shot_cur_actions = np.array(rl_data["actions"][s_id:])
                    hidden_input = torch.cat(
                        [
                            torch.FloatTensor(shot_states).to(device),
                            torch.FloatTensor(shot_pre_actions).to(device),
                            torch.FloatTensor(shot_cur_actions - shot_pre_actions).to(device),
                        ],
                        dim=-1,
                    )

                    memb_out_list = []
                    for memb in hidden_models:
                        memb.reset()
                        hidden_input_n = memb.normalizer.normalize(hidden_input, 0)
                        memb_out = memb.get_mem_out(hidden_input_n).unsqueeze(1)
                        memb_out_list.append(memb_out)
                    shot_hidden_states = torch.stack(memb_out_list, dim=1).cpu().tolist()
                    rl_data["hidden_states"].extend(shot_hidden_states)
                    if len(rl_data["hidden_states"]) != len(rl_data["observations"]):
                        raise ValueError(
                            "Hidden state count mismatch within shot: "
                            f"{len(rl_data['hidden_states'])} vs {len(rl_data['observations'])}"
                        )

            prev_terminal = offline_dst["terminals"][t]
            t += 1
            if t >= total_steps or offline_dst["shotnum"][t] != cur_shot:
                break
            if prev_terminal:
                cur_state = torch.FloatTensor(offline_dst["observations"][t]).to(device)
            else:
                cur_state = next_state

    for k in dataset:
        dataset[k] = np.array(dataset[k])
        print(k, dataset[k].shape)

    action_lb = offline_dst["action_lower_bounds"]
    action_ub = offline_dst["action_upper_bounds"]
    dataset["actuators"] = np.clip(dataset["actuators"], action_lb, action_ub)

    os.makedirs(data_dir, exist_ok=True)
    with h5py.File(os.path.join(data_dir, "full.hdf5"), "w") as hdf:
        for key, value in dataset.items():
            hdf.create_dataset(key, data=value)

    with open(os.path.join(raw_data_dir, "info.pkl"), "rb") as file:
        data_info = pickle.load(file)

    output_dir = os.path.dirname(rl_data_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "used_shots_info.txt"), "w") as f:
        f.write(f"RL shots used ({len(rl_shots_used)}):\n")
        for shot in sorted(rl_shots_used):
            f.write(f"  {shot}\n")
        f.write(f"\nIL shots used ({len(il_shots_used)}):\n")
        for shot in sorted(il_shots_used):
            f.write(f"  {shot}\n")
        f.write(f"\nTracking shots used ({len(tracking_shots_used)}):\n")
        for shot in sorted(tracking_shots_used):
            f.write(f"  {shot}\n")

    print(f"Total RL shots used: {len(rl_shots_used)}")
    print(f"Total IL shots used: {len(il_shots_used)}")
    print(f"Total tracking shots used: {len(tracking_shots_used)}")

    _post_process(rl_data, offline_dst)
    _post_process(il_data, offline_dst)
    _save_bounders(rl_data, offline_dst)
    _save_bounders(il_data, offline_dst)

    for shot_id in tracking_data:
        for k in tracking_data[shot_id]:
            tracking_data[shot_id][k] = np.array(tracking_data[shot_id][k])

        tracking_data[shot_id]["tracking_pre_actions"] = np.clip(
            tracking_data[shot_id]["tracking_pre_actions"],
            offline_dst["action_lower_bounds"],
            offline_dst["action_upper_bounds"],
        )
        tracking_data[shot_id]["tracking_actions"] = np.clip(
            tracking_data[shot_id]["tracking_actions"],
            offline_dst["action_lower_bounds"],
            offline_dst["action_upper_bounds"],
        )

    rl_len = len(rl_data["observations"])
    if not (
        rl_len
        == len(rl_data["actions"])
        == len(rl_data["next_observations"])
        == len(rl_data["terminals"])
        == len(rl_data["time_step"])
        == len(rl_data["hidden_states"])
    ):
        raise ValueError(
            "RL data length mismatch: "
            f"obs={rl_len}, actions={len(rl_data['actions'])}, "
            f"next_obs={len(rl_data['next_observations'])}, terminals={len(rl_data['terminals'])}, "
            f"time_step={len(rl_data['time_step'])}, hidden={len(rl_data['hidden_states'])}"
        )

    if len(il_data["traj_start_indices"]) > 0 and (
        np.max(il_data["traj_start_indices"]) >= len(il_data["observations"])
    ):
        raise ValueError(
            "IL traj_start_indices out of range: "
            f"max={np.max(il_data['traj_start_indices'])}, obs_len={len(il_data['observations'])}"
        )

    _dump_data(rl_data_path, rl_data)
    _dump_data(il_data_path, il_data)

    with h5py.File(tracking_data_path, "w") as hdf:
        for shot_id, shot_data in tracking_data.items():
            group = hdf.create_group(str(shot_id))
            for key, value in shot_data.items():
                group.create_dataset(key, data=value)

    with open(os.path.join(data_dir, "info.pkl"), "wb") as f:
        pickle.dump(data_info, f)


if __name__ == "__main__":
    raw_data_dir = "/zfsauton/project/fusion/data/organized/noshape_gas_benchmark"
    model_gen_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2"
    model_hidden_dir = "/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2"
    action_bound_file = "noshape_gas_flattop.yaml"
    state_bound_file = "noshape_gas_flattop.yaml"
    reference_shot = 161409
    warmup_steps = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rl_shot_list = list(range(reference_shot - 200, reference_shot + 200))
    il_shot_list = list(range(reference_shot - 200, reference_shot + 200))
    tracking_shot_list = [161409, 161410, 161412]

    synthesized_data_dir = raw_data_dir + "_synthesized"
    rl_data_path = synthesized_data_dir + "/rl_data.h5"
    il_data_path = synthesized_data_dir + "/il_data.h5"
    tracking_data_path = synthesized_data_dir + "/tracking_data.h5"

    all_shots = list(set(rl_shot_list) | set(il_shot_list) | set(tracking_shot_list))
    offline_dst = get_raw_data(
        raw_data_dir,
        action_bound_file,
        state_bound_file,
        all_shots,
        warmup_steps,
    )

    synthesize_rollouts(
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
