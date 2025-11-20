import os
import numpy as np
import pandas as pd


alg_base_dirs = {
    # "PPO": "/home/scratch/jiayuc2/temp_1106/ppo_prof_control_zipfit_dens_optimized_lite_256/rotation/results/policy",
    # "GCIL":"/home/scratch/jiayuc2/temp_1106/gcil/rotation/results/seed_1&timestamp_25-1101-050601",
    # "MPPI":"/home/scratch/jiayuc2/temp_1106/mppi&horizon=40&num_samples=1000&lam=2.0&penalty_coef=2.5/rotation/results/seed_1&timestamp_25-1101-124903",
    "CQL": "/home/scratch/jiayuc2/temp_1117/cql&cql_weight=5.0&temperature=1.0&max_q_backup=False&deterministic_backup=True&with_lagrange=False&lagrange_threshold=10.0&cql_alpha_lr=0.0003&num_repeat_actions=10/rotation/results/seed_1&timestamp_25-1116-124536",
    "EDAC": "/home/scratch/jiayuc2/temp_1117/edac&num_critics=50&eta=1.0/rotation/results/seed_1&timestamp_25-1031-105230",
    "TD3BC": "/home/scratch/jiayuc2/temp_1117/td3bc/rotation/results/seed_1&timestamp_25-1031-110054",
    # "MCQ":"/home/scratch/jiayuc2/temp_1106/mcq/rotation/results/seed_1&timestamp_25-1030-140936",
    "COMBO": "/home/scratch/jiayuc2/temp_1117/combo/rotation/results/seed_1&timestamp_25-1101-012025",
    # "IQL":"/home/scratch/jiayuc2/temp_1106/iql/rotation/results/seed_1&timestamp_25-1030-140800",
    "MOPO": "/home/scratch/jiayuc2/temp_1117/mopo&penalty_coef=2.5&rollout_length=5/rotation/results/seed_1&timestamp_25-1101-004542",                    
    "MOBILE": "/home/scratch/jiayuc2/temp_1117/mobile&penalty_coef=1.5&rollout_length=5&real_ratio=0.05/rotation/results/seed_1&timestamp_25-1116-132552",
    # "BAMCTS":"/home/scratch/jiayuc2/temp_1106/bambrl_mcts&penalty_coef=1.5&rollout_length=3&real_ratio=0.05/rotation/results/seed_1&timestamp_25-1030-140607",
    # "ROMBRL":"/home/scratch/jiayuc2/temp_1106/rombrl&grad_mode=1&sl_weight=1000.0&actor_training_epoch=10&onpolicy_rollout_batch_size=2500&onpolicy_rollout_length=10&small_traj_batch=False/rotation/results/seed_1&timestamp_25-1031-122233",
    # "RAMBO":"/home/scratch/jiayuc2/temp_1106/rambo/rotation/results/seed_1&timestamp_25-1030-141605",
   
    
   


}

results = {}

for algo, algo_dir in alg_base_dirs.items():
    print(f"Processing {algo} ...")
    mse_list = []

    for seed in range(10):
        seed_path = os.path.join(algo_dir, str(seed), "saved_data", "shot_data.npz")
        if not os.path.exists(seed_path):
            print(f"Missing: {seed_path}")
            continue

        try:
            data = np.load(seed_path, allow_pickle=True)
            for key in data.files:
                if key.startswith("shot_"):
                    shot_data = data[key].item()
                if "mse_reconstruct" in shot_data:
                    mse_list.append(shot_data["mse_reconstruct"])
                else:
                    print(f"'mse_reconstruct' not found in {seed_path}")
        except Exception as e:
            print(f"Error loading {seed_path}: {e}")

    if not mse_list:
        print(f"No valid mse data for {algo}")
        continue

    mse_array = np.stack(mse_list, axis=0)  # [30, 33]
    mean = mse_array.mean(axis=0)
    se = mse_array.std(axis=0, ddof=1) / np.sqrt(mse_array.shape[0])

    mean_all = mse_array.mean()                                  
    se_all   = mse_array.std(ddof=1) / np.sqrt(mse_array.size)  

    results[algo] = {"mean": mean, "se": se, "mean_all": mean_all, "se_all":se_all}

columns = []
for i in range(33):
    columns.append(f"ρ{i+1}_mean")
    columns.append(f"ρ{i+1}_SE")

df = pd.DataFrame(index=alg_base_dirs.keys(), columns=columns)

for algo in alg_base_dirs.keys():
    mean = results[algo]["mean"]
    se = results[algo]["se"]
    for i in range(33):
        df.loc[algo, f"ρ{i+1}_mean"] = mean[i]
        df.loc[algo, f"ρ{i+1}_SE"] = se[i]
    df.loc[algo, f"mean"] = results[algo]["mean_all"]
    df.loc[algo, f"SE"] = results[algo]["se_all"]


df.index.name = "Algorithm"
save_path = "/home/scratch/jiayuc2/temp_1117/tracking_error_summary.csv"
df.to_csv(save_path)

print(f"\n Saved summary table to:\n{save_path}")
print(df.head(3))
