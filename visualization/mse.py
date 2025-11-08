import os
import numpy as np
import pandas as pd


alg_base_dirs = {
    "PPO": "/home/scratch/jiayuc2/temp_1106/ppo_prof_control_zipfit_dens_optimized_lite_256/rotation/results/policy",
    "GCIL":"/home/scratch/jiayuc2/temp_1106/gcil/rotation/results/seed_1&timestamp_25-1101-050601",
    "MPPI":"/home/scratch/jiayuc2/temp_1106/mppi&horizon=40&num_samples=1000&lam=2.0&penalty_coef=2.5/rotation/results/seed_1&timestamp_25-1101-124903",
    "CQL": "/home/scratch/jiayuc2/temp_1106/cql/rotation/results/seed_1&timestamp_25-1029-214055",
    "EDAC": "/home/scratch/jiayuc2/temp_1106/edac&num_critics=50&eta=1.0/rotation/results/seed_1&timestamp_25-1029-215514",
    "TD3BC": "/home/scratch/jiayuc2/temp_1106/td3bc/rotation/results/seed_1&timestamp_25-1029-223217",
    "MCQ":"/home/scratch/jiayuc2/temp_1106/mcq/rotation/results/seed_1&timestamp_25-1030-140936",
    "COMBO": "/home/scratch/jiayuc2/temp_1106/combo/rotation/results/seed_1&timestamp_25-1030-121418",
    "IQL":"/home/scratch/jiayuc2/temp_1106/iql/rotation/results/seed_1&timestamp_25-1030-140800",
    "MOPO": "/home/scratch/jiayuc2/temp_1106/mopo&penalty_coef=2.5&rollout_length=5/rotation/results/seed_1&timestamp_25-1030-131737",                    
    "MOBILE": "/home/scratch/jiayuc2/temp_1106/mobile&penalty_coef=1.5&rollout_length=5&real_ratio=0.05/rotation/results/seed_1&timestamp_25-1104-044715",
    "BAMCTS":"/home/scratch/jiayuc2/temp_1106/bambrl_mcts&penalty_coef=1.5&rollout_length=3&real_ratio=0.05/rotation/results/seed_1&timestamp_25-1030-140607",
    "ROMBRL":"/home/scratch/jiayuc2/temp_1106/rombrl&grad_mode=1&sl_weight=1000.0&actor_training_epoch=10&onpolicy_rollout_batch_size=2500&onpolicy_rollout_length=10&small_traj_batch=False/rotation/results/seed_1&timestamp_25-1031-122233",
    "RAMBO":"/home/scratch/jiayuc2/temp_1106/rambo/rotation/results/seed_1&timestamp_25-1030-141605",
   
    
   


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

    results[algo] = {"mean": mean, "se": se}

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


df.index.name = "Algorithm"
save_path = "/home/scratch/jiayuc2/temp_1106/tracking_error_summary.csv"
df.to_csv(save_path)

print(f"\n Saved summary table to:\n{save_path}")
print(df.head(3))
