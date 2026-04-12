import os
import numpy as np
import pandas as pd


alg_base_dirs = {
    "PPO": "/export/pgs/fuyang/result_329/ppo&clip_range=0.148&total_timesteps=800000&learning_rate=0.003&gae_lambda=0.98&gamma=0.952&batch_size=2048&n_steps=2048/rotation/test/results/seed_1&timestamp_26-0329-215953",
    "PPO_256": "/export/pgs/fuyang/result_329/ppo&clip_range=0.148&total_timesteps=800000&learning_rate=0.003&gae_lambda=0.98&gamma=0.952&batch_size=2048&n_steps=2048/rotation/test/results/seed_1&timestamp_26-0331-113525",
    #"GCIL":"/home/scratch/jiayuc2/bao/Eval_optuna_out/rotation/gcil&batch_size=64/rotation/results/seed_1&timestamp_25-1130-105233",
    # # "MPPI":"/home/scratch/jiayuc2/temp_1106/mppi&horizon=40&num_samples=1000&lam=2.0&penalty_coef=2.5/rotation/results/seed_1&timestamp_25-1101-124903",
    "CQL&cql_weight=1.0&temperature=1.0": "/export/pgs/fuyang/result_329/cql&cql_weight=1.0&temperature=1.0/rotation/test/results/seed_1&timestamp_26-0330-114138",
    "CQL&cql_weight=2.0&temperature=1.0": "/export/pgs/fuyang/result_329/cql&cql_weight=2.0&temperature=1.0/rotation/test/results/seed_1&timestamp_26-0330-114930",
    "CQL&cql_weight=5.0&temperature=1.0": "/export/pgs/fuyang/result_329/cql&cql_weight=5.0&temperature=1.0/rotation/test/results/seed_1&timestamp_26-0329-223152",
    "EDAC&num_critics=50&eta=5.0": "/export/pgs/fuyang/result_329/edac&num_critics=50&eta=5.0/rotation/test/results/seed_1&timestamp_26-0329-232941",
    "EDAC&num_critics=10&eta=0.1": "/export/pgs/fuyang/result_329/edac&num_critics=10&eta=0.1/rotation/test/results/seed_1&timestamp_26-0330-181254",
    "TD3BC&alpha=1.5": "/export/pgs/fuyang/result_329/td3bc&alpha=1.5/rotation/test/results/seed_1&timestamp_26-0329-214831",
    # "MCQ": "/zfsauton2/home/jiayuc2/bao/synthesize_test1/mcq&lmbda=0.809154&num_sampled_actions=20/rotation/results/seed_1&timestamp_26-0305-094807",
    "COMBO&cql_weight=1&rollout_length=10": "/export/pgs/fuyang/result_329/combo&cql_weight=0.1&rollout_length=10/rotation/test/results/seed_1&timestamp_26-0329-144753",
    #"IQL":"/home/scratch/jiayuc2/bao/Eval_optuna_out/rotation/iql&expectile=0.508981&temperature=1.26298/rotation/results/seed_1&timestamp_25-1130-110438",
    "MOPO&penalty_coef=0.5&rollout_length=7": "/export/pgs/fuyang/result_329/mopo&penalty_coef=0.5&rollout_length=7/rotation/test/results/seed_1&timestamp_26-0330-001453",
    "MOPO&penalty_coef=2.5&rollout_length=5": "/export/pgs/fuyang/result_329/mopo&penalty_coef=2.5&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0329-145405",
    "MOPO&penalty_coef=2.5&rollout_length=5": "/export/pgs/fuyang/result_329/mopo&penalty_coef=2.5&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0329-145405",                    
    "MOBILE&penalty_coef=1.5&rollout_length=1": "/export/pgs/fuyang/result_329/mobile&penalty_coef=1.5&rollout_length=1/rotation/test/results/seed_1&timestamp_26-0329-145725",
    "MOBILE&penalty_coef=1.5&rollout_length=5": "/export/pgs/fuyang/result_329/mobile&penalty_coef=1.5&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0330-115746",
    "MOBILE&penalty_coef=2.0&rollout_length=5": "/export/pgs/fuyang/result_329/mobile&penalty_coef=2.0&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0331-194559",
    "MOBILE&penalty_coef=5.0&rollout_length=5": "/export/pgs/fuyang/result_329/mobile&penalty_coef=5.0&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0331-194559"
    #"BAMCTS":"/home/scratch/jiayuc2/bao/Eval_optuna/rotation/bambrl_mcts&rollout_length=2&penalty_coef=0.7067621971505375&use_ba=False&search_alpha=0.8/rotation/results/seed_1&timestamp_25-1212-144528",
    # "ROMBRL":"/home/scratch/jiayuc2/temp_1106/rombrl&grad_mode=1&sl_weight=1000.0&actor_training_epoch=10&onpolicy_rollout_batch_size=2500&onpolicy_rollout_length=10&small_traj_batch=False/rotation/results/seed_1&timestamp_25-1031-122233",
    # "RAMBO":"/home/scratch/jiayuc2/bao/Eval_optuna/rotation/rambo&rollout_length=2&adv_weight=6.471144404147533e-05/rotation/results/seed_1&timestamp_25-1212-145936",
   
    # "PPO": "/export/pgs/fuyang/result_45/ppo&clip_range=0.148&total_timesteps=800000&learning_rate=0.003&gae_lambda=0.98&gamma=0.952&batch_size=2048&n_steps=2048/dens/test/results/seed_1&timestamp_26-0330-155626",
    # "PPO_256": "/export/pgs/fuyang/result_45/ppo&clip_range=0.148&total_timesteps=800000&learning_rate=0.003&gae_lambda=0.98&gamma=0.952&batch_size=2048&n_steps=2048/dens/test/results/seed_1&timestamp_26-0404-153243",
    # #"GCIL":"/home/scratch/jiayuc2/bao/Eval_optuna_out/rotation/gcil&batch_size=64/rotation/results/seed_1&timestamp_25-1130-105233",
    # # "MPPI":"/home/scratch/jiayuc2/temp_1106/mppi&horizon=40&num_samples=1000&lam=2.0&penalty_coef=2.5/rotation/results/seed_1&timestamp_25-1101-124903",
    # "CQL&cql_weight=2.0&temperature=2.0": "/export/pgs/fuyang/result_45/cql&cql_weight=2.0&temperature=2.0/dens/test/results/seed_1&timestamp_26-0403-141657",
    # "CQL&cql_weight=5.0&temperature=1.0": "/export/pgs/fuyang/result_45/cql&cql_weight=5.0&temperature=1.0/dens/test/results/seed_1&timestamp_26-0403-000536",
    # "CQL&cql_weight=5.0&temperature=5.0": "/export/pgs/fuyang/result_45/cql&cql_weight=5.0&temperature=5.0/dens/test/results/seed_1&timestamp_26-0403-000355",
    # "EDAC&num_critics=10&eta=0.3": "/export/pgs/fuyang/result_45/edac&num_critics=10&eta=0.3/dens/test/results/seed_1&timestamp_26-0403-185636",
    # "EDAC&num_critics=10&eta=2.0": "/export/pgs/fuyang/result_45/edac&num_critics=10&eta=2.0/dens/test/results/seed_1&timestamp_26-0404-150615",
    # "TD3BC&alpha=0.15": "/export/pgs/fuyang/result_45/td3bc&alpha=0.15/dens/test/results/seed_1&timestamp_26-0331-212915",
    # "TD3BC&alpha=0.25": "/export/pgs/fuyang/result_45/td3bc&alpha=0.25/dens/test/results/seed_1&timestamp_26-0331-213453",
    # "TD3BC&alpha=1.0": "/export/pgs/fuyang/result_45/td3bc&alpha=1.0/dens/test/results/seed_1&timestamp_26-0403-000259",
    # # "MCQ": "/zfsauton2/home/jiayuc2/bao/synthesize_test1/mcq&lmbda=0.809154&num_sampled_actions=20/rotation/results/seed_1&timestamp_26-0305-094807",
    # "COMBO&cql_weight=3&rollout_length=7": "/export/pgs/fuyang/result_45/combo&cql_weight=3&rollout_length=7/dens/test/results/seed_1&timestamp_26-0405-153038",
    # "COMBO&cql_weight=5&rollout_length=10": "/export/pgs/fuyang/result_45/combo&cql_weight=5&rollout_length=10/rotation/test/results/seed_1&timestamp_26-0402-163443",
    # "COMBO&cql_weight=10&rollout_length=10": "/export/pgs/fuyang/result_45/combo&cql_weight=10&rollout_length=10/dens/test/results/seed_1&timestamp_26-0404-153236",
    # #"IQL":"/home/scratch/jiayuc2/bao/Eval_optuna_out/rotation/iql&expectile=0.508981&temperature=1.26298/rotation/results/seed_1&timestamp_25-1130-110438",
    # "MOPO&penalty_coef=0.5&rollout_length=5": "/export/pgs/fuyang/result_45/mopo&penalty_coef=0.5&rollout_length=5/dens/test/results/seed_1&timestamp_26-0331-222540",
    # "MOPO&penalty_coef=1&rollout_length=5": "/export/pgs/fuyang/result_45/mopo&penalty_coef=1&rollout_length=5/dens/test/results/seed_1&timestamp_26-0331-221930",
    # "MOPO&penalty_coef=5.0&rollout_length=5": "/export/pgs/fuyang/result_45/mopo&penalty_coef=5.0&rollout_length=5/dens/test/results/seed_1&timestamp_26-0331-222141",                    
    # "MOBILE&penalty_coef=0.5&rollout_length=5": "/export/pgs/fuyang/result_45/mobile&penalty_coef=0.5&rollout_length=5/dens/test/results/seed_1&timestamp_26-0404-160842",
    # "MOBILE&penalty_coef=1.5&rollout_length=1": "/export/pgs/fuyang/result_45/mobile&penalty_coef=1.5&rollout_length=1/dens/test/results/seed_1&timestamp_26-0405-154235",
    # "MOBILE&penalty_coef=1.5&rollout_length=5": "/export/pgs/fuyang/result_45/mobile&penalty_coef=1.5&rollout_length=5/dens/test/results/seed_1&timestamp_26-0404-153635",
    # "MOBILE&penalty_coef=5.0&rollout_length=5": "/export/pgs/fuyang/result_329/mobile&penalty_coef=5.0&rollout_length=5/rotation/test/results/seed_1&timestamp_26-0331-194559"
    # #"BAMCTS":"/home/scratch/jiayuc2/bao/Eval_optuna/rotation/bambrl_mcts&rollout_length=2&penalty_coef=0.7067621971505375&use_ba=False&search_alpha=0.8/rotation/results/seed_1&timestamp_25-1212-144528",
    # # "ROMBRL":"/home/scratch/jiayuc2/temp_1106/rombrl&grad_mode=1&sl_weight=1000.0&actor_training_epoch=10&onpolicy_rollout_batch_size=2500&onpolicy_rollout_length=10&small_traj_batch=False/rotation/results/seed_1&timestamp_25-1031-122233",
    # # "RAMBO":"/home/scratch/jiayuc2/bao/Eval_optuna/rotation/rambo&rollout_length=2&adv_weight=6.471144404147533e-05/rotation/results/seed_1&timestamp_25-1212-145936",
   
   
   


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
save_path = "/export/pgs/fuyang/result_329/tracking_error_summary_rot.csv"
# save_path = "/home/scratch/jiayuc2/bao/Eval_csv/rotation/tracking_error_summary_rambo.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path)

print(f"\n Saved summary table to:\n{save_path}")
print(df.head(3))
