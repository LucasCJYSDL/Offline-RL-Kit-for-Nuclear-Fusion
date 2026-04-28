# Offline RL Kit for Nuclear Fusion

## Quick Start

1. Clone this repository and install the environment from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate offline-rl-fusion
   ```

2. Install this project in editable mode from the repo root:
   ```bash
   pip install -e .
   ```

3. If your data or model directories are not in the default `data/` and `models/` locations, set these environment variables before running the code:
   - `OFFLINERLKIT_DATA_ROOT`
   - `OFFLINERLKIT_RAW_DATA_DIR`
   - `OFFLINERLKIT_PROCESSED_DATA_DIR`
   - `OFFLINERLKIT_MODEL_ROOT`
   - `OFFLINERLKIT_TRAINING_MODEL_DIR`
   - `OFFLINERLKIT_EVALUATION_MODEL_DIR`

- You need to download/clone the repo: [dynamics-toolbox](https://github.com/LucasCJYSDL/dynamics-toolbox). This toolbox is different from Ian's, as we have changed the rpnn class.

## Dynamics Modelling

- (Optional) As an alternative of real experiment data, we provide a script to synthesize data with an exisiting dynamics model:
    ```bash
    python -m dynamics.synthesize_rollouts
    ```
- You can train an ensemble of dynamics models by running the following command:
    ```bash
    python -m dynamics.train_dynamics
    ```
    - By default, the script will use the configuration specified by the "config_name" argument within "train_dynamics.py". You may optinally modify or add configuration files in "dynamics/cfgs".

## Policy Learning

- Please start from converting the raw fusion data to the format required by offline RL or Imitation Learning:
    ```bash
    OFFLINERLKIT_TASK=temp python -m rl_preparation.process_raw_data
    ```
  - Canonical task names are `temp`, `rotation`, `dens`, `pres`, `q`, and `betan`.
  - Legacy names like `pres_EFIT01`, `q_EFIT01`, and `betan_EFIT01` are still accepted, but new examples use the canonical names above.

- You can run different offline RL algorithms simply by:
    ```bash
    python -m rl_scripts.run_ppo --env profile_control_new --task temp
    ```
    - Other available entry points include `run_cql`, `run_iql`, `run_edac`, `run_mcq`, `run_td3bc`, `run_combo`, `run_mobile`, `run_mopo`, `run_bambrl`, `run_rambo`, and `run_rombrl`.
    - Pick the task that matches the data you prepared, for example `temp`, `rotation`, `dens`, `pres`, `q`, or `betan`.
    - Each algorithm now has its own JSON file under `offlinerlkit/configs/` such as `ppo.json`, `cql.json`, `td3bc.json`, and `mppi.json`.
    - To customize algorithm parameters, copy the matching JSON file, edit its `default_params` section, and pass the copy back with `--config` if you want to keep the original untouched. Command-line flags still override config defaults, so you can keep a shared JSON and tweak a few values ad hoc.

- You can run a goal-conditioned imitation learning algorithm by:
    ```bash
    python -m rl_scripts.run_gcil --env profile_control --task rotation
    ```

- As an alternative, we provide scripts to run planning algorithms on the learned dynamics and distill policy functions from the planning results, which can be run by:
    ```bash
    python -m rl_scripts.run_mppi --env profile_control --task rotation
    ```
    - The current planning entry point is `run_mppi`.

- The package now exposes a small convenience API:
  ```python
  import torch
  import offlinerlkit
  import rl_preparation as rp

  print(offlinerlkit.available_envs())
  print(offlinerlkit.available_tasks())
  print(offlinerlkit.available_algo_names())
  rp.configure_task("temp")
  dataset = offlinerlkit.load_dataset("profile_control_new", "temp", device=torch.device("cpu"))
  env = offlinerlkit.make_env("profile_control_new", "temp", device=torch.device("cpu"))
  algo_cfg, cfg_path = offlinerlkit.load_algo_config("ppo")
  print(cfg_path)
  ```

- Please find more instructions on how to run/extend the codebase in this [tutorial](https://drive.google.com/file/d/1PVcsTshC1FaqZ9pweT0eW_SvUdu-ZaWu/view?usp=sharing).

## Evaluations

- Plots of episodic returns during the policy learning process, along with trained policy model checkpoints, will be generated and saved in the 'logs' folder.

- Dynamics training results will be generated and saved in the 'outputs' folder.

- We also provide a unified script in the `visualization` folder to visualize actuator choices made by a trained policy, save profile-tracking outputs, and summarize tracking error for the same run folder in one pass. The script auto-detects the algorithm name from `actor_path`, applies the matching built-in actor settings, and writes outputs under `results/<task>/<algo>/<val|test>/<run>/<seed>/` depending on whether you pass `--test`, so you do not need a separate visualization config file. Run:
  ```bash
  python -m visualization.profile_tracking_save --actor_path <your_run_dir>
  ```
