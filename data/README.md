---
pretty_name: 'n'
viewer: false
task_categories:
- reinforcement-learning
tags:
- offline-rl
- reinforcement-learning
- control
- nuclear-fusion
- hdf5
license: other
---

# Fusion RL Benchmark

## Dataset Summary

This dataset contains offline reinforcement learning and imitation learning data for profile-control tasks in nuclear fusion experiments. 

The release is split into two folders:

- `raw/`: source-style benchmark files and metadata used for dynamics model training
- `processed/`: HDF5 files derived from `raw/`, prepared for offline RL, imitation learning, and tracking evaluation

The processed files are intended to be loaded directly by the training and evaluation code in `rl_preparation`, `envs`, and `scripts` from the project repository.

## Supported Tasks

The repository defines the following canonical control tasks:

- `temp`
- `rotation`
- `dens`
- `pres`
- `q`
- `betan`


## Data Organization

### Directory layout

```text
data/
├─ raw/
│  ├─ il_data.h5
│  ├─ info.pkl
│  ├─ info.zip
│  ├─ rl_data.h5
│  ├─ rl_data_t150_t200.h5
│  ├─ tracking_test.h5
│  ├─ tracking_val.h5
│  └─ used_shots_info.txt
└─ processed/
   ├─ full.hdf5
   ├─ il_data.h5
   ├─ info.pkl
   ├─ rl_data.h5
   ├─ tracking_test.h5
   ├─ tracking_val.h5
   └─ used_shots_info.txt
```

### Main processed files

### Role of `raw/` vs `processed/`

`raw/` is the benchmark data source used to train or re-train the project’s dynamics models. In the associated repository, the raw files are consumed by the preprocessing pipeline and by the dynamics-related workflow before task-specific offline RL datasets are produced.

`processed/` contains the task-ready outputs built from `raw/`. These processed files are the ones typically used by:

- offline RL training
- imitation learning
- held-out tracking validation and test evaluation
- dynamics-aware downstream pipelines that expect the repository’s processed format

#### `processed/full.hdf5`

This is the broadest processed transition store. It contains:

- `states`: `(945828, 26)`
- `next_states`: `(945828, 26)`
- `actuators`: `(945828, 13)`
- `next_actuators`: `(945828, 13)`
- `shotnum`: `(945828,)`
- `time`: `(945828,)`
- actuator and state bounds

This file is useful if you want the main transition stream with shot IDs and timestamps.

#### `processed/rl_data.h5`

This file is the main offline RL training dataset. It contains:

- `observations`: `(849977, 26)`
- `next_observations`: `(849977, 26)`
- `actions`: `(849977, 13)`
- `pre_actions`: `(849977, 13)`
- `terminals`
- `time_step`
- `traj_start_indices`
- `hidden_states`: `(849977, 25, 1, 256)`
- action/state bounds


#### `processed/il_data.h5`

This file is the imitation learning counterpart of `rl_data.h5`. It contains the same core transition fields, but does not include the RNN dynamics `hidden_states`.

#### `processed/tracking_val.h5` and `processed/tracking_test.h5`

These files are grouped by shot ID. Each top-level key is a shot number, and each shot contains:

- `tracking_states`: `(T, 26)`
- `tracking_next_states`: `(T, 26)`
- `tracking_pre_actions`: `(T, 13)`
- `tracking_actions`: `(T, 13)`

Statistics:

- `tracking_val.h5`: 300 shots, sequence length range 130 to 180, average 160.03
- `tracking_test.h5`: 300 shots, sequence length range 130 to 180, average 159.47

These files are intended for rollout-style evaluation and profile tracking experiments.

## Feature Description

### State space

The observation/state dimension is 26. According to `processed/info.pkl`, the state variables are:

- `betan_EFIT01`
- `dssdenest`
- `li_EFIT01`
- `n1rms`
- `vloop`
- `wmhd_EFIT01`
- `temp_component1` to `temp_component4`
- `itemp_component1` to `itemp_component4`
- `dens_component1` to `dens_component4`
- `rotation_component1` to `rotation_component4`
- `pres_EFIT01_component1` to `pres_EFIT01_component2`
- `q_EFIT01_component1` to `q_EFIT01_component2`

The corresponding `next_state_space` stored in metadata is expressed as state velocities/deltas for the same 26 channels.

### Action space

The action dimension is 13. According to `processed/info.pkl`, the actuator channels are:

- `pinj`
- `tinj`
- `ipsiptargt`
- `bt_magnitude`
- `bt_is_positive`
- `gasA`
- `aminor_EFIT01`
- `tritop_EFIT01`
- `tribot_EFIT01`
- `kappa_EFIT01`
- `rmaxis_EFIT01`
- `zmaxis_EFIT01`
- `ech_pwr_total`

The `next_actuator_space` corresponds to actuator velocities/deltas for the same 13 channels.

## How To Load

These files are distributed as HDF5 rather than Arrow/Parquet, so the easiest way to inspect them is with `h5py`.

```python
import h5py

with h5py.File("processed/rl_data.h5", "r") as f:
    print(f["observations"].shape)
    print(f["actions"].shape)
    print(f["hidden_states"].shape)
```

For tracking data:

```python
import h5py

with h5py.File("processed/tracking_test.h5", "r") as f:
    shot_ids = list(f.keys())
    shot = shot_ids[0]
    print(shot)
    print(f[shot]["tracking_states"].shape)
    print(f[shot]["tracking_actions"].shape)
```


## Citation

If you use this dataset, please cite the accompanying project or paper associated with this release. If you have a canonical citation for the benchmark, add it here before publishing.