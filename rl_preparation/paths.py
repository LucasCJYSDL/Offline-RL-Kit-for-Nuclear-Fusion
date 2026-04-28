from offlinerlkit.paths import (
    REPO_ROOT,
    evaluation_model_dir,
    processed_data_dir,
    raw_data_dir,
    training_model_dir,
)

full_data_path = processed_data_dir / "full.hdf5"
rl_data_path = processed_data_dir / "rl_data.h5"
il_data_path = processed_data_dir / "il_data.h5"
tracking_val_data_path = processed_data_dir / "tracking_val.h5"
tracking_test_data_path = processed_data_dir / "tracking_test.h5"

__all__ = [
    "REPO_ROOT",
    "raw_data_dir",
    "training_model_dir",
    "evaluation_model_dir",
    "processed_data_dir",
    "full_data_path",
    "rl_data_path",
    "il_data_path",
    "tracking_val_data_path",
    "tracking_test_data_path",
]
