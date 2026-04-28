import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(env_name: str, default):
    value = os.getenv(env_name)
    path = Path(value) if value else Path(default)
    return path.expanduser().resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


data_root = _resolve_path("OFFLINERLKIT_DATA_ROOT", REPO_ROOT / "data")
raw_data_dir = _resolve_path("OFFLINERLKIT_RAW_DATA_DIR", data_root / "raw")
processed_data_dir = _resolve_path("OFFLINERLKIT_PROCESSED_DATA_DIR", data_root / "processed")
model_root = _resolve_path("OFFLINERLKIT_MODEL_ROOT", REPO_ROOT / "models")
training_model_dir = _resolve_path("OFFLINERLKIT_TRAINING_MODEL_DIR", model_root / "training")
evaluation_model_dir = _resolve_path(
    "OFFLINERLKIT_EVALUATION_MODEL_DIR",
    training_model_dir,
)
log_root = _resolve_path("OFFLINERLKIT_LOG_DIR", REPO_ROOT / "logs")
tune_root = _resolve_path("OFFLINERLKIT_TUNE_DIR", log_root / "tune")
evaluation_root = _resolve_path("OFFLINERLKIT_EVAL_DIR", REPO_ROOT / "results")
dynamics_root = _resolve_path("OFFLINERLKIT_DYNAMICS_DIR", model_root / "dynamics")


for _path in (
    data_root,
    raw_data_dir,
    processed_data_dir,
    model_root,
    training_model_dir,
    evaluation_model_dir,
    log_root,
    tune_root,
    evaluation_root,
    dynamics_root,
):
    _ensure_dir(_path)
