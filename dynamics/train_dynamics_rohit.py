"""
Main file to use for training dynamics models.

Author: Ian Char
"""

import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.seed import seed_everything

import dynamics_toolbox
from dynamics_toolbox.utils.lightning.constructors import (
    construct_all_pl_components_for_training,
)

from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
    load_ensemble_from_parent_dir,
)

import fusion_control
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(
    #config_path=f'{os.environ["FUSION_CONTROL_PROJECT_PATH"]}/cfgs/dynamics',
    config_path=f'{os.path.dirname(__file__)}/cfgs',
    # config_name='pnn')
    # config_name='profile_rnn_gpu')
    # config_name='profile_prnn_gpu')
    # config_name='profile_mlp_gpu')
    # config_name='profile_simplex_gpu')
    # config_name="rpnn_noshape_gas_transfer_step_two_nll",
    # config_name="tpnn_noshape_gas_bms_nll",
    #config_name="rpnn_noshape_gas_benchmark1",
    config_name="rpnn_noshape_gas_benchmark1",
)
def train(cfg: DictConfig) -> None:
    """Train the model."""
    if "model" not in cfg:
        raise ValueError(
            "model must be specified. Choose one of the provided "
            "model configurations and set +model=<model_config>."
        )
    if "seed" in cfg:
        seed_everything(cfg["seed"])
    # Alter config file and add defaults.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])
    with open_dict(cfg):
        if "save_dir" not in cfg:
            cfg["save_dir"] = os.path.join(os.getcwd(), "model")
        elif cfg["save_dir"][0] != "/":
            cfg["save_dir"] = os.path.join(get_original_cwd(), cfg["save_dir"])
        if "gpus" in cfg:
            cfg["gpus"] = str(cfg["gpus"])
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg)
    #load model if load_checkpoint is given

    # print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
    # print("Load checkpoint = ",cfg.get('load_checkpoint',None))
    # if cfg.get('load_checkpoint',None) is not None:
    #     # model = model.load_from_checkpoint(cfg['load_checkpoint'])
    #     print("\n\nLoading model from checkpoint - SEED  = ", cfg['seed'],"\n\n")
    #     load_checkpoint = cfg['load_checkpoint']+f"/{cfg['seed']}"
    #     model = load_model_from_log_dir(load_checkpoint)

    print(OmegaConf.to_yaml(cfg))
    if cfg["logger"] == "mlflow":
        save_path = os.path.join(cfg["save_dir"], logger.experiment_id, logger.run_id)
    else:
        if "run_name" in cfg:
            name = os.path.join(cfg["experiment_name"], cfg["run_name"])
        else:
            name = cfg["experiment_name"]
        save_path = os.path.join(logger.save_dir, name, f"version_{logger.version}")
        save_path = os.getcwd()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    OmegaConf.save(cfg, os.path.join(save_path, "config.yaml"))
    trainer.fit(model, data)
    if data.test_dataloader() is not None:
        test_dict = trainer.test(model, datamodule=data, ckpt_path="best")[0]
        with open("test_results.txt", "w") as f:
            f.write("\n".join([f"{k}: {v}" for k, v in test_dict.items()]))
        tune_metric = cfg.get("tune_metric", "test/loss")
        return_val = test_dict[tune_metric]
        if cfg.get("tune_objective", "minimize") == "maximize":
            return_val *= -1
        return return_val
    else:
        return 0


if __name__ == "__main__":
    train()