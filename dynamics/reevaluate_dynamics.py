"""
Re-evaluate dynamics models.
"""

import argparse
import os
import pickle as pkl

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
    load_ensemble_from_parent_dir,
)
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import torch

import uncertainty_toolbox as uct
from matplotlib import pyplot as plt

from dynamics.orig_kfold_data_module import KFoldFusionDataModule
from dynamics.orig_kfold_sequence_data_module import KFoldSequenceFusionDataModule

import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf

import hydra

#set cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

###########################################################################
# %% Parse the arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    # default = "/home/scratch/rsonker/ech/dynamics_models/rpnn_noshape_gas_step_two_logvar"
    # default = "/home/scratch/rsonker/dynamics_models/rpnn_noshape_gas_new_shots_step_two_nll_bmsinfo",
    # default = '/zfsauton/project/fusion/models/rpnn_noshape_gas_benchmark_step2',
    # default= "/home/scratch/rsonker/dynamics_models/rpnn_noshape_gas_bms_step_one_mse_1_red_v2",
    # default= "/zfsauton/project/fusion/models/rpnn_noshape_gas_new_shots_step_two_nll_bmsinfo",
    # default= "/zfsauton/project/fusion/models/rpnn_minimal_zipfit_cont001_noq_final",
    # default = "/zfsauton/project/fusion/ndeka/models_out/minimal_cakenn_v4_expand_cont002_noq"
    default = "/home/scratch/jiayuc2/bao/dynamic_models/rpnn_noshape_gas_benchmark_synthesize_step2"
)
# default = '/home/scratch/rsonker/FusionControl/dynamics_out/ijcnn/ijcnn_model')
# getting the data from the model cfg
# parser.add_argument(  
#     "--data_dir",
#     type=str,
#     # default='./data/highbetap/v2/25ms/difference_subset')
#     # default='./data/v7/profile/2022-05-10')
#     # default ='/home/scratch/rsonker/FusionControl/data/organised/aug2023')
#     # default = '/home/scratch/rsonker/FusionControl/data/2023-02-01/wshapecontrol')
#     # default="/home/scratch/rsonker/FusionControl/data/2023-02-01/wshapecontrol",
#     default = "/home/scratch/rsonker/ech/data/organized/noshape_qrfe_tm_1_v2",
# )
parser.add_argument("--samples_per_pt", type=int, default=1) # 30 only when sampling
parser.add_argument("--convert_to_difference", type=int, default=0)
parser.add_argument("--is_ensemble", type=int, default=1)
parser.add_argument("--is_rnn", type=int, default=1)
parser.add_argument("--is_gaussian", default=0, type=int)   ## NO GAUSSIAN
parser.add_argument("--do_uct", default=0, type=int)
parser.add_argument("--warm_up", default=0, type=int)
parser.add_argument("--cuda_device", default=3, type=int)
parser.add_argument("--use_model_cfg_data_module", type=int, default=1)
parser.add_argument("--provide_model_for_te_shots", type=str, default="/home/scratch/rsonker/dynamics_models/rpnn_noshape_gas_bms_step_one_mse_1_red")
parser.add_argument(
    "--te_relative_path",
    type=str,
    # default='100/te_shots.npy')
    default="0/te_shots.npy",
)

# Set this using the model name
# parser.add_argument(
#     "--output_path", default="model_out/rpnn_noshape_qrfe_tm_1-wotm", type=str
# )
args = parser.parse_args()
device = (
    torch.device(f"cuda:{args.cuda_device}")
    if args.cuda_device is not None and args.cuda_device >= 0 and torch.cuda.is_available()
    else torch.device("cpu")
)

###########################################################################
# %% Load in the model, dataset, and get the correct shots.
###########################################################################
if args.is_ensemble:
    model = load_ensemble_from_parent_dir(args.model_dir)
    te_shots = np.load(os.path.join(args.model_dir, args.te_relative_path))
    if device.type == "cuda":
        for memb in model.members:
            memb.to(device)
            memb.eval()
else:
    model = load_model_from_log_dir(args.model_dir)
    te_shots = np.load(os.path.join(args.model_dir, args.te_relative_path))
    if device.type == "cuda":
        model = model.to(device)
        model.eval()
    else:
        model.eval()


# te_shots = np.load("/zfsauton/project/fusion/models/rpnn_noshape_gas_flat_top_step_two_logvar/0/te_shots.npy")

modelname = os.path.basename(args.model_dir)
output_path = os.path.join("model_out", modelname)

#load organization config.yaml from data directory - no need for this
# cfg_data = OmegaConf.load(os.path.join(data_dir, "config.yaml"))
# abs_states = cfg_data.get("override_to_abs_for_scalar_states", [])



#load config.yaml from model directory
cfg = OmegaConf.load(os.path.join(args.model_dir, "0/config.yaml"))
data_dir = cfg['data_path']

if args.use_model_cfg_data_module:
    # data_folder = cfg['data_module']['data_path'].split("/")[-1]
    # if cfg['data_module']['data_path'].split("/")[1]!='zfsauton':
    #     data_dir = os.path.join("/zfsauton/project/fusion/data/organized", data_folder)
    #     cfg['data_module']['data_path'] = data_dir
    cfg["data_module"]["num_workers"] = 0
    cfg["data_module"]["pin_memory"] = False
    dataset = hydra.utils.instantiate(cfg['data_module'], _recursive_=False)
    remove_states=cfg.get("data_module").get("remove_states", [])
    remove_actuators=cfg.get("data_module").get("remove_actuators", [])

    print("**** SETTING THE VAL LOADER AS TEST LOADER ****")
    dataset._num_workers = 0
    dataset._pin_memory = False
    dataset.val_dataloader = dataset.test_dataloader
else:

    # Set other model filepaths
    model_path = args.provide_model_for_te_shots
    try:
        te_shots = np.load(os.path.join(model_path, args.te_relative_path))
    except:
        raise ValueError("Please provide correct model path for test shots or set --use_model_cfg_data_module to 1")

    

    # data_dir = "/home/scratch/rsonker/data/organized/noshape_gas_flattop_bmsinfo"
    data_dir = data_dir

    all_data = load_from_hdf5(os.path.join(data_dir, "full.hdf5"))
    te_idxs = [sn in te_shots for sn in all_data["shotnum"]]
    te_data = {}
    num_rows = len(all_data["shotnum"])
    for key, value in all_data.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == num_rows:
            te_data[key] = value[te_idxs]
        else:
            te_data[key] = value
    if args.is_rnn:
        dataset_class = KFoldSequenceFusionDataModule
    else:
        dataset_class = KFoldFusionDataModule
    print("Data Class Used : ", dataset_class)
    remove_states=cfg.get("data_module").get("remove_states", [])
    remove_actuators=cfg.get("data_module").get("remove_actuators", [])

    dataset = dataset_class(
        data_path=data_dir,
        batch_size=512,
        shot_train_length=cfg['data_module']['shot_train_length'],
        n_folds=1,
        te_fold=1,
        pin_memory=False,
        prop_validation=1.0,
        loaded_data=te_data,
        min_shot_amt=2,
        remove_states=cfg.get("data_module").get("remove_states", []),
        remove_actuators=cfg.get("data_module").get("remove_actuators", []),
    )
    dataset._num_workers = 0
    dataset._pin_memory = False

print("\n\n *************************** \n\n")
# te_shots = te_shots[:5]
print("Gaussian sampling is ", args.is_gaussian)
print("Number of test shots = ", te_shots.shape)
print("#Samples per point = ",args.samples_per_pt)

with open(os.path.join(data_dir, "info.pkl"), "rb") as f:
    info = pkl.load(f)
state_size = len(info["state_space"])

###########################################################################
# %% Form the predictions on the test data.
###########################################################################
if args.is_ensemble:
    all_models = model.members
else:
    all_models = [model]
predictions = []
pbar = tqdm(total=len(all_models) * args.samples_per_pt)
labels = []
fill_labels = True
real_samples = 0
for memb in all_models:
    memb_preds = []
    for itr in range(args.samples_per_pt):
        itr_preds = []
        with torch.no_grad():
            for batch in dataset.val_dataloader(): # NOTE VAL loader may be set as test loader right now depending on dataset selection
                batch = [b.to(device) for b in batch]
                normed_batch = memb.normalizer.normalize_batch(batch)
                if args.is_gaussian:
                    net_out = memb.get_net_out(normed_batch) #todo
                    mean = net_out["mean"]
                    std = (0.5 * net_out["logvar"]).exp()
                    samples = torch.randn_like(mean) * std + mean
                else:
                    samples = memb.get_net_out(normed_batch)["mean"]
                samples = memb.normalizer.unnormalize(samples, 1)
                # there's no need for this - net predicts full state, labels are set in abs in org script
                # if cfg_data.get('labels_are_velocities',False):
                #     for s in abs_states:
                #         s_idx = info["state_space"].index(s)
                #         samples[..., s_idx] = samples[..., s_idx] + batch[0][..., s_idx]
                if args.convert_to_difference:
                    samples -= batch[0][..., :state_size]
                if args.is_rnn:
                    samples = samples[:, args.warm_up :]
                    valid_mask = batch[-1][:, args.warm_up :].squeeze(-1).bool()
                    samples = samples[valid_mask]
                itr_preds.append(samples)
                if fill_labels:
                    if args.convert_to_difference:
                        label_to_append = batch[1] - batch[0][..., :state_size]
                    else:
                        label_to_append = batch[1]
                    label_to_append = label_to_append[:, args.warm_up :]
                    if args.is_rnn:
                        real_samples += int(torch.sum(batch[-1]))
                        label_to_append = label_to_append[valid_mask]
                    else:
                        real_samples += batch[0].shape[0]
                    labels.append(label_to_append.cpu())
            fill_labels = False
        memb_preds.append(torch.cat(itr_preds, dim=0).cpu())
        pbar.update(1)
    predictions.append(torch.stack(memb_preds).cpu())
predictions = torch.stack(predictions)
labels = torch.cat(labels, dim=0)
pbar.close()


###########################################################################
# %% Save the predictions
###########################################################################
# torch.save(predictions, 'v7preds.pt')
# torch.save(predictions, 'v7labels.pt')
# predictions = torch.load('v7preds.pt')


###########################################################################
# %% Get the MSE and EV
###########################################################################
stats = []

manual_next_state_space = cfg['model']['dim_name_map']

stat_labels = (
    ["Model", "MSE"]
    + [f"{x} MSE" for x in manual_next_state_space] 
    + [f"{x} EV" for x in manual_next_state_space]
)

# stat_labels = (
#     ["Model", "MSE"]
#     + [f"{info['next_state_space'][i]} MSE" for i in range(len(info["next_state_space"]))]
#     + [f"{info['next_state_space'][i]} EV" for i in range(len(info["next_state_space"]))]
# )

pop_state_labels = []
for s in remove_states:
    pop_state_labels.append(f"{s} MSE")
    pop_state_labels.append(f"{s} EV")
    pop_state_labels.append(f"{s}_velocity EV")
    pop_state_labels.append(f"{s}_velocity MSE")

stat_labels = [s for s in stat_labels if s not in pop_state_labels]
assert len(stat_labels)-2 == predictions.shape[-1]*2

# Get the statistics for each model.
for memb_num, model_preds in enumerate(predictions):
    model_stats = [f"Model {memb_num}"]
    model_mean_preds = torch.mean(model_preds, dim=0)
    # Get athe MSE
    model_stats.append(
        float(torch.sum(torch.sum((model_mean_preds - labels) ** 2, dim=-1)))
        / real_samples
    )

    mse_dim = torch.sum(((model_mean_preds - labels) ** 2), dim=0) / real_samples

    for pred_dim in range(model_mean_preds.shape[-1]):
        model_stats.append(float(mse_dim[pred_dim]))

    # Compute the explained variance and MSE
    for pred_dim in range(model_mean_preds.shape[-1]):
        true_var = torch.var(labels[..., pred_dim])
        resid_var = torch.var(labels[..., pred_dim] - model_mean_preds[..., pred_dim])
        model_stats.append(float(1 - resid_var / true_var))

    # model_stats.append(np.mean(model_stats[2:]))
    stats.append(model_stats)
# Get the statistics for the ensemble as a whole.
ens_preds = torch.mean(torch.mean(predictions, dim=0), dim=0)
ens_stats = ["Ensemble"]
# Get athe MSE
ens_stats.append(float(torch.mean(torch.sum((ens_preds - labels) ** 2, dim=-1))))

mse_dim = torch.mean(((ens_preds - labels) ** 2), dim=0)
# Get MSE Dimension wise
for pred_dim in range(ens_preds.shape[-1]):
    ens_stats.append(float(mse_dim[pred_dim]))

# Compute the explained variance
for pred_dim in range(ens_preds.shape[-1]):
    true_var = torch.var(labels[..., pred_dim])
    resid_var = torch.var(labels[..., pred_dim] - ens_preds[..., pred_dim])
    ens_stats.append(float(1 - resid_var / true_var))
ens_stats.append(np.mean(ens_stats[2:]))
stats.append(ens_stats)

# Transpose the table.
rows = [[el[j] for el in [stat_labels] + stats] for j in range(len(stat_labels))]
table = tabulate(rows)
print(table)
# with open('ijcnn_wshape_val.txt', 'w') as f:
#     f.write(table)

# create dataframe
df = pd.DataFrame(rows[1:], columns=rows[0])
# df.to_csv('te_shots.csv', index=False)

timestamp = datetime.now().strftime("%Y%m%d-%H%M")
DIR = os.path.join(output_path, timestamp)
os.makedirs(DIR, exist_ok=True)
df.to_csv(f"{DIR}/te_shots.csv", index=False)
print(f"File saved - {DIR}/te_shots.csv")

# saving the predictions and actuals
ensemble_results = {}
for i,field in zip(range(ens_preds.shape[-1]), manual_next_state_space):
    ensemble_results[field] = {}
    ensemble_results[field]['actual'] = labels[...,i].detach().cpu().numpy()
    ensemble_results[field]['predicted'] = ens_preds[...,i].detach().cpu().numpy()

with open(f"{DIR}/ensemble_results.pkl", "wb") as f:
    pkl.dump(ensemble_results, f)
print(f"File saved - {DIR}/ensemble_results.pkl")


###########################################################################
# %% Get the uncertainty metrics.
###########################################################################
predictions = predictions.view(-1, predictions.shape[-2], predictions.shape[-1])
labels = labels.cpu().numpy()
uct_metrics = {}
if len(predictions) > 1 and args.do_uct:
    mean_ests = torch.mean(predictions, dim=0).numpy()
    std_ests = torch.std(predictions, dim=0).numpy()
    for nidx, dimname in enumerate(info["next_state_space"]):
        print("=" * 30, dimname, "=" * 30)
        uct_metrics[dimname] = uct.metrics.get_all_metrics(
            mean_ests[..., nidx], std_ests[..., nidx], labels[..., nidx]
        )
        uct.viz.plot_calibration(
            mean_ests[..., nidx], std_ests[..., nidx], labels[..., nidx]
        )
        # plt.show()
        #save plot
        plt.savefig(f"{DIR}/{dimname}_calibration.png")

    ###########################################################################
    # %% Get average miscal
    ###########################################################################
    # print("\n\n\nUCT metrics report - ")
    # print(uct_metrics.values()[0])
    miscals = [v["avg_calibration"]["miscal_area"] for v in uct_metrics.values()]
    sharps = [v["sharpness"]['sharp'] for v in uct_metrics.values()]
    rmse = [v['accuracy']['rmse'] for v in uct_metrics.values()]
    r2 = [v['accuracy']['r2'] for v in uct_metrics.values()]
    print(miscals, np.mean(miscals))

    # df1 = pd.DataFrame.from_dict(uct_metrics)
    # df1.to_csv(f"{DIR}/uct_metrics.csv", index=False)

    with open(f"{DIR}/uct_metrics.pkl", "wb") as f:
        pkl.dump(uct_metrics, f)

    print(f"File saved - {DIR}/uct_metrics.csv")

    df2 = pd.DataFrame()
    df2['dim'] = info["next_state_space"]
    df2['miscal_area'] = miscals
    df2['sharpness'] = sharps
    df2['rmse'] = rmse
    df2['r2'] = r2
    df2.to_csv(f"{DIR}/uct_summary.csv", index=False)
