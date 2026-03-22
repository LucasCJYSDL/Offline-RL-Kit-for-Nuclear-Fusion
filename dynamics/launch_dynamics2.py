"""
Launch a bunch of jobs with specific GPU allocation.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import os
import subprocess
import time
from typing import Any, Dict

import numpy as np

# START_GPU = 0
command = 'python dynamics/train_dynamics_rohit.py'
time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = "/home/scratch/jiayuc2/bao/dynamic_models/logs"
os.makedirs(log_dir, exist_ok=True)

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
available_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))

log_path = f'{log_dir}/{time_str}.txt'
max_jobs_per_gpu = 1 #with batch size 1024, 1 job takes 10584 memory 
gpu_counts = {gpu_id: 0 for gpu_id in available_gpus}  # Track active jobs per GPU
max_running = len(available_gpus) * max_jobs_per_gpu
running = []
num_models_in_ensembles = 25

# Args -- config file for training
arg_dict = OrderedDict({
    'cn': [
        # "rpnn_minimal_cakenn_nll_mse_v4_exp002_q",
        # "rpnn_minimal_cakenn_nll_mse_v4_exp002_noq_fix",
        # "rpnn_minimal_cakenn_nll_mse_v4_exp0002_noq_fix25",
        # "rpnn_noshape_gas_benchmark_step2_t",
        # "rpnn_noshape_gas_bms_step_one_mse_1f_red",
        # "rpnn_noshape_gas_bms_step_one_logvar",
        #"rpnn_noshape_gas_bms_step_one_mse_1_red_v2",
        #"rpnn_noshape_gas_bms_step_one_mse_1_red_v4",
        #"rpnn_noshape_gas_bms_step_one_mse_1_red_v3",
        "rpnn_noshape_gas_benchmark2",
    ],
    'seed': [t for t in range(num_models_in_ensembles)],
})


@dataclass
class Job:
    proc: subprocess.Popen
    gpu: int
    args: Dict[str, Any]


def prune_completed_job():
    """Remove completed jobs and free up GPU slots."""
    for jidx, job in enumerate(running):
        if job.proc.poll() is not None:  # Job has finished
            gpu_counts[job.gpu] -= 1  # Free up the GPU slot
            with open(log_path, 'a') as f:
                f.write(f'{datetime.now()}\t Finished \t {job.args}\n')
            running.pop(jidx)
            return True
    return False


def add_job(args):
    """Assign a job to an available GPU."""
    if len(running) >= max_running:
        while not prune_completed_job():
            time.sleep(30)

    # Find an available GPU with an open job slot
    available_gpu = None
    for gpu in available_gpus:
        if gpu_counts[gpu] < max_jobs_per_gpu:
            available_gpu = gpu
            break

    if available_gpu is None:
        print("No available GPUs for a new job.")
        return

    gpu_counts[available_gpu] += 1
    cmd = command
    cmd += f' -cn {args["cn"]}'

    for k, v in args.items():
        if k != 'cn':
            cmd += f' {k}={v}'

    cmd += f' cuda_device={available_gpu}'  # Pass correct GPU ID

    with open(log_path, 'a') as f:
        f.write(f'{datetime.now()}\t Starting \t {args}\n')

    proc = subprocess.Popen(cmd, shell=True)
    running.append(Job(proc, available_gpu, args))


# Start Logging
with open(log_path, 'w') as f:
    f.write('Timestamp \t Status \t Args\n')

# Generate job combinations and execute them
arg_keys = list(arg_dict.keys())
num_each_args = np.array([len(arg_dict[k]) for k in arg_keys])
arg_idxs = np.zeros(len(arg_dict), dtype=int)

while True:
    add_job({k: arg_dict[k][arg_idxs[kidx]] for kidx, k in enumerate(arg_keys)})
    arg_idxs[0] += 1

    for ii in range(len(arg_idxs) - 1):
        if arg_idxs[ii] >= num_each_args[ii]:
            arg_idxs[ii] = 0
            arg_idxs[ii + 1] += 1

    if np.any(arg_idxs >= num_each_args):
        break

# Wait for remaining jobs to complete
while len(running) > 0:
    prune_completed_job()
    time.sleep(30)

with open(log_path, 'a') as f:
    f.write('Done!\n')