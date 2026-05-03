"""
Launch a bunch of dynamics training jobs with specific GPU allocation.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import os
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional


DEFAULT_COMMAND = ["python", "dynamics/train_dynamics.py"]
DEFAULT_CN = "rpnn_noshape_gas_benchmark2"
DEFAULT_LOG_DIR = "/home/scratch/jiayuc2/bao/dynamic_models/logs"
DEFAULT_GPUS = "0,1,2,3,4,5,6,7"
DEFAULT_MAX_JOBS_PER_GPU = 1  # With batch size 1024, 1 job takes ~10.6 GB.
DEFAULT_NUM_MODELS = 25
POLL_INTERVAL_SECONDS = 30


@dataclass
class Job:
    proc: subprocess.Popen
    gpu: int
    args: Dict[str, Any]


@dataclass
class LauncherState:
    log_path: str
    available_gpus: List[int]
    hydra_overrides: List[str]
    max_jobs_per_gpu: int
    gpu_counts: Dict[int, int]
    max_running: int
    running: List[Job]


def parse_args(argv: Optional[List[str]] = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Launch batched dynamics-model training jobs."
    )
    parser.add_argument(
        "--cn",
        default=DEFAULT_CN,
        help="Hydra config name under dynamics/cfgs, without the .yaml suffix.",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=DEFAULT_NUM_MODELS,
        help="Number of ensemble members to train. Seeds will run from 0 to num_models - 1.",
    )
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="Comma-separated GPU ids to use, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--max-jobs-per-gpu",
        type=int,
        default=DEFAULT_MAX_JOBS_PER_GPU,
        help="Maximum number of concurrent jobs to place on each GPU.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory used to store launcher log files.",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if args.num_models <= 0:
        parser.error("--num-models must be a positive integer.")
    if args.max_jobs_per_gpu <= 0:
        parser.error("--max-jobs-per-gpu must be a positive integer.")
    validate_hydra_overrides(parser, hydra_overrides)
    return args, hydra_overrides


def parse_gpu_list(gpu_arg: str) -> List[int]:
    gpu_strs = [part.strip() for part in gpu_arg.split(",") if part.strip()]
    if not gpu_strs:
        raise ValueError("At least one GPU id must be provided.")
    gpu_ids = [int(gpu_id) for gpu_id in gpu_strs]
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU ids must be non-negative integers.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU ids must not contain duplicates.")
    return gpu_ids


def validate_hydra_overrides(
    parser: argparse.ArgumentParser, hydra_overrides: List[str]
) -> None:
    reserved_prefixes = ("seed=", "cuda_device=", "cn=")
    reserved_flags = {"-cn", "--config-name", "-m", "--multirun"}
    for override in hydra_overrides:
        if override in reserved_flags:
            parser.error(f"{override} is reserved by the launcher.")
        if override.startswith(reserved_prefixes):
            parser.error(f"{override} conflicts with launcher-managed arguments.")


def build_arg_dict(args: argparse.Namespace) -> OrderedDict:
    return OrderedDict(
        {
            "cn": [args.cn],
            "seed": list(range(args.num_models)),
        }
    )


def generate_job_args(arg_dict: OrderedDict) -> List[Dict[str, Any]]:
    arg_keys = list(arg_dict.keys())
    arg_values = [arg_dict[key] for key in arg_keys]
    return [dict(zip(arg_keys, values)) for values in product(*arg_values)]


def prune_completed_job(state: LauncherState) -> bool:
    """Remove a completed job and free up its GPU slot."""
    for jidx, job in enumerate(state.running):
        if job.proc.poll() is not None:
            state.gpu_counts[job.gpu] -= 1
            with open(state.log_path, "a") as handle:
                handle.write(f"{datetime.now()}\t Finished \t {job.args}\n")
            state.running.pop(jidx)
            return True
    return False


def add_job(state: LauncherState, job_args: Dict[str, Any]) -> None:
    """Assign a job to an available GPU."""
    if len(state.running) >= state.max_running:
        while not prune_completed_job(state):
            time.sleep(POLL_INTERVAL_SECONDS)

    available_gpu = None
    for gpu in state.available_gpus:
        if state.gpu_counts[gpu] < state.max_jobs_per_gpu:
            available_gpu = gpu
            break

    if available_gpu is None:
        print("No available GPUs for a new job.")
        return

    state.gpu_counts[available_gpu] += 1
    cmd = list(DEFAULT_COMMAND)
    cmd.extend(["-cn", job_args["cn"]])
    for key, value in job_args.items():
        if key != "cn":
            cmd.append(f"{key}={value}")
    cmd.append(f"cuda_device={available_gpu}")
    cmd.extend(state.hydra_overrides)

    cmd_str = shlex.join(cmd)
    with open(state.log_path, "a") as handle:
        handle.write(f"{datetime.now()}\t Starting \t {job_args}\t {cmd_str}\n")

    proc = subprocess.Popen(cmd)
    state.running.append(Job(proc, available_gpu, job_args))


def main() -> None:
    args, hydra_overrides = parse_args()
    try:
        available_gpus = parse_gpu_list(args.gpus)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in available_gpus)
    os.makedirs(args.log_dir, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"{args.log_dir}/{time_str}.txt"
    state = LauncherState(
        log_path=log_path,
        available_gpus=available_gpus,
        hydra_overrides=hydra_overrides,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        gpu_counts={gpu_id: 0 for gpu_id in available_gpus},
        max_running=len(available_gpus) * args.max_jobs_per_gpu,
        running=[],
    )

    with open(log_path, "w") as handle:
        handle.write("Timestamp \t Status \t Args \t Command\n")

    for job_args in generate_job_args(build_arg_dict(args)):
        add_job(state, job_args)

    while state.running:
        prune_completed_job(state)
        time.sleep(POLL_INTERVAL_SECONDS)

    with open(log_path, "a") as handle:
        handle.write("Done!\n")


if __name__ == "__main__":
    main()
