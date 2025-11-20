import argparse
import random
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, Any
import numpy as np
import torch
import optuna
from optuna.trial import TrialState

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GenericHyperparameterTuner:
    """Generic hyperparameter tuner for offline RL algorithms"""
    
    def __init__(self, algo_name: str, config_file: str, base_args, gpu_ids: list = None):
        self.algo_name = algo_name.lower()
        self.base_args = base_args
        self.gpu_ids = gpu_ids or [base_args.cuda_id]  # Default to single GPU
        self.gpu_index = 0
        
        # Load algorithm configuration from JSON file
        self.config = self._load_config(config_file)
        
        # Dynamically import the training function
        self._import_train_function()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load algorithm configuration from JSON file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                all_configs = json.load(f)
            
            if self.algo_name not in all_configs:
                raise ValueError(f"Algorithm '{self.algo_name}' not found in {config_file}")
            
            config = all_configs[self.algo_name]
            
            # Validate required fields
            required_fields = ["tunable_params", "train_module", "record_params"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' for algorithm '{self.algo_name}'")
            
            print(f"Loaded configuration for {self.algo_name} from {config_file}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_file}: {e}")
    
    def _import_train_function(self):
        """Dynamically import the train function from the corresponding module"""
        try:
            module_name = self.config["train_module"]
            exec(f"from {module_name} import train")
            self.train_fn = locals()['train']
        except ImportError as e:
            print(f"Error importing {self.config['train_module']}: {e}")
            raise
    
    def _get_next_gpu(self) -> int:
        """Get next GPU ID in round-robin fashion"""
        gpu_id = self.gpu_ids[self.gpu_index % len(self.gpu_ids)]
        self.gpu_index += 1
        return gpu_id
    
    def create_args_for_trial(self, trial):
        """Create arguments for a specific trial based on algorithm config"""
        # Copy base args
        args = argparse.Namespace(**vars(self.base_args))
        
        # Assign GPU to this trial (round-robin across available GPUs)
        args.cuda_id = self._get_next_gpu()
        
        # Suggest tunable hyperparameters based on configuration
        for param_name, param_config in self.config["tunable_params"].items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                value = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                value = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "categorical":
                value = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            setattr(args, param_name, value)
        
        # Set unique seed for each trial
        args.seed = self.base_args.seed # + trial.number
        
        # Reduce training time for hyperparameter search
        # Allow these to be overridden by config
        if hasattr(args, 'epoch'):
            args.epoch = self.config.get("tuning_epoch", 100)
        if hasattr(args, 'step_per_epoch'):
            args.step_per_epoch = self.config.get("tuning_step_per_epoch", 500)
        if hasattr(args, 'eval_episodes'):
            args.eval_episodes = self.config.get("tuning_eval_episodes", 3)
        
        # Enable early stopping if available and configured
        if self.config.get("enable_early_stop", True):
            if hasattr(args, 'early_stop'):
                args.early_stop = True
                args.early_stop_wait_epochs = self.config.get("early_stop_wait_epochs", 10)
                args.reward_improvement_threshold = self.config.get("reward_improvement_threshold", 50)
        
        return args
    
    def extract_reward_from_logs(self, log_dir: str) -> float:
        """Extract the best reward from training logs"""
        try:
            import pandas as pd
            
            # Look for CSV files with training progress
            csv_files = []
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith('progress.csv') and 'policy_training' in file:
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print(f"Warning: No CSV files found in {log_dir}")
                return -1000.0
            
            # Read the latest CSV file
            df = pd.read_csv(csv_files[-1])
            
            # Try to find reward column using config
            reward_column = self.config.get("reward_column", None)
            
            if reward_column and reward_column in df.columns:
                best_reward = df[reward_column].max()
                return float(best_reward) if not np.isnan(best_reward) else -1000.0
            
            # Fallback: Look for eval reward columns
            reward_columns = [col for col in df.columns 
                            if 'eval' in col.lower() and 'reward' in col.lower()]
            
            if reward_columns:
                best_reward = df[reward_columns[0]].max()
                return float(best_reward) if not np.isnan(best_reward) else -1000.0
            
            # Fallback to common column names
            possible_columns = ['reward', 'return', 'episode_reward', 'eval_return', 'eval_reward']
            for col in possible_columns:
                if col in df.columns:
                    best_reward = df[col].max()
                    return float(best_reward) if not np.isnan(best_reward) else -1000.0
            
            print(f"Warning: No reward column found. Available: {list(df.columns)}")
            return -1000.0
            
        except Exception as e:
            print(f"Error extracting reward from logs: {str(e)}")
            return -1000.0
    
    def objective(self, trial):
        """Objective function for Optuna"""
        try:
            # Create arguments for this trial
            args = self.create_args_for_trial(trial)
            
            # Print trial info with GPU assignment
            param_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in trial.params.items()])
            print(f"Trial {trial.number} (GPU {args.cuda_id}): {param_str}")
            
            # Train the model
            self.train_fn(args)
            
            # Extract reward from logs
            log_base_dir = f"logs/{args.task}"
            if os.path.exists(log_base_dir):
                reward = self.extract_reward_from_logs(log_base_dir)
            else:
                reward = self.extract_reward_from_logs("./logs")
            
            # Clean up trial logs to save space
            import shutil
            if os.path.exists(log_base_dir):
                shutil.rmtree(log_base_dir)
            
            print(f"Trial {trial.number} completed with reward: {reward:.4f}")
            return reward
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return -1000.0


def get_tuning_args():
    """Get command line arguments for hyperparameter tuning"""
    parser = argparse.ArgumentParser(description="Generic Optuna-based hyperparameter tuner for offline RL")
    
    # Algorithm and configuration
    parser.add_argument("--algo", type=str, required=True,
                       help="Algorithm to tune (must exist in config file)")
    parser.add_argument("--config", type=str, default="algo_config.json",
                       help="Path to algorithm configuration JSON file")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="profile_control",
                       help="Environment name")
    parser.add_argument("--task", type=str, default="rotation",
                       help="Task name")
    parser.add_argument("--cuda-id", type=int, default=7,
                       help="Default CUDA device ID (used if --gpu-ids not specified)")
    parser.add_argument("--gpu-ids", type=int, nargs='+', default=None,
                       help="List of GPU IDs to use for parallel trials (e.g., --gpu-ids 4 5 6 7)")
    parser.add_argument("--seed", type=int, default=1,
                       help="Base random seed")
    
    # Optuna settings
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default=None,
                       help="Optuna study name (default: {algo}_optimization)")
    parser.add_argument("--storage", type=str, default="sqlite:////home/scratch/jiayuc2/fy/optuna_study.db",
                       help="Optuna storage URL for distributed optimization")
    parser.add_argument("--output-dir", type=str, default="/home/scratch/jiayuc2/fy/optuna_results",
                       help="Directory to save optimization results")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Number of parallel jobs (one per GPU recommended)")
    
    return parser.parse_args()


def create_base_training_args(algo_name: str, config: Dict, tuning_args) -> argparse.Namespace:
    """Create base training arguments from configuration"""
    args = argparse.Namespace()
    
    # Environment settings
    args.env = tuning_args.env
    args.task = tuning_args.task
    args.cuda_id = tuning_args.cuda_id
    args.seed = tuning_args.seed
    args.algo_name = algo_name
    
    # Load default parameters from config
    if "default_params" in config:
        for param_name, param_value in config["default_params"].items():
            setattr(args, param_name, param_value)
    
    return args


def save_optimization_results(study, output_dir: str, algo_name: str, tuning_args):
    """Save optimization results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best parameters
    best_params_file = os.path.join(output_dir, f"{algo_name}_best_params_{timestamp}.json")
    best_params = {
        "algorithm": algo_name,
        "best_value": float(study.best_trial.value) if study.best_trial else None,
        "best_params": study.best_trial.params if study.best_trial else {},
        "environment": tuning_args.env,
        "task": tuning_args.task,
        "n_trials": tuning_args.n_trials,
        "gpu_ids": tuning_args.gpu_ids or [tuning_args.cuda_id],
        "n_jobs": tuning_args.n_jobs,
        "timestamp": timestamp
    }
    
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{algo_name}_detailed_results_{timestamp}.json")
    detailed_results = []
    
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": float(trial.value) if trial.value else None,
            "params": trial.params,
            "state": str(trial.state),
            "datetime_start": str(trial.datetime_start),
            "datetime_complete": str(trial.datetime_complete),
            "duration": str(trial.duration) if trial.duration else None
        }
        detailed_results.append(trial_data)
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{algo_name}_optimization_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{algo_name.upper()} Hyperparameter Optimization Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Environment: {tuning_args.env}\n")
        f.write(f"Task: {tuning_args.task}\n")
        f.write(f"GPUs used: {tuning_args.gpu_ids or [tuning_args.cuda_id]}\n")
        f.write(f"Number of parallel jobs: {tuning_args.n_jobs}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}\n")
        f.write(f"Pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}\n")
        f.write(f"Failed trials: {len([t for t in study.trials if t.state == TrialState.FAIL])}\n\n")
        
        if study.best_trial:
            f.write("Best Trial Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Value: {study.best_trial.value:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in study.best_trial.params.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write(f"\nTrial Number: {study.best_trial.number}\n")
            f.write(f"Duration: {study.best_trial.duration}\n\n")
        
        f.write("Parameter Importance:\n")
        f.write("-" * 40 + "\n")
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                f.write(f"  {param}: {imp:.4f}\n")
        except:
            f.write("  Could not calculate parameter importance\n")
    
    print(f"\nResults saved:")
    print(f"  Best parameters: {best_params_file}")
    print(f"  Detailed results: {results_file}")
    print(f"  Summary: {summary_file}")
    
    return best_params_file


def main():
    tuning_args = get_tuning_args()
    
    # Set study name if not provided
    if tuning_args.study_name is None:
        tuning_args.study_name = f"{tuning_args.algo}_optimization"
    
    # Determine GPU IDs to use
    if tuning_args.gpu_ids:
        gpu_ids = tuning_args.gpu_ids
        print(f"Using GPUs: {gpu_ids}")
    else:
        gpu_ids = [tuning_args.cuda_id]
        print(f"Using default GPU: {tuning_args.cuda_id}")
    
    # Load algorithm configuration
    try:
        with open(tuning_args.config, 'r') as f:
            algo_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{tuning_args.config}' not found")
        print(f"Please create a configuration file with the following structure:")
        print(json.dumps({
            "example_algo": {
                "tunable_params": {
                    "param1": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                    "param2": {"type": "int", "low": 1, "high": 10}
                },
                "train_module": "run_example",
                "record_params": ["param1", "param2"],
                "default_params": {
                    "epoch": 1000,
                    "batch_size": 256
                }
            }
        }, indent=2))
        sys.exit(1)
    
    # Create base training arguments
    base_args = create_base_training_args(tuning_args.algo, 
                                         algo_configs.get(tuning_args.algo, {}), 
                                         tuning_args)
    
    # Create tuner
    print(f"Initializing {tuning_args.algo.upper()} hyperparameter tuner...")
    try:
        tuner = GenericHyperparameterTuner(tuning_args.algo, tuning_args.config, base_args, gpu_ids)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create study
    if tuning_args.storage:
        study = optuna.create_study(
            study_name=tuning_args.study_name,
            storage=tuning_args.storage,
            direction='maximize',
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            direction='maximize',
            study_name=tuning_args.study_name
        )
    
    # Add pruner
    study.pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=10
    )
    
    print(f"\nStarting {tuning_args.algo.upper()} hyperparameter optimization")
    print(f"Environment: {tuning_args.env}, Task: {tuning_args.task}")
    print(f"Number of trials: {tuning_args.n_trials}")
    print(f"Parallel jobs: {tuning_args.n_jobs}")
    print(f"Parameters to optimize: {list(tuner.config['tunable_params'].keys())}")
    print("=" * 70)
    
    # Run optimization
    study.optimize(tuner.objective, n_trials=tuning_args.n_trials, n_jobs=tuning_args.n_jobs)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"{tuning_args.algo.upper()} OPTIMIZATION COMPLETED!")
    print("=" * 70)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    if study.best_trial:
        print(f"\nBest trial value: {study.best_trial.value:.4f}")
        print("Best parameters:")
        for key, value in study.best_trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    # Save results
    best_params_file = save_optimization_results(study, tuning_args.output_dir, 
                                               tuning_args.algo, tuning_args)
    
    # Generate command for final training
    if study.best_trial:
        print(f"\nTo run final training with best parameters:")
        print(f"python run_{tuning_args.algo}.py --env {tuning_args.env} --task {tuning_args.task} \\")
        for key, value in study.best_trial.params.items():
            param_flag = f"--{key.replace('_', '-')}"
            print(f"  {param_flag} {value} \\")
        print(f"  --cuda-id {tuning_args.cuda_id}")


if __name__ == "__main__":
    main()