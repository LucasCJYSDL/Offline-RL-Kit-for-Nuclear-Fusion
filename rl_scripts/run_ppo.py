import argparse
import random
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import CallbackList
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.policy.model_free.ppo import PPOPolicy, BestModelConvertCallback
from offlinerlkit.utils.logger import Logger, make_log_dirs
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from offlinerlkit.callbacks import TensorBoardLoggingCallback, FusionSpecificCallback, TrainingProgressCallback


def get_args():
    parser = argparse.ArgumentParser(description="PPO for Nuclear Fusion Control")
    
    # 
    parser.add_argument("--algo-name", type=str, default="ppo")
    

    parser.add_argument("--output-dir", type=str, default="/home/scratch/jiayuc2/rl_out_off/test_bao/test", 
                       help="Specify output directory path, use default path if not specified")
    # parser.add_argument("--output-dir", type=str, default=None, 
    #                     help="Specify output directory path, use default path if not specified")
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-3) 
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=20 )
    parser.add_argument("--gamma", type=float, default=0.952)#0.952
    parser.add_argument("--gae-lambda", type=float, default=0.98)#0.98
    parser.add_argument("--clip-range", type=float, default=0.148)
    parser.add_argument("--ent-coef", type=float, default=0.0067)
    parser.add_argument("--vf-coef", type=float, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
 #   parser.add_argument("--hidden-dims", type=int, nargs='*', default=[250, 250])
    parser.add_argument("--pol-hidden-dims", type=int, nargs='*', default=[250, 250],
                       help="Policy network hidden layer dimensions")
    parser.add_argument("--val-hidden-dims", type=int, nargs='*', default=[250, 250],
                       help="Value network hidden layer dimensions")
    
    # training parameters
    parser.add_argument("--total-timesteps", type=int, default=5_010)#1500_000
    parser.add_argument("--eval-freq", type=int, default=2_400)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--save-freq", type=int, default=2_400)
    
    # environment parameter
    parser.add_argument("--max-episode-length", type=int, default=150) #200
    parser.add_argument("--warm_start_amount", type=int, default=4)
    parser.add_argument("--min_start_idx", type=int, default=4)
    parser.add_argument("--max_start_idx",  type=int, default=20)
    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="rotation") # betan_EFIT01
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=3)
    return parser.parse_args()
    
#action_space
class GymnasiumWrapper(gym.Env):
    """
    self environment -> Gymnasium compatiable
    """
    def __init__(self, custom_env,action_dim, obs_dim):
        super().__init__()
        self.env = custom_env

        # Set up the observation space 

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        
        obs = self.env.reset()
        info = {}

        # Handle torch.Tensor observations and flatten them to 1D
        obs = obs.cpu().numpy().flatten()

        return obs, info

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(1, -1)  # Add batch dimension (1, action_dim)
        result = self.env.step(action)
        # Old format: (obs, reward, done, info)
        obs, reward, done, info = result

        # Handle torch.Tensor observations and flatten them to 1D
        obs = obs.cpu().numpy().flatten()


        return obs, reward, done, False, info  # add truncated




def train(args=get_args()):

    args.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {args.device}")

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # offline rl data and env
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(
        args.env, args.task, args.device
    )
    
    args.obs_shape = (offline_data['observations'].shape[1],)
    args.action_dim = offline_data['actions'].shape[1]
    args.state_dim = len(offline_data['state_idxs'])
    args.control_dim = len(offline_data['action_idxs'])
    
    print(f"observation dimension: {args.obs_shape}")
    print(f"state dimension: {args.state_dim}")
    print(f"action dimension: {args.action_dim}")
    print(f"control dimension: {args.control_dim}")
    
    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        model_path=training_dyn_model_dir,
        device=args.device
    )
    

    termination_fn = env.is_done
    reward_fn = sa_processor.get_reward_new
    dynamics = EnsembleDynamics(
        dynamics_model,
        termination_fn,
        reward_fn,
        penalty_coef=0.0  
    )
    
    # create PPO policy
    print("Creating PPO policy...")

    # Adjust PPO parameters to ensure callbacks can work correctly
    adjusted_n_steps = min(args.n_steps, args.total_timesteps // 4)  # Ensure at least 4 rollouts
    adjusted_n_steps = max(adjusted_n_steps, 256)  


    if args.output_dir is not None:
        base_dir = args.output_dir
        param_signature = (f"lr{args.learning_rate}_"
                  f"steps{args.n_steps}_"
                  f"batch{args.batch_size}_"
                  f"epochs{args.n_epochs}_"
                  f"gamma{args.gamma}_"
                  f"gaelambda{args.gae_lambda}_"
                  f"clip{args.clip_range}_"
                  f"ent{args.ent_coef}_"
                  f"vf{args.vf_coef}_"
                  f"maxgrad{args.max_grad_norm}_"
                  f"timesteps{args.total_timesteps}_"
                  #f"hidden{'x'.join(map(str, args.hidden_dims))}")
                  f"pol{'x'.join(map(str, args.pol_hidden_dims))}_"
                  f"val{'x'.join(map(str, args.val_hidden_dims))}")
        output_dir = os.path.join(base_dir, f"{args.task}_{args.algo_name}_seed{args.seed}",param_signature)
        os.makedirs(output_dir, exist_ok=True)
        
    else:
        output_dir = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    
    tb_log_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    
    policy = PPOPolicy(
        dynamics=dynamics,
        state_idxs=offline_data['state_idxs'],
        action_idxs=offline_data['action_idxs'],
        sa_processor=sa_processor,
        offline_data=offline_data,
        device=str(args.device),
        learning_rate=args.learning_rate,
        n_steps=adjusted_n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        #hidden_dims=args.hidden_dims,
        pol_hidden_dims=args.pol_hidden_dims,  
        val_hidden_dims=args.val_hidden_dims,  
        tensorboard_log=tb_log_dir  
    )

    policy.env.max_episode_length = args.max_episode_length

    # Create logger configuration
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(output_dir, output_config)
    
    print(f"output direction: {output_dir}")

    print("Starting training...")
    print(f"total_timesteps: {args.total_timesteps}")
    print(f"evaluation frequency: {args.eval_freq}")
    print(f"save frequency: {args.save_freq}")

    print("creating evaluation environment...")
    eval_env_raw = env  

    obs_dim =  offline_data['observations'].shape[1]
    action_dim = offline_data['act_dim']

    eval_env = GymnasiumWrapper(eval_env_raw, action_dim=action_dim, obs_dim=obs_dim)

    # Use Monitor wrapper to avoid SB3 warnings and ensure observation space consistency
    from stable_baselines3.common.monitor import Monitor
    eval_env = Monitor(eval_env)

    callbacks = []

   
    tb_callback = TensorBoardLoggingCallback(verbose=1, log_freq=1000)
    callbacks.append(tb_callback)
    
    fusion_callback = FusionSpecificCallback(
        eval_env=eval_env,
        verbose=1,
        eval_freq=args.eval_freq,
        n_eval_episodes=5
    )
    callbacks.append(fusion_callback)
    
    progress_callback = TrainingProgressCallback(
        save_path=output_dir,
        verbose=1,
        log_freq=5000
    )
    callbacks.append(progress_callback)

    # evaluation callback
    if args.eval_freq > 0:
        best_model_convert_callback = BestModelConvertCallback(
            eval_env=eval_env,
            save_path=os.path.join(output_dir, "checkpoint"),
            pol_hidden_dims=args.pol_hidden_dims,
            val_hidden_dims=args.val_hidden_dims,
            n_eval_episodes=args.eval_episodes,
            eval_freq=args.eval_freq,
            log_path=os.path.join(output_dir, "evaluations"),
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(best_model_convert_callback)
        print(f"Best model checkpoint callback configured: Evaluating every {args.eval_freq} steps for {args.eval_episodes} episodes, saving best model to checkpoint/policy.pth")



    # combine callbacks
    callback_list = CallbackList(callbacks) if callbacks else None

    # Training loop
    start_time = time.time()

    try:
        policy.model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        print("")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

    # Training complete
    total_time = time.time() - start_time
    print(f"total time: {total_time:.2f}seconds")

    # Save the final model
    final_model_path = os.path.join(output_dir, "model", "policy")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    policy.save(final_model_path)

    logger.close()
    print("# training complete!")


if __name__ == "__main__":
    train()
