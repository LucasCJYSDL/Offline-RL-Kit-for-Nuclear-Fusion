import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, TanhDiagGaussian, Actor
from stable_baselines3 import PPO


def convert_sb3_to_offlinerl_format(sb3_state_dict: dict, pol_hidden_dims: list, val_hidden_dims: list) -> dict:
    """
    Convert SB3 state_dict to offlinerlkit format

    Args:
        sb3_state_dict: SB3 model's state_dict (from model.policy.state_dict())
        pol_hidden_dims: List of hidden layer dimensions for policy network
        val_hidden_dims: List of hidden layer dimensions for value network

    Returns:
        offlinerl_state_dict: Converted state_dict
    """
    offlinerl_state_dict = {}
    
    # 1. 转换 policy network (mlp_extractor.policy_net -> actor_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(pol_hidden_dims):
        # SB3: mlp_extractor.policy_net.{layer_idx}.weight/bias
        # OfflineRL: actor.backbone.model.{i*2}.weight/bias
        weight_key_sb3 = f'mlp_extractor.policy_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.policy_net.{layer_idx}.bias'
        
        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'actor.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'actor.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]
        
        layer_idx += 2  # Linear + ReLU
    
    # 2. 转换 action_net (mean 输出层)
    if 'action_net.weight' in sb3_state_dict:
        offlinerl_state_dict['actor.dist_net.mu.weight'] = sb3_state_dict['action_net.weight']
        offlinerl_state_dict['actor.dist_net.mu.bias'] = sb3_state_dict['action_net.bias']
    
    # 3. Convert log_std (independent parameter)
    if 'log_std' in sb3_state_dict:
        log_std = sb3_state_dict['log_std']

        if log_std.dim() == 2:
            # gSDE: log_std shape is (latent_dim, action_dim)
            # Convert to fixed sigma_param: (action_dim, 1)
            # Strategy: take average across all latent dimensions
            avg_log_std = log_std.mean(dim=0, keepdim=True).T  # (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = avg_log_std
        else:
            # Standard PPO (1D tensor): log_std shape is (action_dim,)
            # Convert to sigma_param: (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = log_std.unsqueeze(-1)
    
    # 4. 转换 value network (mlp_extractor.value_net -> critic_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(val_hidden_dims):
        weight_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.bias'
        
        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]
        
        layer_idx += 2  # Linear + ReLU
    
    # 5. 转换 value_net (value 输出层)
    if 'value_net.weight' in sb3_state_dict:
        offlinerl_state_dict['critic1.last.weight'] = sb3_state_dict['value_net.weight']
        offlinerl_state_dict['critic1.last.bias'] = sb3_state_dict['value_net.bias']
    
    # For compatibility, also create critic2 (usually same as critic1)
    for key in list(offlinerl_state_dict.keys()):
        if key.startswith("critic1."):
            critic2_key = key.replace("critic1.", "critic2.")
            offlinerl_state_dict[critic2_key] = offlinerl_state_dict[key].clone()
    
    return offlinerl_state_dict


class ControllerBest:
    """
    Controller that loads the best model from best_model.zip and converts it to OfflineRL format
    """
    def __init__(self, args):
        # Set network architecture parameters (all experiments use [250, 250])
        self.pol_hidden_dims = [250, 250]
        self.val_hidden_dims = [250, 250]

        # Build best_model path
        cwd = os.getcwd()
        best_model_dir = os.path.join(cwd, args.actor_path, 'best_model')
        best_model_zip = os.path.join(best_model_dir, 'best_model.zip')
        converted_pth = os.path.join(best_model_dir, 'policy.pth')

        # Check if best_model.zip exists
        if not os.path.exists(best_model_zip):
            raise FileNotFoundError(
                f"Best model not found at {best_model_zip}\n"
                f"Please make sure the training used EvalCallback to save the best model."
            )

        # Check if already converted, if not then convert
        if not os.path.exists(converted_pth):
            print(f"Converting best model from {best_model_zip} to OfflineRL format...")
            self._convert_best_model(best_model_zip, converted_pth, args.device)
            print(f"Converted model saved to {converted_pth}")
        else:
            print(f"Using existing converted model at {converted_pth}")

        # Create actor network
        actor_backbone = MLP(input_dim=args.obs_dim, hidden_dims=args.hidden_dims)
        if not args.stochastic_actor:
            self.actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
        else:
            dist = TanhDiagGaussian(
                latent_dim=getattr(actor_backbone, "output_dim"),
                output_dim=args.action_dim,
                unbounded=True,
                conditioned_sigma=False,  # Use fixed sigma_param
                max_mu=args.max_action
            )
            self.actor = ActorProb(actor_backbone, dist, args.device)

        # Load converted weights
        checkpoint = torch.load(converted_pth, map_location=args.device)
        
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("critic"):
                continue
            state_dict[k.replace("actor.", "")] = v
        
        self.actor.load_state_dict(state_dict)
        self.actor.eval()  # evaluation only
        
        # important arguments
        self.deterministic_mode = args.deterministic_mode
        self.stochastic_actor = args.stochastic_actor
    
    def _convert_best_model(self, best_model_zip: str, output_pth: str, device):
        """
        Load SB3 model from best_model.zip and convert to OfflineRL format

        Args:
            best_model_zip: Path to best_model.zip
            output_pth: Path to output .pth file
            device: Device
        """
        # Load SB3 model (no environment needed)
        print(f"Loading SB3 model from {best_model_zip}...")
        model = PPO.load(best_model_zip, env=None, device=device)

        # Get policy's state_dict
        sb3_state_dict = model.policy.state_dict()

        # Convert to OfflineRL format
        print("Converting to OfflineRL format...")
        converted_state_dict = convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,
            self.val_hidden_dims
        )

        # Save converted model
        os.makedirs(os.path.dirname(output_pth), exist_ok=True)
        torch.save(converted_state_dict, output_pth)
        print(f"Conversion complete!")
    
    def act(self, obs):
        """
        Select action based on observation

        Args:
            obs: Observation

        Returns:
            action: Action
        """
        with torch.no_grad():
            if not self.stochastic_actor:
                action = self.actor(obs)
            else:
                dist = self.actor(obs)
                if self.deterministic_mode:
                    squashed_action, raw_action = dist.mode()
                else:
                    squashed_action, raw_action = dist.rsample()
                action = squashed_action
        
        return action.cpu().numpy()

