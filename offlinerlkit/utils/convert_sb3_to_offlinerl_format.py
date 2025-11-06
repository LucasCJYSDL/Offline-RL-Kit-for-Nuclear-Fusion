
import torch
from typing import Dict




def convert_sb3_to_offlinerl_format(sb3_state_dict: Dict[str, torch.Tensor], pol_hidden_dims: list, val_hidden_dims: list) -> Dict[str, torch.Tensor]:
    """
    Converts an SB3 state_dict to the offlinerlkit format.

    Args:
        sb3_state_dict: The state_dict from the SB3 model.
        pol_hidden_dims: A list of hidden layer dimensions for the policy network.
        val_hidden_dims: A list of hidden layer dimensions for the value network.

    Returns:
        offlinerl_state_dict: The converted state_dict.
    """
    offlinerl_state_dict = {}

    # 1. Convert policy network (mlp_extractor.policy_net -> actor_backbone)
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

    # 2. Convert action_net (the mean output layer)
    if 'action_net.weight' in sb3_state_dict:
        offlinerl_state_dict['actor.dist_net.mu.weight'] = sb3_state_dict[f'action_net.weight']
        offlinerl_state_dict['actor.dist_net.mu.bias'] = sb3_state_dict[f'action_net.bias']

    # 3. Convert log_std (independent parameter), not needed for deterministic evaluation
    if 'log_std' in sb3_state_dict:
        log_std = sb3_state_dict['log_std']
        if log_std.dim() == 2:
            avg_log_std = log_std.mean(dim=0, keepdim=True).T  # (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = avg_log_std
        else:
            offlinerl_state_dict['actor.dist_net.sigma_param'] = log_std.unsqueeze(-1)


    # 4. Convert value network (mlp_extractor.value_net -> critic_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(val_hidden_dims):
        weight_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.bias'

        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]

        layer_idx += 2  # Linear + ReLU

    # 5. Convert value_net (the value output layer)
    if 'value_net.weight' in sb3_state_dict:
        offlinerl_state_dict['critic1.last.weight'] = sb3_state_dict['value_net.weight']
        offlinerl_state_dict['critic1.last.bias'] = sb3_state_dict['value_net.bias']

    # For compatibility, also create critic2 (usually the same as critic1)
    for key in list(offlinerl_state_dict.keys()):
        if key.startswith("critic1."):
            critic2_key = key.replace("critic1.", "critic2.")
            offlinerl_state_dict[critic2_key] = offlinerl_state_dict[key].clone()

    return offlinerl_state_dict