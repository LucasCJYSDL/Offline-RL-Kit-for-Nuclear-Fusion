import os
import torch
from stable_baselines3.common.callbacks import BaseCallback
from ..utils.convert_sb3_to_offlinerl_format import convert_sb3_to_offlinerl_format


class ConvertAndSaveCallback(BaseCallback):
    """
    Custom Callback to automatically convert weights when saving.
    """
    def __init__(self, save_path: str, pol_hidden_dims: list, val_hidden_dims: list, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.pol_hidden_dims = pol_hidden_dims  # New parameter
        self.val_hidden_dims = val_hidden_dims  # New parameter
        self.save_freq = save_freq

    def _init_callback(self) -> None:
        # Create save directory
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_converted_model()
        return True

    def _save_converted_model(self) -> None:
        """Saves the converted model."""
        if self.save_path is None:
            return

        # Get the state_dict from the SB3 model
        sb3_state_dict = self.model.policy.state_dict()

        # Convert to OfflineRL format
        converted_state_dict = convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,  # Pass the new parameter
            self.val_hidden_dims   # Pass the new parameter
        )

        # Save the converted model
        save_file = os.path.join(self.save_path, "policy.pth")
        torch.save(converted_state_dict, save_file)

        if self.verbose > 0:
            print(f"Converted model saved to {save_file}")