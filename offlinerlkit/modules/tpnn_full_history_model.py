"""Stateless, variable-length inference wrapper for trained TPNN models."""

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
)


class TPNNFullHistoryModel(nn.Module):
    """Run TPNN members on full, left-aligned histories.

    Context is intentionally managed outside the TPNN.  This wrapper never
    uses the model's scalar ``_time_step`` or internal ``_history_buffer``.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        microbatch_size: int = 4096,
        token_budget: Optional[int] = None,
        attention_budget: Optional[int] = None,
        expected_member_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        model_root = Path(model_path).expanduser().resolve()
        self.model_path = str(model_root)
        if not model_root.is_dir():
            raise FileNotFoundError(
                f"TPNN model directory does not exist: {self.model_path}"
            )
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive.")
        if expected_member_count is not None:
            if isinstance(expected_member_count, (bool, np.bool_)) or not (
                isinstance(expected_member_count, (int, np.integer))
            ):
                raise TypeError("expected_member_count must be an integer.")
            expected_member_count = int(expected_member_count)
            if expected_member_count < 1:
                raise ValueError("expected_member_count must be positive.")

        member_dirs = self._ordered_member_dirs(
            model_root, expected_member_count
        )
        self.expected_member_count = len(member_dirs)
        self.member_manifest = [
            self._build_member_manifest(member_idx, member_dir)
            for member_idx, member_dir in enumerate(member_dirs)
        ]

        # Do not use load_ensemble_from_parent_dir or
        # load_ensemble_from_list_of_log_dirs here.  Both ultimately apply a
        # lexicographic sort, which maps member index 2 to directory "10" in a
        # 25-member ensemble.  Loading one member at a time preserves the
        # audited numeric member order in ``member_manifest``.
        loaded_models = [
            load_model_from_log_dir(str(member_dir))
            for member_dir in member_dirs
        ]
        self.all_models = nn.ModuleList(loaded_models)
        self.device = torch.device(device)
        self.microbatch_size = int(microbatch_size)

        first = self.all_models[0]
        self.input_dim = int(first.input_dim)
        self.output_dim = int(first.output_dim)
        if not hasattr(first, "_block_size"):
            raise TypeError(
                f"Model {type(first).__name__} is not a TPNN with block_size."
            )
        self.block_size = int(first._block_size)
        # The row limit alone is unsafe for variable-length attention.  Keep
        # each dynamic batch within the already benchmarked worst-case work of
        # 1024 full-block histories, while allowing up to 4096 short histories.
        default_budget_rows = 1024
        if token_budget is None:
            token_budget = default_budget_rows * self.block_size
        if attention_budget is None:
            attention_budget = (
                default_budget_rows * self.block_size * self.block_size
            )
        for budget_name, budget_value, minimum in (
            ("token_budget", token_budget, self.block_size),
            (
                "attention_budget",
                attention_budget,
                self.block_size * self.block_size,
            ),
        ):
            if isinstance(budget_value, (bool, np.bool_)) or not isinstance(
                budget_value, (int, np.integer)
            ):
                raise TypeError(f"{budget_name} must be an integer.")
            if int(budget_value) < minimum:
                raise ValueError(
                    f"{budget_name} must be at least {minimum} so one "
                    "maximum-length history always fits."
                )
        self.token_budget = int(token_budget)
        self.attention_budget = int(attention_budget)

        for member in self.all_models:
            if type(member).__name__ != "TPNN":
                raise TypeError(
                    "TPNNFullHistoryModel only accepts TPNN members, got "
                    f"{type(member).__name__}."
                )
            if int(member.input_dim) != self.input_dim:
                raise ValueError("TPNN members have different input dimensions.")
            if int(member.output_dim) != self.output_dim:
                raise ValueError(
                    "TPNN members have different output dimensions."
                )
            if int(member._block_size) != self.block_size:
                raise ValueError("TPNN members have different block sizes.")
            member.to(self.device)
            member.eval()

        self.num_ensemble = len(self.all_models)
        self.member_list = np.arange(self.num_ensemble, dtype=np.int64)

    @staticmethod
    def _ordered_member_dirs(
        model_root: Path,
        expected_member_count: Optional[int],
    ) -> List[Path]:
        numeric_names = {
            child.name
            for child in model_root.iterdir()
            if child.is_dir() and child.name.isdigit()
        }
        if expected_member_count is None:
            if not numeric_names:
                raise ValueError(
                    f"No numeric TPNN member directories found in {model_root}."
                )
            expected_member_count = len(numeric_names)

        expected_names = {
            str(member_idx) for member_idx in range(expected_member_count)
        }
        if numeric_names != expected_names:
            missing = sorted(
                expected_names - numeric_names, key=lambda name: int(name)
            )
            unexpected = sorted(
                numeric_names - expected_names, key=lambda name: int(name)
            )
            raise ValueError(
                "TPNN member directory set must be exactly "
                f"0..{expected_member_count - 1}; missing={missing}, "
                f"unexpected={unexpected}."
            )
        return [
            model_root / str(member_idx)
            for member_idx in range(expected_member_count)
        ]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _configured_seed(config: Any, member_idx: int) -> Optional[int]:
        configured_seeds = []
        for config_key in ("seed", "data_module.seed"):
            value = OmegaConf.select(config, config_key, default=None)
            if value is not None:
                try:
                    seed = int(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Member {member_idx} has a non-integer "
                        f"{config_key}: {value!r}."
                    ) from error
                configured_seeds.append((config_key, seed))

        for config_key, seed in configured_seeds:
            if seed != member_idx:
                raise ValueError(
                    f"Member directory {member_idx} has {config_key}={seed}; "
                    "the configured training seed must match its numeric "
                    "member directory."
                )
        if not configured_seeds:
            return None
        unique_seeds = {seed for _, seed in configured_seeds}
        if len(unique_seeds) != 1:
            raise ValueError(
                f"Member {member_idx} has inconsistent configured seeds: "
                f"{configured_seeds}."
            )
        return configured_seeds[0][1]

    @staticmethod
    def _unique_checkpoint(member_dir: Path, member_idx: int) -> Path:
        checkpoint_dirs = sorted(
            path
            for path in member_dir.rglob("checkpoints")
            if path.is_dir()
        )
        if len(checkpoint_dirs) != 1:
            raise ValueError(
                f"Member {member_idx} must contain exactly one checkpoints "
                f"directory, found {len(checkpoint_dirs)}."
            )
        checkpoint_entries = list(checkpoint_dirs[0].iterdir())
        checkpoints = [
            path
            for path in checkpoint_entries
            if path.is_file() and path.suffix == ".ckpt"
        ]
        if len(checkpoint_entries) != 1 or len(checkpoints) != 1:
            raise ValueError(
                f"Member {member_idx} must contain exactly one checkpoint "
                f"file, found entries={[path.name for path in checkpoint_entries]}."
            )
        return checkpoints[0].resolve()

    @classmethod
    def _build_member_manifest(
        cls, member_idx: int, member_dir: Path
    ) -> Dict[str, Any]:
        config_path = member_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Member {member_idx} config not found: {config_path}."
            )
        config = OmegaConf.load(config_path)
        training_seed = cls._configured_seed(config, member_idx)
        checkpoint_path = cls._unique_checkpoint(member_dir, member_idx)
        checkpoint_match = re.fullmatch(
            r"epoch=(\d+)-step=(\d+)\.ckpt", checkpoint_path.name
        )
        if checkpoint_match is None:
            raise ValueError(
                f"Cannot parse epoch and step from member {member_idx} "
                f"checkpoint name: {checkpoint_path.name}."
            )

        return {
            "member_idx": int(member_idx),
            "source_directory": str(member_dir.resolve()),
            "training_seed": training_seed,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_match.group(1)),
            "checkpoint_step": int(checkpoint_match.group(2)),
            "checkpoint_size": int(checkpoint_path.stat().st_size),
            "config_sha256": cls._sha256(config_path),
            "checkpoint_sha256": cls._sha256(checkpoint_path),
        }

    def validate_member_indices(
        self,
        member_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> np.ndarray:
        """Validate a non-empty, unique subset of source ensemble members."""
        if member_indices is None:
            return self.member_list.copy()
        raw_indices = np.asarray(member_indices)
        if raw_indices.ndim != 1 or raw_indices.size == 0:
            raise ValueError(
                "member_indices must be a non-empty one-dimensional sequence."
            )
        if raw_indices.dtype == np.bool_ or not np.issubdtype(
            raw_indices.dtype, np.integer
        ):
            raise TypeError("member_indices must contain only integers.")
        indices = raw_indices.astype(np.int64, copy=False)
        if np.unique(indices).size != indices.size:
            raise ValueError("member_indices must not contain duplicates.")
        invalid = (indices < 0) | (indices >= self.num_ensemble)
        if np.any(invalid):
            raise IndexError(
                "TPNN member indices are out of range: "
                f"{indices[invalid].tolist()}."
            )
        return indices.copy()

    def get_member_manifest(
        self,
        member_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a detached, JSON-serializable checkpoint manifest."""
        indices = self.validate_member_indices(member_indices)
        return copy.deepcopy(
            [self.member_manifest[int(idx)] for idx in indices]
        )

    def save_member_manifest(
        self,
        output_path: Union[str, Path],
        member_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> Path:
        """Write the exact member/checkpoint identity used by this wrapper."""
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        indices = self.validate_member_indices(member_indices)
        payload = {
            "model_path": self.model_path,
            "source_num_members": self.num_ensemble,
            "active_member_indices": indices.tolist(),
            "num_members": int(indices.size),
            "members": self.get_member_manifest(indices),
        }
        with output.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
        return output

    def train(self, mode: bool = True):
        """Keep the frozen rollout dynamics in evaluation mode."""
        super().train(False)
        for member in self.all_models:
            member.eval()
        return self

    def random_member_idxs(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.member_list, size=int(batch_size))

    @staticmethod
    def _output_scaling(member, output: torch.Tensor) -> torch.Tensor:
        scaling = getattr(member.normalizer, "1_scaling")
        if not isinstance(scaling, torch.Tensor):
            scaling = torch.as_tensor(scaling, dtype=output.dtype)
        return scaling.to(device=output.device, dtype=output.dtype)

    def validate_member_idx(self, member_idx: int) -> int:
        """Validate and normalize a scalar internal ensemble member index."""
        if isinstance(member_idx, (bool, np.bool_)) or not isinstance(
            member_idx, (int, np.integer)
        ):
            raise TypeError("member_idx must be an integer scalar.")
        normalized_idx = int(member_idx)
        if normalized_idx < 0 or normalized_idx >= self.num_ensemble:
            raise IndexError(
                f"member_idx {normalized_idx} is outside [0, "
                f"{self.num_ensemble})."
            )
        return normalized_idx

    def _validate_padded_inputs(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = np.asarray(padded_inputs, dtype=np.float32)
        lengths = np.asarray(sequence_lengths, dtype=np.int64).reshape(-1)
        if inputs.ndim != 3:
            raise ValueError(
                "padded_inputs must have shape [batch, sequence, input_dim]."
            )
        if inputs.shape[0] == 0:
            raise ValueError("padded_inputs batch must not be empty.")
        if inputs.shape[0] != lengths.shape[0]:
            raise ValueError(
                "padded_inputs and sequence_lengths have different batches."
            )
        if inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"TPNN input width is {inputs.shape[2]}, expected "
                f"{self.input_dim}."
            )
        if np.any(lengths < 1) or np.any(lengths > inputs.shape[1]):
            raise ValueError("A sequence length is outside the padded tensor.")
        if inputs.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {inputs.shape[1]} exceeds TPNN block_size "
                f"{self.block_size}."
            )
        return inputs, lengths

    def plan_microbatches(
        self,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
    ) -> List[np.ndarray]:
        """Sort histories by length and pack them under compute budgets.

        Each returned array contains row indices into the caller's batch.
        Apart from the maximum row count, a batch must satisfy both
        ``rows * max_length <= token_budget`` and
        ``rows * max_length**2 <= attention_budget``.  The latter mirrors the
        materialized causal-attention matrix used by the trained TPNN.
        """
        lengths = np.asarray(sequence_lengths, dtype=np.int64).reshape(-1)
        if lengths.size == 0:
            return []
        if np.any(lengths < 1) or np.any(lengths > self.block_size):
            raise ValueError(
                "A sequence length is outside [1, block_size]."
            )

        order = np.argsort(lengths, kind="stable")
        sorted_lengths = lengths[order]
        batches: List[np.ndarray] = []
        start = 0
        while start < order.size:
            end = start
            hard_end = min(start + self.microbatch_size, order.size)
            while end < hard_end:
                candidate_rows = end - start + 1
                candidate_max_length = int(sorted_lengths[end])
                padded_tokens = candidate_rows * candidate_max_length
                attention_cells = (
                    candidate_rows
                    * candidate_max_length
                    * candidate_max_length
                )
                if (
                    padded_tokens > self.token_budget
                    or attention_cells > self.attention_budget
                ):
                    break
                end += 1
            if end == start:
                # Constructor validation guarantees a single history of up to
                # block_size fits both budgets.
                raise RuntimeError(
                    "Dynamic TPNN budgets cannot fit one history."
                )
            batches.append(order[start:end])
            start = end
        return batches

    def _validate_planned_microbatch(
        self, sequence_lengths: np.ndarray
    ) -> None:
        rows = int(sequence_lengths.shape[0])
        max_length = int(sequence_lengths.max())
        if rows > self.microbatch_size:
            raise ValueError(
                f"Microbatch has {rows} rows, maximum is "
                f"{self.microbatch_size}."
            )
        if rows * max_length > self.token_budget:
            raise ValueError("Microbatch exceeds the configured token budget.")
        if rows * max_length * max_length > self.attention_budget:
            raise ValueError(
                "Microbatch exceeds the configured attention budget."
            )

    def _forward_member_last_token(
        self,
        member_idx: int,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return physical-space delta mean/std for only the last token."""
        member = self.all_models[member_idx]
        normalized_inputs = member.normalizer.normalize(inputs, 0)
        if member._input_mask:
            normalized_inputs = normalized_inputs * member._mask.to(
                device=inputs.device, dtype=inputs.dtype
            )

        encoded = member._encoder(normalized_inputs)
        if member._use_layer_norm:
            encoded = member._layer_norm(encoded)
        if member._use_positional_encoding:
            encoded = (
                encoded
                + member._positional_encoding[:, : encoded.shape[1], :]
            )
        for transformer_block in member._transformer_blocks:
            encoded = transformer_block(encoded)

        row_indices = torch.arange(inputs.shape[0], device=inputs.device)
        last_hidden = encoded[row_indices, lengths - 1]
        mean_normalized, logvar_normalized = member._decoder(last_hidden)
        mean = member.normalizer.unnormalize(mean_normalized, 1)
        std = self._output_scaling(
            member, logvar_normalized
        ) * torch.exp(0.5 * logvar_normalized)
        return mean, std

    def _inputs_to_device(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.as_tensor(
            np.ascontiguousarray(padded_inputs),
            dtype=torch.float32,
            device=self.device,
        )
        lengths = torch.as_tensor(
            np.ascontiguousarray(sequence_lengths),
            dtype=torch.long,
            device=self.device,
        )
        return inputs, lengths

    def _predict_member_microbatch(
        self,
        member_idx: int,
        padded_inputs: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs, lengths = self._inputs_to_device(
            padded_inputs, sequence_lengths
        )
        with torch.inference_mode():
            mean, std = self._forward_member_last_token(
                member_idx, inputs, lengths
            )
            combined = torch.cat((mean, std), dim=-1).cpu().numpy()
        output_dim = self.output_dim
        return (
            np.asarray(combined[:, :output_dim], dtype=np.float32),
            np.asarray(combined[:, output_dim:], dtype=np.float32),
        )

    def _predict_all_members_microbatch(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs, lengths = self._inputs_to_device(
            padded_inputs, sequence_lengths
        )
        means = []
        stds = []
        with torch.inference_mode():
            for member_idx in range(self.num_ensemble):
                mean, std = self._forward_member_last_token(
                    member_idx, inputs, lengths
                )
                means.append(mean)
                stds.append(std)
            combined = torch.cat(
                (torch.stack(means), torch.stack(stds)), dim=-1
            ).cpu().numpy()
        return (
            np.asarray(combined[:, :, : self.output_dim], dtype=np.float32),
            np.asarray(combined[:, :, self.output_dim :], dtype=np.float32),
        )

    def predict_selected_with_uncertainty(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
        states: np.ndarray,
        member_indices: Union[np.ndarray, Sequence[int]],
        uncertainty_mode: str,
        ensemble_member_indices: Optional[
            Union[np.ndarray, Sequence[int]]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run all members while returning only selected deltas and penalty.

        The raw padded history is copied to the GPU once for the whole
        microbatch and shared by all ensemble members.  Uncertainty is reduced
        on the GPU; selected mean/std and uncertainty are copied back together.
        """
        inputs_np, lengths_np = self._validate_padded_inputs(
            padded_inputs, sequence_lengths
        )
        self._validate_planned_microbatch(lengths_np)
        states_np = np.asarray(states, dtype=np.float32)
        indices_np = np.asarray(member_indices, dtype=np.int64).reshape(-1)
        active_member_indices = self.validate_member_indices(
            ensemble_member_indices
        )
        expected_state_shape = (inputs_np.shape[0], self.output_dim)
        if states_np.shape != expected_state_shape:
            raise ValueError(
                f"states must have shape {expected_state_shape}, got "
                f"{states_np.shape}."
            )
        if indices_np.shape[0] != inputs_np.shape[0]:
            raise ValueError("member_indices batch does not match inputs.")
        assigned_to_active_member = np.isin(
            indices_np, active_member_indices
        )
        if not np.all(assigned_to_active_member):
            raise ValueError(
                "Selected trajectory member is outside the active "
                "uncertainty ensemble: "
                f"{indices_np[~assigned_to_active_member][:10].tolist()}."
            )
        if uncertainty_mode not in {
            "aleatoric",
            "pairwise-diff",
            "ensemble_std",
        }:
            raise ValueError(
                f"Unknown uncertainty mode: {uncertainty_mode!r}."
            )

        inputs, lengths = self._inputs_to_device(inputs_np, lengths_np)
        states_tensor = torch.as_tensor(
            np.ascontiguousarray(states_np),
            dtype=torch.float32,
            device=self.device,
        )
        indices_tensor = torch.as_tensor(
            np.ascontiguousarray(indices_np),
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            selected_means = torch.empty(
                expected_state_shape,
                dtype=inputs.dtype,
                device=self.device,
            )
            selected_stds = torch.empty_like(selected_means)
            if uncertainty_mode == "aleatoric":
                uncertainty = torch.zeros(
                    inputs.shape[0],
                    dtype=inputs.dtype,
                    device=self.device,
                )
                next_state_means = None
            else:
                uncertainty = None
                next_state_means = []

            for member_idx in active_member_indices:
                member_idx = int(member_idx)
                mean, std = self._forward_member_last_token(
                    member_idx, inputs, lengths
                )
                assigned = indices_tensor == member_idx
                if torch.any(assigned):
                    selected_means[assigned] = mean[assigned]
                    selected_stds[assigned] = std[assigned]
                if uncertainty_mode == "aleatoric":
                    uncertainty = torch.maximum(
                        uncertainty, torch.linalg.vector_norm(std, dim=-1)
                    )
                else:
                    next_state_means.append(mean + states_tensor)

            if uncertainty_mode != "aleatoric":
                ensemble_means = torch.stack(next_state_means, dim=0)
                if uncertainty_mode == "pairwise-diff":
                    centered = (
                        ensemble_means
                        - ensemble_means.mean(dim=0, keepdim=True)
                    )
                    uncertainty = torch.linalg.vector_norm(
                        centered, dim=-1
                    ).amax(dim=0)
                else:
                    uncertainty = torch.sqrt(
                        ensemble_means.var(dim=0, unbiased=False).mean(dim=-1)
                    )

            combined = torch.cat(
                (
                    selected_means,
                    selected_stds,
                    uncertainty.unsqueeze(-1),
                ),
                dim=-1,
            ).cpu().numpy()
        return (
            np.asarray(
                combined[:, : self.output_dim], dtype=np.float32
            ),
            np.asarray(
                combined[:, self.output_dim : 2 * self.output_dim],
                dtype=np.float32,
            ),
            np.asarray(combined[:, -1], dtype=np.float32),
        )

    def predict_padded(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return physical-space delta means/stds with shape ``[M,B,D]``."""
        inputs, lengths = self._validate_padded_inputs(
            padded_inputs, sequence_lengths
        )

        means_out = np.empty(
            (self.num_ensemble, inputs.shape[0], self.output_dim),
            dtype=np.float32,
        )
        stds_out = np.empty_like(means_out)
        for rows in self.plan_microbatches(lengths):
            max_length = int(lengths[rows].max())
            means, stds = self._predict_all_members_microbatch(
                inputs[rows, :max_length], lengths[rows]
            )
            means_out[:, rows] = means
            stds_out[:, rows] = stds
        return means_out, stds_out

    def predict_all_members(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Explicit alias for all-member ``[M,B,D]`` inference."""
        return self.predict_padded(padded_inputs, sequence_lengths)

    def predict_member_padded(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
        member_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return one member's physical delta mean/std with shape ``[B,D]``.

        Each member performs its own input normalization, output
        unnormalization, and log-variance scaling.
        """
        normalized_member_idx = self.validate_member_idx(member_idx)
        inputs, lengths = self._validate_padded_inputs(
            padded_inputs, sequence_lengths
        )
        means_out = np.empty(
            (inputs.shape[0], self.output_dim), dtype=np.float32
        )
        stds_out = np.empty_like(means_out)
        for rows in self.plan_microbatches(lengths):
            max_length = int(lengths[rows].max())
            means, stds = self._predict_member_microbatch(
                normalized_member_idx,
                inputs[rows, :max_length],
                lengths[rows],
            )
            means_out[rows] = means
            stds_out[rows] = stds
        return means_out, stds_out

    def predict_one_member(
        self,
        padded_inputs: np.ndarray,
        sequence_lengths: Union[np.ndarray, Sequence[int]],
        member_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for selected single-member ``[B,D]`` inference."""
        return self.predict_member_padded(
            padded_inputs, sequence_lengths, member_idx
        )
