"""Stable identities for frozen models and initialized policy modules."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import OmegaConf


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_identity_sha256(members: List[Dict[str, Any]]) -> str:
    """Hash checkpoint identities without machine-specific path strings."""
    identity_fields = (
        "member_idx",
        "training_seed",
        "checkpoint_epoch",
        "checkpoint_step",
        "checkpoint_size",
        "config_sha256",
        "checkpoint_sha256",
    )
    identity = [
        {key: member.get(key) for key in identity_fields}
        for member in members
    ]
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_numeric_ensemble_manifest(
    model_path: Union[str, Path],
    expected_member_count: int,
) -> Dict[str, Any]:
    """Audit a numeric 0..N-1 ensemble without relying on loader ordering."""
    root = Path(model_path).expanduser().resolve(strict=True)
    expected_count = int(expected_member_count)
    if expected_count < 1:
        raise ValueError("expected_member_count must be positive.")
    numeric_names = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit()
    }
    expected_names = {str(idx) for idx in range(expected_count)}
    if numeric_names != expected_names:
        raise ValueError(
            "Ensemble member directories must be exactly "
            f"0..{expected_count - 1}; "
            f"missing={sorted(expected_names - numeric_names)}, "
            f"unexpected={sorted(numeric_names - expected_names)}."
        )

    members = []
    checkpoint_pattern = re.compile(
        r"epoch=(\d+)-step=(\d+)\.ckpt"
    )
    for member_idx in range(expected_count):
        member_dir = (root / str(member_idx)).resolve()
        config_path = member_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Member {member_idx} config not found: {config_path}."
            )
        checkpoints = sorted(member_dir.rglob("*.ckpt"))
        if len(checkpoints) != 1:
            raise ValueError(
                f"Member {member_idx} must contain exactly one checkpoint, "
                f"found {len(checkpoints)}."
            )
        checkpoint_path = checkpoints[0].resolve()
        match = checkpoint_pattern.fullmatch(checkpoint_path.name)
        if match is None:
            raise ValueError(
                "Cannot parse epoch/step from checkpoint "
                f"{checkpoint_path.name!r}."
            )
        config = OmegaConf.load(config_path)
        configured_seed = OmegaConf.select(
            config, "seed", default=None
        )
        training_seed: Optional[int] = None
        if configured_seed is not None:
            training_seed = int(configured_seed)
            if training_seed != member_idx:
                raise ValueError(
                    f"Member directory {member_idx} has seed="
                    f"{training_seed}."
                )
        members.append(
            {
                "member_idx": member_idx,
                "source_directory": str(member_dir),
                "training_seed": training_seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": int(match.group(1)),
                "checkpoint_step": int(match.group(2)),
                "checkpoint_size": checkpoint_path.stat().st_size,
                "config_sha256": sha256_file(config_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    return {
        "model_path": str(root),
        "num_members": expected_count,
        "ensemble_identity_sha256": manifest_identity_sha256(members),
        "members": members,
    }


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash an initialized module state independently of its device."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
