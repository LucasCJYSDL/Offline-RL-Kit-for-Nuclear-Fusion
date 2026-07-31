"""MOPO policy using full-history TPNN synthetic rollouts."""

from offlinerlkit.policy.model_based.mopo import MOPOPolicy
from offlinerlkit.policy.model_based.tpnn_rollout_mixin import (
    TPNNRolloutMixin,
)


class TPNNMOPOPolicy(TPNNRolloutMixin, MOPOPolicy):
    """Keep MOPO's learning objective and replace only model rollouts."""

