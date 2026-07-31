"""COMBO policy using full-history TPNN synthetic rollouts."""

from offlinerlkit.policy.model_based.combo import COMBOPolicy
from offlinerlkit.policy.model_based.tpnn_rollout_mixin import (
    TPNNRolloutMixin,
)


class TPNNCOMBOPolicy(TPNNRolloutMixin, COMBOPolicy):
    """Keep COMBO's learning objective and replace only model rollouts."""

