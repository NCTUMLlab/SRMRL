"""PyTorch Policies."""
from garage.torch.policies.categorical_cnn_policy import CategoricalCNNPolicy
from garage.torch.policies.context_conditioned_policy import (
    ContextConditionedPolicy)
from garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from garage.torch.policies.discrete_cnn_policy import DiscreteCNNPolicy
from garage.torch.policies.discrete_qf_argmax_policy import (
    DiscreteQFArgmaxPolicy)
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.torch.policies.policy import Policy
from garage.torch.policies.tanh_gaussian_mlp_policy import (
    TanhGaussianMLPPolicy)
from garage.torch.policies.skill_conditioned_policy import (
    SkillConditionedPolicy)
from garage.torch.policies.skill_policy import (
    SkillPolicy)
from garage.torch.policies.tanh_gmm_mlp_policy import (
    TanhGMMMLPPolicy)
__all__ = [
    'CategoricalCNNPolicy',
    'DeterministicMLPPolicy',
    'DiscreteCNNPolicy',
    'DiscreteQFArgmaxPolicy',
    'GaussianMLPPolicy',
    'Policy',
    'TanhGaussianMLPPolicy',
    'ContextConditionedPolicy',
    'SkillConditionedPolicy',
    'SkillPolicy',
    'TanhGMMMLPPolicy'
]
