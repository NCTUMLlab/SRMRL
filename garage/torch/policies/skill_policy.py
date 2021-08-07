"""A policy used in training meta reinforcement learning algorithms.

It is used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement
Learning). The paper on PEARL can be found at https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from garage.torch import global_device


# pylint: disable=attribute-defined-outside-init
# pylint does not recognize attributes initialized as buffers in constructor
class SkillPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In PEARL, policies are conditioned on current state and a latent context
    (adaptation data) variable Z. This inference network estimates the
    posterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.embeddings.ContextEncoder): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_information_bottleneck (bool): True if latent context is not
            deterministic; false otherwise.
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.

    """

    def __init__(self, skill_encoder, policy, H=1, num_skills=10, dist_class='Categorical'):
        super().__init__()
        self._skill_encoder = skill_encoder
        self._policy = policy
        self._H = H
        self._num_skills=num_skills
        self._steps = 0
        self._skill_embed = None
        self._dist_class=dist_class

    def infer_skill(self, _in, rsample=True):
        skill_out = self._skill_encoder.forward(_in)
        if self._dist_class == 'Categorical':
            skill_dist = D.Categorical(logits=skill_out)
            if rsample:
                skill_embed = F.gumbel_softmax(skill_out, tau=1, hard=True)
            else:
                index = skill_out.max(dim=-1, keepdim=True)[1]
                skill_embed = torch.zeros_like(skill_out).scatter_(-1, index, 1.)
        else:
            skill_mean = skill_out[:, :self._num_skills]
            skill_sigma_squared = F.softplus(skill_out[:, self._num_skills:])
            skill_std = torch.sqrt(skill_sigma_squared)
            skill_dist = D.Normal(skill_mean, skill_std)
            if rsample:
                skill_embed = skill_dist.rsample()        
            else:
                skill_embed = skill_mean
                
        return skill_dist, skill_embed

    # pylint: disable=arguments-differ
    def forward(self, obs, rsample=True):
        """Given observations and context, get actions and probs from policy.

        Args:
            obs (torch.Tensor): Observation values, with shape
                :math:`(X, N, O)`. X is the number of tasks. N is batch size. O
                 is the size of the flattened observation space.
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        Returns:
            tuple:
                * torch.Tensor: Predicted action values.
                * np.ndarray: Mean of distribution.
                * np.ndarray: Log std of distribution.
                * torch.Tensor: Log likelihood of distribution.
                * torch.Tensor: Sampled values from distribution before
                    applying tanh transformation.
            torch.Tensor: z values, with shape :math:`(N, L)`. N is batch size.
                L is the latent dimension.

        """
        skill_dist, skill_embed = self.infer_skill(obs, rsample=rsample)
        
        policy_in = torch.cat([obs, skill_embed], dim=-1)
        policy_out = self._policy(policy_in)

        return policy_out, skill_dist, skill_embed

    def get_actions(self, obs):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values, with shape :math:`(1, O)`.
                O is the size of the flattened observation space.

        Returns:
            torch.Tensor: Output action value, with shape :math:`(1, A)`.
                A is the size of the flattened action space.
            dict:
                * np.ndarray[float]: Mean of the distribution.
                * np.ndarray[float]: Standard deviation of logarithmic values
                    of the distribution.

        """
        obs = torch.as_tensor(obs, device=global_device()).float()
        # print(obs.shape, obs)
        if self._steps == 0:
            with torch.no_grad():
                skill_dist, skill_embed = self.infer_skill(obs, rsample=False)
                self._skill_embed = skill_embed
        # print(skill_embed.shape, skill_embed)
        policy_in = torch.cat([obs, self._skill_embed], dim=-1)
        # print(policy_in.shape, policy_in)
        action, info = self._policy.get_actions(policy_in)
        # print(self._skill_embed.shape)
        # info['skill_embed'] = self._skill_embed.squeeze(0).to('cpu').detach().numpy()
        # print(info['skill_embed'].shape)
        self._steps = (self._steps + 1) % self._H
        return action, info

    def get_action(self, obs):
        obs = torch.as_tensor(obs[None], device=global_device()).float()
        action, info = self.get_actions(obs)
        return action, info

    def reset(self, do_resets=None):
        """only for rnn policy
        """
        pass
    
    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._policy, self._skill_encoder]

