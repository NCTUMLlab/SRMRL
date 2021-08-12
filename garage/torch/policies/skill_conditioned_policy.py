import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from garage.torch import global_device, product_of_gaussians


# pylint: disable=attribute-defined-outside-init
# pylint does not recognize attributes initialized as buffers in constructor
class SkillConditionedPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In SRMRL, policies are conditioned on current state, a latent context
    (adaptation data) variable Z and a skill embedding. This inference network
    estimates theposterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z. The skill proposed network propose 
    skills based on current observation and z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.embeddings.ContextEncoder): Recurrent or
            permutation-invariant context encoder.
        skill_encoder (garage.torch.embeddings.ContextEncoder): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_information_bottleneck (bool): True if latent context is not
            deterministic; false otherwise.
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.
        num_skills (int): Dimension of skill embeddings
        dist_class (string): Distribution of skill
        soft (bool): Whether or not to use soft skill embeddings
        gmm (bool): Whether or not to use gmm policy

    """

    def __init__(self, latent_dim, context_encoder, skill_encoder, policy,
                use_information_bottleneck, use_next_obs, H=5, num_skills=10,
                dist_class='Categorical', soft=False, gmm=True):
        super().__init__()
        self._latent_dim = latent_dim
        self._context_encoder = context_encoder
        self._skill_encoder = skill_encoder
        self._policy = policy
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs = use_next_obs
        self._H = H
        self._num_skills=num_skills
        self._steps = 0
        self._skill_embed = None
        self._dist_class=dist_class
        self._soft=soft
        self._gmm=gmm

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.reset_belief()

    def reset_belief(self, num_tasks=1):
        r"""Reset :math:`q(z \| c)` to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        """
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self._latent_dim).to(global_device())
        if self._use_information_bottleneck:
            var = torch.ones(num_tasks, self._latent_dim).to(global_device())
        else:
            var = torch.zeros(num_tasks, self._latent_dim).to(global_device())
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_from_belief()
        # reset the context collected so far
        self._context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self._context_encoder.reset()

    def sample_from_belief(self):
        """Sample z using distributions from current means and variances."""
        if self._use_information_bottleneck:
            posteriors = [
                torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                    torch.unbind(self.z_means), torch.unbind(self.z_vars))
            ]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def update_context(self, timestep):
        """Append single transition to the current context.

        Args:
            timestep (garage._dtypes.TimeStep): Timestep containing transition
                information to be added to context.

        """
        o = torch.as_tensor(timestep.observation[None, None, ...],
                            device=global_device()).float()
        a = torch.as_tensor(timestep.action[None, None, ...],
                            device=global_device()).float()
        r = torch.as_tensor(np.array([timestep.reward])[None, None, ...],
                            device=global_device()).float()
        no = torch.as_tensor(timestep.next_observation[None, None, ...],
                             device=global_device()).float()

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        r"""Compute :math:`q(z \| c)` as a function of input context and sample new z.

        Args:
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        """
        params = self._context_encoder.forward(context)
        params = params.view(context.size(0), -1,
                             self._context_encoder.output_dim)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self._use_information_bottleneck:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [
                product_of_gaussians(m, s)
                for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
            ]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_from_belief()

    def infer_skill(self, _in, rsample=True):
        """Infer skill given input, if rsample make the sample differentiable
        """
        skill_out = self._skill_encoder.forward(_in)
        if self._dist_class == 'Categorical':
            skill_dist = D.Categorical(logits=skill_out)
            if self._soft:
                skill_embed = skill_dist.probs
            else:
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
                
        return skill_dist, skill_embed, skill_out

    # pylint: disable=arguments-differ
    def forward(self, obs, context, hierarchical=False, rsample=True):
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
        self.infer_posterior(context)
        self.sample_from_belief()
        task_z = self.z

        # task, batch
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        # infer skill
        skill_dist, skill_embed, skill_out = self.infer_skill(obs_z, rsample=rsample)
        
        if hierarchical:
            policy_in = torch.cat([obs_z, skill_embed.detach()], dim=1)
        else:
            policy_in = torch.cat([obs_z, skill_embed], dim=1)
        dist = self._policy(policy_in)[0]

        if self._gmm:
            pre_tanh, actions = dist.sample_with_pre_tanh_value()
        else:
            pre_tanh, actions = dist.rsample_with_pre_tanh_value()

        log_pi = dist.log_prob(value=actions, pre_tanh_value=pre_tanh)
        log_pi = log_pi.unsqueeze(1)
        mean = dist.mean.to('cpu').detach().numpy()
        log_std = (dist.variance**.5).log().to('cpu').detach().numpy()

        policy_out = [actions, mean, log_std, log_pi, pre_tanh]
        if self._gmm:
            policy_out.append(dist.mix_weights)
        
        return policy_out, task_z, skill_dist, skill_embed, skill_out

    def get_action(self, obs):
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
        z = self.z
        obs = torch.as_tensor(obs[None], device=global_device()).float()
        obs_in = torch.cat([obs, z], dim=1)
        if self._steps == 0:
            with torch.no_grad():
                skill_dist, skill_embed, skill_out = self.infer_skill(obs_in, rsample=False)
                self._skill_embed = skill_embed

        policy_in = torch.cat([obs_in, self._skill_embed], dim=1)
        action, info = self._policy.get_action(policy_in)

        self._steps = (self._steps + 1) % self._H
        return action, info

    def compute_kl_div(self):
        r"""Compute :math:`KL(q(z|c) \| p(z))`.

        Returns:
            float: :math:`KL(q(z|c) \| p(z))`.

        """
        prior = torch.distributions.Normal(
            torch.zeros(self._latent_dim).to(global_device()),
            torch.ones(self._latent_dim).to(global_device()))
        posteriors = [
            torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
                torch.unbind(self.z_means), torch.unbind(self.z_vars))
        ]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior)
            for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._context_encoder, self._policy, self._skill_encoder]

    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: Context values, with shape :math:`(X, N, C)`.
                X is the number of tasks. N is batch size. C is the combined
                size of observation, action, reward, and next observation if
                next observation is used in context. Otherwise, C is the
                combined size of observation, action, and reward.

        """
        return self._context
