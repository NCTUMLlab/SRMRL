import torch
from torch.distributions import Distribution, Normal, Categorical, MixtureSameFamily, Independent

class TanhGMM(Distribution):
    """
    Represent distribution of X where
    X ~ tanh(Z)
    Z ~ GMM(mix, comp)
        - mix is a Categorical distribution given logits
        - comps are Gaussians given mu and sigma
    """
    def __init__(self, mix_logits, comp_means, comp_stds, epsilon=1e-6):
        # note : the sample from Categorical(multinomial) require real probs
        # so inf, nan and negative should be avoided via regularization
        self.mix = Categorical(logits=mix_logits)
        self.comp = Independent(Normal(comp_means, comp_stds), 1)
        self.gmm = MixtureSameFamily(self.mix, self.comp)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.gmm.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:

        note: the pre_tanh_value calulation is dangerous if input is 1
        it will return inf which will cause boom of learning
        So give the pre_tanh_value !!!
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2

        # pdf of tanhGMM distribution
        # gmm.log_prob return the log_prob of each n-D Gaussian(already sum over)
        # so we need to sum the log value to keep same dim
        return self.gmm.log_prob(pre_tanh_value) - torch.log(
           1 - value * value + self.epsilon
        ).sum(axis=1)

    def sample(self, sample_shape=torch.Size()):
        z = self.gmm.sample(sample_shape=torch.Size())
        return torch.tanh(z)

    def sample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.

        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.

        """
        z = self.gmm.sample(sample_shape)
        return z, torch.tanh(z)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self.gmm.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self.gmm.variance

    @property
    def mix_weights(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self.mix.probs