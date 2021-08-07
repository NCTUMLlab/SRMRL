"""GMMMLPModule."""
import abc

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhGMM
from garage.torch.modules.mlp_module import MLPModule
from garage.torch import NonLinearity

import copy

class GMMMLPBaseModule(nn.Module):
    """Base of GMMMLPModel.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=TanhGMM):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers['min_std_param']
        self._max_std_param = buffers['max_std_param']

    @abc.abstractmethod
    def _get_mix_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            TanhGMM distribution

        """
        mix, mean, log_std_uncentered = self._get_mix_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mix, mean, std)

        return dist

class GMMMLPModule(nn.Module):
    """GMMMLPModule.
    """

    def __init__(self,
                 num_comps,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._layers = nn.ModuleList()
        self._output_dim = output_dims
        self._num_comps = num_comps

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, num_comps)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, num_comps)
        output_b_inits_mean = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits[0], num_comps)
        output_b_inits_log_std = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits[1], num_comps) 
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, num_comps)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._means_output_layers = nn.ModuleList()
        self._log_std_output_layers = nn.ModuleList()

        for i in range(num_comps):
            output_layer = self._get_output_layer(
                prev_size,
                output_dims[i],
                output_w_inits[i],
                output_b_inits_mean[i],
                output_nonlinearities[i]
            )

            self._means_output_layers.append(output_layer)

            output_layer = self._get_output_layer(
                prev_size,
                output_dims[i],
                output_w_inits[i],
                output_b_inits_log_std[i],
                output_nonlinearities[i]
            )

            self._log_std_output_layers.append(output_layer)
        
        self._mix_output_layer = self._get_output_layer(
            prev_size,
            num_comps,
            output_w_inits[0],
            output_b_inits[0],
            output_nonlinearities[0]
        )
        

    def _get_output_layer(self, prev_size, output_dim, output_w_init, output_b_init, output_nonlinearity):
        output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, output_dim)
        output_w_init(linear_layer.weight)
        output_b_init(linear_layer.bias)
        output_layer.add_module('linear', linear_layer)

        if output_nonlinearity:
            output_layer.add_module('non_linearity',
                                    NonLinearity(output_nonlinearity))
        
        return output_layer

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, num_comps):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            num_comps (int): number of head

        Returns:
            list: list of variables (length of num_comps)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to num_comps

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * num_comps
            if len(var) == num_comps:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'num_comps ({}), but {} provided.')
            raise ValueError(msg.format(var_name, num_comps, var))
        return [copy.deepcopy(var) for _ in range(num_comps)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        mix = self._mix_output_layer(x).reshape(-1, self._num_comps)

        means = torch.vstack(
            [output_layer(x) for output_layer in self._means_output_layers]
        ).view(-1, self._num_comps, self._output_dim)

        log_stds = torch.vstack(
            [output_layer(x) for output_layer in self._log_std_output_layers]
        ).view(-1, self._num_comps, self._output_dim)

        return (mix, means, log_stds)


class GMMMLPMultiCompsModule(GMMMLPBaseModule):
    """GMMMLPMultiCompsModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 num_comps,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=TanhGMM):
        super(GMMMLPMultiCompsModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._shared_mix_mean_log_std_network = GMMMLPModule(
            num_comps=num_comps,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                lambda x: nn.init.constant_(x, self._init_std.item())
            ],
            layer_normalization=self._layer_normalization)

    def _get_mix_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mixture coefficient of GMM distribution.
            torch.Tensor: The means of GMM distribution.
            torch.Tensor: The variances of GMM distribution.

        """
        return self._shared_mix_mean_log_std_network(*inputs)
