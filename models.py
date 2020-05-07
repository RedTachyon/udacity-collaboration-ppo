from typing import Dict, Tuple, Callable, Iterable, Optional

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, MultivariateNormal
from torch.nn import functional as F

from utils import with_default_config, get_activation, get_activation_module, get_initializer, matrix_diag


class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = 'cpu'

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        # Output: action_dist, value
        raise NotImplementedError

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.device = 'cuda'

    def cpu(self):
        super().cpu()
        self.device = 'cpu'


class MLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        torch.manual_seed(0)

        default_config = {
            "input_size": 24,
            "num_actions": 2,
            "activation": "relu",

            "sigma0": 1.,

            "hidden_sizes": (64, 64),
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        layer_sizes = (input_size,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])
        self.policy_mu_head = nn.Linear(layer_sizes[-1], num_actions)

        self.v_hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.std = nn.Parameter(torch.ones(1, num_actions) * self.config["sigma0"])

        self.value_head = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:

        int_policy = x
        for layer in self.hidden_layers:
            int_policy = layer(int_policy)
            int_policy = self.activation(int_policy)

        action_mu = torch.tanh(self.policy_mu_head(int_policy))  # [-1, 1]

        int_value = x
        for layer in self.v_hidden_layers:
            int_value = layer(int_value)
            int_value = self.activation(int_value)

        value = self.value_head(int_value)

        # cov = matrix_diag(self.std)
        # action_distribution = MultivariateNormal(loc=action_mu, covariance_matrix=cov)
        action_distribution = Normal(loc=action_mu, scale=self.std)

        return action_distribution, value
