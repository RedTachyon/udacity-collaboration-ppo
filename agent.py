import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from models import BaseModel, MLPModel

from typing import Tuple

from utils import DataBatch


class BaseAgent:
    """This might be removed, since all models use the same agent.
    Might be useful for implementing different agents, e.g. not-learning ones"""

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_actions(self, obs_batch: Tensor,
                        deterministic: bool = False) -> Tuple:
        raise NotImplementedError

    def compute_single_action(self, obs: np.ndarray,
                              deterministic: bool = False):
        raise NotImplementedError

    def evaluate_actions(self, data_batch: DataBatch):
        raise NotImplementedError

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()


class Agent(BaseAgent):
    model: BaseModel

    def __init__(self, model: BaseModel):
        super().__init__(model)

    def compute_actions(self, obs_batch: Tensor,
                        deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            deterministic: whether to always take the best action

        Returns:
            action, logprob of the action, entropy
        """
        action_distribution: Normal
        with torch.no_grad():
            action_distribution, value = self.model(obs_batch)

        if deterministic:
            actions = action_distribution.loc
        else:
            actions = action_distribution.sample()

        # actions = actions.clamp(-1, 1)

        logprobs = action_distribution.log_prob(actions).sum(1)
        entropies = action_distribution.entropy().sum(1)

        return actions, logprobs, entropies, value

    def compute_single_action(self, obs: np.ndarray,
                              deterministic: bool = False) -> Tuple[np.ndarray, float, float, float]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: flat observation array in shape either
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, entropy
        """
        obs = torch.tensor([obs])

        with torch.no_grad():
            action, logprob, entropy, value = self.compute_actions(obs, deterministic)  # fix nans

        return action.cpu().numpy().ravel(), logprob.item(), entropy.item(), value.item()

    def evaluate_actions(self, data_batch: DataBatch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            data_batch: data collected from a Collector for this agent

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = data_batch['observations']
        action_batch = data_batch['actions']

        action_distribution, values = self.model(obs_batch)
        action_logprobs = action_distribution.log_prob(action_batch).sum(1)
        values = values.view(-1)
        entropies = action_distribution.entropy().sum(1)

        return action_logprobs, values, entropies


# class DDPGAgent(BaseAgent):


if __name__ == '__main__':
    model = MLPModel({})
    agent = Agent(model)

    action = agent.compute_single_action(np.random.randn(33).astype(np.float32))
