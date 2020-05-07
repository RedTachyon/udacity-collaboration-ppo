from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from unityagents import UnityEnvironment
from utils import with_default_config, get_optimizer, DataBatch, discount_rewards_to_go, Timer, write_dict, \
    get_episode_rewards, minibatches, Batcher, index_data, transpose_batch, DataBatchT, get_episode_lens


def batch_to_gpu(data_batch: DataBatch) -> DataBatch:
    new_batch = {}
    for key in data_batch:
        new_batch[key] = data_batch[key].cuda()
    return new_batch


class PPOptimizer:
    """
    A class that holds two (or a different number of) agents, and is responsible for performing the weight updates,
    using data collected by the collector

    The set of agents should not be changed. The state_dict should be alright to be loaded?
    """

    def __init__(self, agents: Dict[str, Agent],
                 config: Dict[str, Any]):

        self.agents = agents

        default_config = {
            # GD settings
            "optimizer": "adam",
            "optimizer_kwargs": {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-7,
                "weight_decay": 0,
                "amsgrad": False
            },

            # "batch_size": 64,
            "minibatches": 32,

            # PPO settings
            "ppo_steps": 5,
            "eps": 0.1,  # PPO clip parameter
            "target_kl": 0.01,  # KL divergence limit
            "value_loss_coeff": 0.1,

            "entropy_coeff": 0.01,

            "max_grad_norm": 0.5,

            # GPU
            "use_gpu": False,
        }
        self.config = with_default_config(config, default_config)

        self.optimizers = {
            agent_id: get_optimizer(self.config["optimizer"])(agent.model.parameters(),
                                                              **self.config["optimizer_kwargs"])
            for agent_id, agent in self.agents.items()
        }

        self.eps: float = self.config["eps"]

    def train_on_data(self, data_batch: DataBatch,
                      step: int = 0,
                      writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
        """
        Performs a single update step with PPO on the given batch of data.

        Args:
            data_batch: DataBatch, dictionary
            step:
            writer:

        Returns:

        """
        metrics = {}
        timer = Timer()

        entropy_coeff = self.config["entropy_coeff"]

        data_batch: DataBatchT = transpose_batch(data_batch)

        for agent_id in self.agents:
            agent = self.agents[agent_id]
            agent_batch = data_batch[agent_id]

            optimizer = self.optimizers[agent_id]

            ####################################### Unpack and prepare the data #######################################

            if self.config["use_gpu"]:
                agent_batch = batch_to_gpu(agent_batch)
                agent.cuda()

            # Initialize metrics
            kl_divergence = 0.
            ppo_step = -1
            value_loss = torch.tensor(0)
            policy_loss = torch.tensor(0)
            loss = torch.tensor(0)

            batcher = Batcher(agent_batch['dones'].size(0) // self.config["minibatches"],
                              [np.arange(agent_batch['dones'].size(0))])

            # Start a timer
            timer.checkpoint()

            for ppo_step in range(self.config["ppo_steps"]):
                batcher.shuffle()

                # for indices, agent_minibatch in minibatches(agent_batch, self.config["batch_size"], shuffle=True):
                while not batcher.end():
                    batch_indices = batcher.next_batch()[0]
                    batch_indices = torch.tensor(batch_indices).long()

                    agent_minibatch = index_data(agent_batch, batch_indices)
                    # Evaluate again after the PPO step, for new values and gradients
                    logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(agent_minibatch)

                    advantages_batch = agent_minibatch['advantages']
                    old_logprobs_minibatch = agent_minibatch['logprobs']  # logprobs of taken actions
                    discounted_batch = agent_minibatch['rewards_to_go']

                    ######################################### Compute the loss #############################################
                    # Surrogate loss
                    prob_ratio = torch.exp(logprob_batch - old_logprobs_minibatch)
                    surr1 = prob_ratio * advantages_batch
                    surr2 = prob_ratio.clamp(1. - self.eps, 1 + self.eps) * advantages_batch
                    # surr2 = torch.where(advantages_batch > 0,
                    #                     (1. + self.eps) * advantages_batch,
                    #                     (1. - self.eps) * advantages_batch)

                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = 0.5 * (value_batch - discounted_batch) ** 2
                    # import pdb; pdb.set_trace()
                    loss = (torch.mean(policy_loss)
                            + (self.config["value_loss_coeff"] * torch.mean(value_loss))
                            - (entropy_coeff * torch.mean(entropy_batch)))

                    ############################################# Update step ##############################################
                    optimizer.zero_grad()
                    loss.backward()
                    if self.config["max_grad_norm"] is not None:
                        nn.utils.clip_grad_norm_(agent.model.parameters(), self.config["max_grad_norm"])
                    optimizer.step()

                # logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(agent_batch)
                #
                # kl_divergence = torch.mean(old_logprobs_batch - logprob_batch).item()
                # if abs(kl_divergence) > self.config["target_kl"]:
                #     break

            agent.cpu()

            ep_lens = get_episode_lens(agent_batch['dones'].cpu())
            ep_rewards = torch.tensor([torch.sum(rewards) for rewards in torch.split(agent_batch['rewards'], ep_lens)])

            metrics[f"{agent_id}/episode_reward_mean"] = torch.mean(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_median"] = torch.median(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_min"] = torch.min(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_max"] = torch.max(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_std"] = torch.std(ep_rewards).item()

            # Training-related metrics
            metrics[f"{agent_id}/time_update"] = timer.checkpoint()
            metrics[f"{agent_id}/kl_divergence"] = kl_divergence
            metrics[f"{agent_id}/ppo_steps_made"] = ppo_step + 1
            metrics[f"{agent_id}/policy_loss"] = torch.mean(policy_loss).cpu().item()
            metrics[f"{agent_id}/value_loss"] = torch.mean(value_loss).cpu().item()
            metrics[f"{agent_id}/total_loss"] = loss.detach().cpu().item()
            # metrics[f"{agent_id}/rewards"] = agent_batch['rewards'].cpu().sum().item()
            metrics[f"{agent_id}/mean_std"] = agent.model.std.mean().item()

            # Other metrics
            # metrics[f"agent/mean_entropy"] = torch.mean(entropy_batch).item()

            # Write the metrics to tensorboard
            write_dict(metrics, step, writer)

        return metrics
