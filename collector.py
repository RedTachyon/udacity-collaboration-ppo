from collections import OrderedDict, defaultdict
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Any

import gym
import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from agent import Agent
from utils import with_default_config, unpack, DataBatch, discount_rewards_to_go, discount_td_rewards, pack

from unityagents import UnityEnvironment
from unityagents.brain import BrainInfo

T = TypeVar('T')


def append_dict(var: Dict[str, T], data_dict: Dict[str, List[T]]):
    """
    Works like append, but operates on dictionaries of lists and dictionaries of values (as opposed to lists and values)

    Args:
        var: values to be appended
        data_dict: lists to be appended to
    """
    for key, value in var.items():
        data_dict[key].append(value)


class Memory:
    """
    Holds the rollout data in a dictionary
    """

    def __init__(self, agents: List[str]):
        """
        Creates the memory container. 
        """

        self.agents = agents

        # Dictionaries to hold all the relevant data, with appropriate type annotations
        _observations: Dict[str, List[np.ndarray]] = {agent: [] for agent in self.agents}
        _actions: Dict[str, List[int]] = {agent: [] for agent in self.agents}
        _rewards: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _logprobs: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _entropies: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _dones: Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        _values: Dict[str, List[float]] = {agent: [] for agent in self.agents}

        self.data = {
            "observations": _observations,
            "actions": _actions,
            "rewards": _rewards,
            "logprobs": _logprobs,
            "entropies": _entropies,
            "values": _values,
            "dones": _dones,
        }

    def store(self,
              obs: Dict[str, np.ndarray],
              action: Dict[str, np.ndarray],
              reward: Dict[str, float],
              logprob: Dict[str, float],
              entropy: Dict[str, float],
              value: Dict[str, float],
              done: Dict[str, bool]):

        update = (obs, action, reward, logprob, entropy, value, done)
        for key, var in zip(self.data, update):
            append_dict(var, self.data[key])

    def reset(self):
        for key in self.data:
            self.data[key] = {agent: [] for agent in self.agents}

    def apply_to_agent(self, func: Callable) -> Dict[str, Any]:
        return {
            agent: func(agent) for agent in self.agents
        }

    def get_torch_data(self, gamma: float = 0.99, tau: float = 0.95) -> DataBatch:
        """
        Gather all the recorded data into torch tensors (still keeping the dictionary structure)
        """
        observations = self.apply_to_agent(lambda agent: torch.tensor(np.stack(self.data["observations"][agent])))
        actions = self.apply_to_agent(lambda agent: torch.tensor(self.data["actions"][agent]))
        rewards = self.apply_to_agent(lambda agent: torch.tensor(self.data["rewards"][agent]))
        logprobs = self.apply_to_agent(lambda agent: torch.tensor(self.data["logprobs"][agent]))
        dones = self.apply_to_agent(lambda agent: torch.tensor(self.data["dones"][agent]))

        entropies = self.apply_to_agent(lambda agent: torch.tensor(self.data["entropies"][agent]))
        values = self.apply_to_agent(lambda agent: torch.tensor(self.data["values"][agent]))

        # rewards_to_go = discount_rewards_to_go(rewards, dones, gamma)
        #
        # advantages = (rewards_to_go - values)
        # advantages = (advantages - advantages.mean())
        # advantages = advantages / (torch.sqrt(torch.mean(advantages ** 2)) + 1e-8)

        rewards_to_go, advantages = discount_td_rewards(rewards["Agent0"],
                                                        values["Agent0"],
                                                        dones["Agent0"],
                                                        gamma,
                                                        tau)

        advantages = (advantages - advantages.mean()) / advantages.std()

        torch_data = {
            "observations": self.apply_to_agent(lambda agent: observations[agent][:-1]),
            "actions": self.apply_to_agent(lambda agent: actions[agent][:-1]),
            "rewards": self.apply_to_agent(lambda agent: rewards[agent][:-1]),
            "logprobs": self.apply_to_agent(lambda agent: logprobs[agent][:-1]),
            "entropies": self.apply_to_agent(lambda agent: entropies[agent][:-1]),
            "dones": self.apply_to_agent(lambda agent: dones[agent][:-1]),
            "values": self.apply_to_agent(lambda agent: values[agent][:-1]),
            "rewards_to_go": {agent: rewards_to_go for agent in self.agents},
            "advantages": {agent: advantages for agent in self.agents}
        }

        return torch_data

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


class Collector:
    """
    Class to perform data collection from two agents.
    """

    def __init__(self, agents: Dict[str, Agent], env: UnityEnvironment, brain_name: str = "TennisBrain"):
        self.env = env
        self.agents = agents
        self.agent_ids = list(agents.keys())
        self.memory = Memory(list(agents.keys()))
        self.brain_name = brain_name

    def collect_data(self,
                     num_steps: int = 1000,
                     deterministic: Optional[Dict[str, bool]] = None,
                     disable_tqdm: bool = True,
                     train_mode: bool = True,
                     gamma: float = 0.99,
                     tau: float = 0.95) -> DataBatch:
        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            deterministic: whether each agent should use the greedy policy; False by default
            disable_tqdm: whether a live progress bar should be (not) displayed
            train_mode:

        Returns: dictionary with the gathered data
        """

        if deterministic is None:
            deterministic = defaultdict(lambda: False)

        self.reset()

        obs, _, _, _ = unpack(self.env.reset(train_mode=train_mode)[self.brain_name], self.agent_ids)

        for step in trange(num_steps + 1, disable=disable_tqdm):
            # Compute the action for each agent
            action_info = {  # action, logprob, entropy
                agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
                                                                      deterministic[agent_id])
                for agent_id in self.agent_ids
            }

            action = {agent_id: action_info[agent_id][0] for agent_id in self.agent_ids}
            logprob = {agent_id: action_info[agent_id][1] for agent_id in self.agent_ids}
            entropy = {agent_id: action_info[agent_id][2] for agent_id in self.agent_ids}
            value = {agent_id: action_info[agent_id][3] for agent_id in self.agent_ids}

            # Actual step in the environment
            next_obs, reward, done, info = unpack(self.env.step(pack(action))[self.brain_name], self.agent_ids)

            # Saving to memory
            self.memory.store(obs, action, reward, logprob, entropy, value, done)
            obs = next_obs

        return self.memory.get_torch_data(gamma=gamma, tau=tau)

    def collect_episodes(self,
                         num_episodes: int = 100,
                         deterministic: Optional[Dict[str, bool]] = None,
                         disable_tqdm: bool = True,
                         train_mode: bool = True,
                         gamma: float = 0.99,
                         tau: float = 0.95) -> DataBatch:
        if deterministic is None:
            deterministic = defaultdict(lambda: False)

        self.reset()

        for episode in trange(num_episodes, disable=disable_tqdm):
            obs, _, _, _ = unpack(self.env.reset(train_mode=train_mode)[self.brain_name], self.agent_ids)
            done_ = False
            while not done_:
            # Compute the action for each agent
                action_info = {  # action, logprob, entropy
                    agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
                                                                          deterministic[agent_id])
                    for agent_id in self.agent_ids
                }

                action = {agent_id: action_info[agent_id][0] for agent_id in self.agent_ids}
                logprob = {agent_id: action_info[agent_id][1] for agent_id in self.agent_ids}
                entropy = {agent_id: action_info[agent_id][2] for agent_id in self.agent_ids}
                value = {agent_id: action_info[agent_id][3] for agent_id in self.agent_ids}

                # Actual step in the environment
                next_obs, reward, done, info = unpack(self.env.step(pack(action))[self.brain_name], self.agent_ids)

                # Saving to memory
                self.memory.store(obs, action, reward, logprob, entropy, value, done)
                obs = next_obs
                done_ = done['Agent0']

        return self.memory.get_torch_data(gamma=gamma, tau=tau)

    def reset(self):
        self.memory.reset()

    def change_agent(self, agents_to_replace: Dict[str, Agent]):
        """Replace the agents in the collector"""
        for agent_id in agents_to_replace:
            if agent_id in self.agents:
                self.agents[agent_id] = agents_to_replace[agent_id]

    def update_agent_state_dict(self, agents_to_update: Dict[str, Dict]):
        """Update the state dict of the agents in the collector"""
        for agent_id in agents_to_update:
            if agent_id in self.agents:
                self.agents[agent_id].model.load_state_dict(agents_to_update[agent_id])


if __name__ == '__main__':
    pass

    # env = foraging_env_creator({})
    #
    # agent_ids = ["Agent0", "Agent1"]
    #
    # agents: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents, env, {})
    #
    # data_steps = runner.collect_data(num_steps=1000, disable_tqdm=False)
    # data_episodes = runner.collect_data(num_episodes=2, disable_tqdm=False)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')
