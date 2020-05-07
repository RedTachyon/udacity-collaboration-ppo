import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agent import Agent
from unityagents import UnityEnvironment
from utils import Timer, with_default_config, write_dict
from collector import Collector
from policy_optimization import PPOptimizer


class Trainer:
    def __init__(self,
                 agents: Dict[str, Agent],
                 env: UnityEnvironment,
                 config: Dict[str, Any]):
        self.agents = agents
        self.env = env
        self.config = config

    def train(self, num_iterations: int,
              disable_tqdm: bool = False,
              save_path: Optional[str] = None,
              **collect_kwargs):
        raise NotImplementedError


class PPOTrainer(Trainer):
    """This performs training in a sampling paradigm, where each agent is stored, and during data collection,
    some part of the dataset is collected with randomly sampled old agents"""
    def __init__(self, agents: Dict[str, Agent], env: UnityEnvironment, config: Dict[str, Any]):
        super().__init__(agents, env, config)

        default_config = {
            "steps": 2000,

            # Tensorboard settings
            "tensorboard_name": None,  # str, set explicitly

            "gamma": .99,  # Discount factor
            "tau": .95,

            # PPO
            "ppo_config": {
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
        }

        self.config = with_default_config(config, default_config)

        self.collector = Collector(agents=self.agents, env=self.env)
        self.ppo = PPOptimizer(agents=agents, config=self.config["ppo_config"])

        # Setup tensorboard
        self.writer: SummaryWriter
        if self.config["tensorboard_name"]:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path = Path.home() / "drlnd_logs" / f"{self.config['tensorboard_name']}_{dt_string}"
            self.writer = SummaryWriter(str(self.path))

            self.agent_paths = [
                self.path / agent_id for agent_id in self.agents
            ]

            for agent_path in self.agent_paths:
                os.mkdir(str(agent_path))

            # Log the configs
            with open(str(self.path / "trainer_config.json"), "w") as f:
                json.dump(self.config, f)

            with open(str(self.path / f"agent0_config.json"), "w") as f:
                json.dump(self.agents["Agent0"].model.config, f)

            with open(str(self.path / f"agent1_config.json"), "w") as f:
                json.dump(self.agents["Agent1"].model.config, f)

            self.path = str(self.path)
        else:
            self.writer = None

    def train(self, num_iterations: int,
              save_path: Optional[str] = None,
              disable_tqdm: bool = False,
              **collect_kwargs):

        print(f"Begin training, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        # Store the first agent
        # saved_agents = [copy.deepcopy(self.agent.model.state_dict())]

        if save_path:
            for path, (agent_id, agent) in zip(self.agent_paths, self.agents.items()):
                torch.save(agent.model, os.path.join(str(path), "base_agent.pt"))

        rewards = []

        for step in trange(num_iterations, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            # data_batch = self.collector.collect_data(num_episodes=self.config["episodes"])
            data_batch = self.collector.collect_data(num_steps=self.config["steps"],
                                                     gamma=self.config["gamma"],
                                                     tau=self.config["tau"])
            data_time = timer.checkpoint()
            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(data_batch, step, writer=self.writer)

            # eval_batch = self.collector.collect_data(num_steps=1001)
            # reward = eval_batch['rewards'].sum().item()
            # rewards.append(reward)

            end_time = step_timer.checkpoint()

            # Save the agent to disk
            if save_path:
                for path, (agent_id, agent) in zip(self.agent_paths, self.agents.items()):
                    torch.save(agent.model.state_dict(), os.path.join(str(path), f"weights_{step + 1}"))

            # Write training time metrics to tensorboard
            time_metrics = {
                "agent/time_data": data_time,
                "agent/time_total": end_time,
                # "agent/eval_reward": reward
            }

            write_dict(time_metrics, step, self.writer)

        # return rewards


if __name__ == '__main__':
    pass
