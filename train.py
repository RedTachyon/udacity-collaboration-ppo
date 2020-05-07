from unityagents import UnityEnvironment
import numpy as np
import torch
from models import MLPModel
from agent import Agent
from collector import Collector, Memory
from utils import unpack, discount_rewards_to_go, get_episode_rewards
from policy_optimization import PPOptimizer
from trainers import PPOTrainer

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
torch.manual_seed(0)
np.random.seed(0)


env = UnityEnvironment(file_name='Reacher.app')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

model_config = {
            "input_size": 33,
            "num_actions": 4,
            "activation": "relu",

            "hidden_sizes": (512, 512),

            "initializer": None,#"kaiming_uniform",
        }

model = MLPModel(model_config)

agent = Agent(model)

trainer_config = {
    "steps": 2049,

    # Tensorboard settings
    "tensorboard_name": "test",  # str, set explicitly

    # PPO
    "ppo_config": {
        # GD settings
        "optimizer": "adam",
        "optimizer_kwargs": {
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-5,
            "weight_decay": 0,
            "amsgrad": False
        },
        "gamma": .99,  # Discount factor
        "tau": .95,

        "batch_size": 64,

        # PPO settings
        "ppo_steps": 10,  # How many max. gradient updates in one iterations
        "eps": 0.2,  # PPO clip parameter
        "target_kl": 0.1,  # KL divergence limit
        "value_loss_coeff": 1.,
        "entropy_coeff": 0.0,
        "max_grad_norm": 5,

        # Backpropagation settings
        "use_gpu": False,
    }
}

trainer = PPOTrainer(agent, env, trainer_config)

trainer.train(num_iterations=10000, save_path=trainer.path)