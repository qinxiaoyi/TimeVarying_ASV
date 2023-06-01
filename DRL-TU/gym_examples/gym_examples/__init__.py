import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
     id="gym_examples/SpkembdUpdateEnv-v0",
     entry_point="gym_examples.envs:SpkembdUpdateEnv_v0",
     max_episode_steps=200,
)

register(
     id="gym_examples/SpkembdUpdateEnv_multihead",
     entry_point="gym_examples.envs:SpkembdUpdateEnv_multihead",
     max_episode_steps=200,
)

