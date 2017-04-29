import gym
import numpy as np
import random
import ounoise

class IExplorationStrategy:
    def __init__(self, agent, environment, seed=None, **kwargs):
        self.agent = agent
        self.environment = environment
        self.seed = seed

    def action(self, state, exploration_parameter):
        pass


class EpsilonGreedyStrategy(IExplorationStrategy):
    def __init__(self, agent, environment, seed=None):
        IExplorationStrategy.__init__(self, agent, environment, seed)
        random.seed(seed)

    def action(self, state, exploration_probability):
        if random.uniform(0, 1) < exploration_probability:
            action = self.environment.action_space.sample()
        else:
            action = self.agent.action(state)

        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        return action


class OUStrategy(IExplorationStrategy):
    def __init__(self, agent, environment, seed=None, mu=0, theta=0.15, sigma=0.2):
        IExplorationStrategy.__init__(self, agent, environment, seed)
        self.ou_noise = ounoise.OUNoise(
            environment.action_space.shape[0], mu=mu, theta=theta, sigma=sigma, seed=seed,
            bounds=self.environment.action_space)

    def action(self, state, noise_scale):
        action = self.agent.action(state)
        noise = noise_scale * self.ou_noise.noise()

        action = action + noise
        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        return action
