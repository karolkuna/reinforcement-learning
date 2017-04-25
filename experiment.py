import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import pprint
from displayframesasgif import display_frames_as_gif
from movingaverage import MovingAverage
import imageio

class Experiment:
    def __init__(self, path, session, environment, settings, render_environment=False, render_frequency=60):
        self.path = path
        self.session = session
        self.saver = tf.train.Saver()
        self.environment = environment
        self.settings = settings
        self.render_environment = render_environment
        self.render_frequency = render_frequency

        self.reward_100ma = MovingAverage(100)
        self.reward_history = []
        self.cumulative_reward_history = []
        self.episode_reward = 0
        self.episode_duration = 0
        self.td_error_history = []
        self.model_error_history = []
        self.reward_error_history = []
        self.episode_reward_history = []
        self.episode_duration_history = []
        self.episode_history = []

        self.frames = []

        if not os.path.exists(self.path):
            os.mkdir(self.path)
        f = open(self.path + "/settings.txt", "w+")
        pprint.pprint(self.settings, f)
        f.close()

    def record(self, t, state, action, reward, next_state, done, td_error, model_error=0, reward_error=0):
        self.reward_history.append(reward)
        self.reward_100ma.add_value(reward)

        cumulative_reward = reward
        if len(self.cumulative_reward_history) > 0:
            cumulative_reward += self.cumulative_reward_history[-1]
        self.cumulative_reward_history.append(cumulative_reward)

        self.td_error_history.append(td_error)
        self.model_error_history.append(model_error)
        self.reward_error_history.append(reward_error)
        self.episode_history.append(len(self.episode_reward_history))

        self.episode_reward += reward
        self.episode_duration += 1

        if done:
            print("Episode over. Total reward: {}. Solved in {} steps.".format(self.episode_reward, self.episode_duration))
            self.episode_reward_history.append(self.episode_reward)
            self.episode_duration_history.append(self.episode_duration)
            self.episode_reward = 0
            self.episode_duration = 0

        if self.render_environment and t % self.render_frequency == 0:
            self.frames.append(self.environment.render(mode = 'rgb_array'))

    def print_all_tf_variables(self):
        variables = tf.trainable_variables()
        values = self.session.run(variables)

        for variable, value in zip(variables, values):
            print(variable.name, value)

    def display_frames_as_gif(self):
        display_frames_as_gif(self.frames)
        imageio.mimsave(self.path + "/frames.gif", self.frames)

    def plot_reward(self, skip_steps=0):
        plt.title('Reward in time')
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plot = plt.plot(self.reward_history[skip_steps:])
        plt.savefig(self.path + "/reward.png", dpi=300)
        return plot

    def plot_cumulative_reward(self, skip_steps=0):
        plt.title('Cumulative reward in time')
        plt.xlabel("Time step")
        plt.ylabel("Cumulative reward")
        plot = plt.plot(self.cumulative_reward_history[skip_steps:])
        plt.savefig(self.path + "/cumulative_reward.png", dpi=300)
        return plot

    def plot_td_error(self, skip_steps=0):
        plt.title('Absolute TD error in time')
        plt.xlabel("Time step")
        plt.ylabel("TD error")
        plot = plt.plot(self.td_error_history[skip_steps:])
        plt.savefig(self.path + "/td_error.png", dpi=300)
        return plot

    def plot_model_error(self, skip_steps=0):
        plt.title('Transition model error in time')
        plt.xlabel("Time step")
        plt.ylabel("Squared error")
        plot = plt.plot(self.model_error_history[skip_steps:])
        plt.savefig(self.path + "/model_error.png", dpi=300)
        return plot

    def plot_reward_error(self, skip_steps=0):
        plt.title('Reward model error in time')
        plt.xlabel("Time step")
        plt.ylabel("Squared error")
        plot = plt.plot(self.reward_error_history[skip_steps:])
        plt.savefig(self.path + "/reward_error.png", dpi=300)
        return plot

    def plot_episode_duration(self, skip_episodes=0):
        plt.title('Duration of episodes')
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plot = plt.plot(self.episode_duration_history[skip_episodes:])
        plt.savefig(self.path + "/episode_duration.png", dpi=300)
        return plot

    def plot_episode_reward(self, skip_episodes=0):
        plt.title('Cumulative reward of episodes')
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plot = plt.plot(self.episode_reward_history[skip_episodes:])
        plt.savefig(self.path + "/episode_reward.png", dpi=300)
        return plot

    def save(self):
        df = pd.DataFrame(columns=["t", "episode", "reward", "cumulative_reward", "td_error", "model_error", "reward_error"])
        df.t = range(len(self.reward_history))
        df.episode = self.episode_history
        df.reward = self.reward_history
        df.cumulative_reward = self.cumulative_reward_history
        df.td_error = self.td_error_history
        df.model_error = self.model_error_history
        df.reward_error = self.reward_error_history
        df.to_csv(self.path + "/timesteps.csv", index=None, float_format='%.10f')

        df_episodes = pd.DataFrame(columns=["episode", "reward", "duration"])
        df_episodes.episode = range(len(self.episode_reward_history))
        df_episodes.reward = self.episode_reward_history
        df_episodes.duration = self.episode_duration_history
        df_episodes.to_csv(self.path + "/episodes.csv", index=None, float_format='%.10f')

        self.saver.save(self.session, self.path + "/model.ckpt")
