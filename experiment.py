import tensorflow as tf
from displayframesasgif import display_frames_as_gif
from movingaverage import MovingAverage

class Experiment:
    def __init__(self, environment, settings, render_environment=False, render_frequency=60):
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

        self.frames = []


    def record(self, t, state, action, reward, next_state, done, td_error):
        self.reward_history.append(reward)
        self.reward_100ma.add_value(reward)


        cumulative_reward = reward
        if len(self.cumulative_reward_history) > 0:
            cumulative_reward += self.cumulative_reward_history[-1]
        self.cumulative_reward_history.append(cumulative_reward)

        self.td_error_history.append(td_error)

        self.episode_reward += reward
        self.episode_duration += 1

        if done:
            print("Episode over. Total reward: {}. Solved in {} steps.".format(self.episode_reward, self.episode_duration))
            self.episode_reward = 0
            self.episode_duration = 0

        if self.render_environment and t % self.render_frequency == 0:
            self.frames.append(self.environment.render(mode = 'rgb_array'))




    def print_all_tf_variables(self, session):
        variables = tf.trainable_variables()
        values = session.run(variables)

        for variable, value in zip(variables, values):
            print(variable.name, value)

    def display_frames_as_gif(self):
        display_frames_as_gif(self.frames)
