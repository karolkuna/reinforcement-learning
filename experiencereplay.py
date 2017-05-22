import math
import numpy as np
import replaybuffer as rb


class ExperienceReplay:
    def __init__(self, agent, environment, max_size, episodic=True):
        self.agent = agent
        self.environment = environment
        self.max_size = max_size
        self.replay_buffer = rb.ReplayBuffer(max_size, agent.state_dim, agent.action_dim)
        self.episodic = episodic

    def add_experience(self, state, action, reward, next_state, done):
        state_vector = np.array(state, ndmin=1)
        action_vector = np.array(action, ndmin=1)
        reward_vector = np.array(reward, ndmin=1)
        next_state_vector = np.array(next_state, ndmin=1)

        self.replay_buffer.add(state_vector, action_vector, reward_vector, next_state_vector, done)

    def train_agent(self, batch_size, training_steps=1):
        for i in range(training_steps):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch(batch_size)
            self.agent.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, agent, environment, max_size, episodic=True):
        self.environment = environment
        self.agent = agent
        self.max_size = max_size
        self.replay_buffer = rb.PrioritizedReplayBuffer(max_size, agent.state_dim, agent.action_dim, parallel=True)
        self.last_td_error = 0
        self.episodic = episodic

    def add_experience(self, state, action, reward, next_state, done, clip_priority_multiple=25):
        state_vector = np.array(state, ndmin=1)
        action_vector = np.array(action, ndmin=1)
        reward_vector = np.array(reward, ndmin=1)
        next_state_vector = np.array(next_state, ndmin=1)

        average_td_error = self.replay_buffer.get_average_priority()
        self.last_td_error = np.clip(self.agent.get_td_error(state_vector, action_vector, reward_vector, next_state_vector, done), 0, clip_priority_multiple * average_td_error)
        priority = math.fabs(self.last_td_error)
        self.replay_buffer.add(state_vector, action_vector, reward_vector, next_state_vector, done, priority)

    def update_oldest_priorities(self, count):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch_by_ids(range(count))
        td_errors = self.agent.get_td_error_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        priorities = [math.fabs(td_error) for td_error in td_errors]
        self.replay_buffer.update_oldest_priorities(priorities)

    def get_last_td_error(self):
        return self.last_td_error


class ModelBasedPrioritizedExperienceReplay():
    def __init__(self, agent, environment, max_size, episodic):
        self.agent = agent
        self.environment = environment
        self.max_size = max_size
        self.episodic = episodic
        self.replay_buffer = rb.PrioritizedReplayBuffer(max_size, agent.state_dim, agent.action_dim, parallel=True)
        self.model_replay_buffer = rb.PrioritizedReplayBuffer(max_size, agent.state_dim, agent.action_dim, parallel=True)
        self.reward_replay_buffer = rb.PrioritizedReplayBuffer(max_size, agent.state_dim, agent.action_dim, parallel=True)
        self.last_td_error = 0
        self.last_model_error = 0
        self.last_reward_error = 0

    def add_experience(self, state, action, reward, next_state, done, clip_priority_multiple=25):
        # make sure all inputs are vectors
        state_vector = np.array(state, ndmin=1)
        action_vector = np.array(action, ndmin=1)
        reward_vector = np.array(reward, ndmin=1)
        next_state_vector = np.array(next_state, ndmin=1)

        average_td_error = self.replay_buffer.get_average_priority()
        average_model_error = self.model_replay_buffer.get_average_priority()
        average_reward_error = self.reward_replay_buffer.get_average_priority()

        self.last_td_error = np.clip(self.agent.get_td_error(state_vector), 0, clip_priority_multiple * average_td_error)
        self.last_model_error = np.clip(self.agent.get_model_error(state_vector, action_vector, next_state_vector), 0, clip_priority_multiple * average_model_error)
        self.last_reward_error = np.clip(self.agent.get_reward_error(state_vector, action_vector, reward_vector), 0, clip_priority_multiple * average_reward_error)

        self.replay_buffer.add(state_vector, None, None, None, None, math.fabs(self.last_td_error))
        self.model_replay_buffer.add(state_vector, action_vector, None, next_state_vector, None, self.last_model_error)
        self.reward_replay_buffer.add(state_vector, action_vector, reward_vector, None, None, self.last_reward_error)

        if done and self.episodic:
            # add several dummy transitions from end state to itself with zero reward
            # thanks to this trick, episodic and open-ended environments can be treated in the same way
            for i in range(10):
                rnd_action = np.array(self.environment.action_space.sample(), ndmin=1)
                self.model_replay_buffer.add(next_state_vector, rnd_action, None, next_state_vector, None, 1)
                self.reward_replay_buffer.add(next_state_vector, rnd_action, np.array(0, ndmin=1), None, None, 1)


    def update_oldest_priorities(self, count):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch_by_ids(range(count))
        td_errors = self.agent.get_td_error_batch(state_batch)
        self.replay_buffer.update_oldest_priorities([math.fabs(td_error) for td_error in td_errors])

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.model_replay_buffer.get_batch_by_ids(range(count))
        model_errors = self.agent.get_model_error_batch(state_batch, action_batch, next_state_batch)
        self.model_replay_buffer.update_oldest_priorities(model_errors)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.reward_replay_buffer.get_batch_by_ids(range(count))
        reward_errors = self.agent.get_reward_error_batch(state_batch, action_batch, reward_batch)
        self.reward_replay_buffer.update_oldest_priorities(reward_errors)

    def get_last_td_error(self):
        return self.last_td_error

    def get_last_model_error(self):
        return self.last_model_error

    def get_last_reward_error(self):
        return self.last_reward_error

    def train_agent(self, batch_size, training_steps=1):
        for i in range(training_steps):
            state_batch, action_batch, _r, next_state_batch, _d = self.model_replay_buffer.get_batch(batch_size)
            self.agent.train_model(state_batch, action_batch, next_state_batch)

            state_batch, action_batch, reward_batch, _ns, _d = self.reward_replay_buffer.get_batch(batch_size)
            self.agent.train_reward(state_batch, action_batch, reward_batch)

            state_batch, _a, _r, _ns, _d = self.replay_buffer.get_batch(batch_size)
            self.agent.train_actor(state_batch)
            self.agent.train_value(state_batch)
