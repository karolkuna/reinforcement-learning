import math
import replaybuffer as rb

class ExperienceReplay:
    def __init__(self, agent, max_size):
        self.agent = agent
        self.max_size = max_size
        self.replay_buffer = rb.ReplayBuffer(max_size, agent.state_dim, agent.action_dim)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_agent(self, batch_size, training_steps=1):
        for i in range(training_steps):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch(batch_size)
            self.agent.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, agent, max_size):
        self.agent = agent
        self.max_size = max_size
        self.replay_buffer = rb.PrioritizedReplayBuffer(max_size, agent.state_dim, agent.action_dim, parallel=True)
        self.last_td_error = 0

    def add_experience(self, state, action, reward, next_state, done):
        self.last_td_error = self.agent.get_td_error(state, action, reward, next_state, done)
        priority = math.fabs(self.last_td_error)
        self.replay_buffer.add(state, action, reward, next_state, done, priority)

    def update_oldest_priorities(self, count):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch_by_ids(range(count))
        td_errors = self.agent.get_td_error_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        priorities = [math.fabs(td_error) for td_error in td_errors]
        self.replay_buffer.update_oldest_priorities(priorities)

    def get_last_td_error(self):
        return self.last_td_error
