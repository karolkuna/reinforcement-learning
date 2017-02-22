import numpy as np
import random

class SimpleBuffer(object):
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.buffer = np.zeros([size, dim])
        self.filled_size = 0
        self.latest_added_id = -1

    def add(self, value):
        self.latest_added_id = (self.latest_added_id + 1) % self.size
        self.buffer[self.latest_added_id] = value

        if self.filled_size < self.size:
            self.filled_size += 1

    def get_batch(self, batch_size, ids=None):
        if ids is None:
            ids = np.random.random_integers(0, self.filled_size - 1, batch_size)

        return self.buffer[ids]

    def get_latest_value(self):
        if self.latest_added_id == -1:
            return None
        else:
            return self.buffer[self.latest_added_id]


class ReplayBuffer(object):

    def __init__(self, buffer_size, state_dim, action_dim):
        self.size = buffer_size
        self.filled_size = 0
        
        self.state_buffer = SimpleBuffer(buffer_size, state_dim)
        self.action_buffer = SimpleBuffer(buffer_size, action_dim)
        self.reward_buffer = SimpleBuffer(buffer_size, 1)
        self.next_state_buffer = SimpleBuffer(buffer_size, state_dim)
        self.done_buffer = SimpleBuffer(buffer_size, 1)

    def get_random_ids(self, batch_size):
        return np.random.random_integers(0, self.filled_size - 1, batch_size)

    def get_batch(self, batch_size, ids=None):
        if ids is None:
            ids = self.get_random_ids(batch_size)

        return self.state_buffer.get_batch(batch_size, ids), \
               self.action_buffer.get_batch(batch_size, ids), \
               self.reward_buffer.get_batch(batch_size, ids), \
               self.next_state_buffer.get_batch(batch_size, ids), \
               self.done_buffer.get_batch(batch_size, ids), \
               ids

    def add(self, state, action, reward, next_state, done):
        self.filled_size = min(self.filled_size + 1, self.size)

        self.state_buffer.add(state)
        self.action_buffer.add(action)
        self.reward_buffer.add(reward)
        self.next_state_buffer.add(next_state)
        self.done_buffer.add(done)

class Priority(object):
    def __init__(self, priority, prev_priority_sum, buffer_id):
        self.priority = priority
        self.prev_priority_sum = prev_priority_sum
        self.priority_sum = priority + prev_priority_sum
        self.buffer_id = buffer_id

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, state_dim, action_dim):
        ReplayBuffer.__init__(self, buffer_size, state_dim, action_dim)
        self.priorities = list()

    def add(self, state, action, reward, next_state, done, priority=1):
        ReplayBuffer.add(self, state, action, reward, next_state, done)
        
        prev_priority_sum = 0
        if len(self.priorities) > 0:
            prev_priority_sum = self.priorities[-1].priority_sum

        self.priorities.append(Priority(priority, prev_priority_sum, self.state_buffer.latest_added_id))

        if len(self.priorities) > self.size:
            self.priorities.pop(0)

    def find_priority(self, value):
        from_id = 0
        to_id = len(self.priorities) - 1

        while from_id < to_id:
            middle_id = (from_id + to_id) // 2
            middle_value = self.priorities[middle_id].priority_sum

            if value > middle_value:
                from_id = middle_id + 1
            elif value < middle_value:
                to_id = middle_id
            else:
                return self.priorities[middle_id]

        return self.priorities[to_id]


    def get_batch(self, batch_size, proportional_to_priorities=True):
        if not proportional_to_priorities:
            return ReplayBuffer.get_batch(self, batch_size)

        batch_ids = []
        for i in range(batch_size):
            rnd_nb = random.uniform(self.priorities[0].prev_priority_sum, self.priorities[-1].priority_sum)
            batch_ids.append(self.find_priority(rnd_nb).buffer_id)

        return ReplayBuffer.get_batch(self, batch_size, batch_ids)

    def change_priority(self, buffer_id, new_priority):
        # buffers use rotational memory, but priority list doesn't. buffer_id has to be transformed
        p_id = buffer_id
        buffer_latest_id = self.state_buffer.latest_added_id

        if self.filled_size == self.size:
            if buffer_id <= self.state_buffer.latest_added_id:
                p_id = buffer_id + (self.size - buffer_latest_id - 1)
            else:
                p_id = buffer_id - buffer_latest_id - 1

        self.priorities[p_id].priority = new_priority

    def recalculate_sums(self):
        self.priorities[0].prev_priority_sum = 0
        self.priorities[0].priority_sum = self.priorities[0].priority

        for i in range(1, self.size):
            self.priorities[i].prev_priority_sum = self.priorities[i - 1].priority_sum
            self.priorities[i].priority_sum = self.priorities[i].prev_priority_sum + self.priorities[i].priority

