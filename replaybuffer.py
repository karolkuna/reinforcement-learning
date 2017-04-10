import math
import numpy as np
import random
import multiprocessing
import multiprocessing.queues
from operator import itemgetter


class SimpleBuffer(object):
    def __init__(self, max_size, dim):
        self.max_size = max_size
        self.dim = dim
        self.buffer = list()

    def add(self, value):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

        self.buffer.append(value)

    def get_batch(self, batch_size, ids=None):
        if ids is None:
            ids = np.random.random_integers(0, len(self.buffer) - 1, batch_size)

        return itemgetter(*ids)(self.buffer)


class ReplayBuffer(object):

    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        
        self.state_buffer = SimpleBuffer(max_size, state_dim)
        self.action_buffer = SimpleBuffer(max_size, action_dim)
        self.reward_buffer = SimpleBuffer(max_size, 1)
        self.next_state_buffer = SimpleBuffer(max_size, state_dim)
        self.done_buffer = SimpleBuffer(max_size, 1)

    def get_random_ids(self, batch_size):
        return np.random.random_integers(0, len(self.state_buffer.buffer) - 1, batch_size)

    def get_batch(self, batch_size, ids=None):
        if ids is None:
            ids = self.get_random_ids(batch_size)

        return self.state_buffer.get_batch(batch_size, ids), \
               self.action_buffer.get_batch(batch_size, ids), \
               self.reward_buffer.get_batch(batch_size, ids), \
               self.next_state_buffer.get_batch(batch_size, ids), \
               self.done_buffer.get_batch(batch_size, ids)

    def add(self, state, action, reward, next_state, done):
        self.state_buffer.add(state)
        self.action_buffer.add(action)
        self.reward_buffer.add(reward)
        self.next_state_buffer.add(next_state)
        self.done_buffer.add(done)


class Priority(object):
    def __init__(self, priority, prev_priority_sum):
        self.priority = priority
        self.prev_priority_sum = prev_priority_sum
        self.priority_sum = priority + prev_priority_sum


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, state_dim, action_dim, parallel=True):
        ReplayBuffer.__init__(self, max_size, state_dim, action_dim)
        self.max_size = max_size
        self.priorities = list()
        self.parallel = parallel
        if parallel:
            self.worker_queues = None
            self.priorities_journal = None
            self.pool = None
            self.init_worker_pool()

    def init_worker_pool(self):
        self.worker_queues = []  # per-process queues used to synchronize changes of priorities list
        self.priorities_journal = list()  # history of changes made to priorities list that require sync
        worker_queue_ids = multiprocessing.queues.SimpleQueue()  # queue used to assign each process its own queue

        for i in range(multiprocessing.cpu_count()):
            queue = multiprocessing.queues.SimpleQueue()
            self.worker_queues.append(queue)
            worker_queue_ids.put(i)

        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),
                                         initializer=init_worker_process,
                                         initargs=(self.priorities, self.max_size, worker_queue_ids, self.worker_queues))

    def add_priority(self, priority):
        if len(self.priorities) == 0:
            prev_priority_sum = 0
        else:
            prev_priority_sum = self.priorities[-1].priority_sum

        if len(self.priorities) >= self.max_size:
            self.priorities.pop(0)

        new_priority = Priority(priority, prev_priority_sum)
        self.priorities.append(new_priority)
        
        if self.parallel:
            self.priorities_journal.append(new_priority)

    def add(self, state, action, reward, next_state, done, priority=1):
        ReplayBuffer.add(self, state, action, reward, next_state, done)
        self.add_priority(priority)
        
    def find_id_by_sampled_value(self, value):
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
                return middle_id

        return to_id


    def get_batch(self, batch_size, proportional_to_priorities=True, decay_old_samples_priority=False):
        """
        :param batch_size: number of samples to return 
        :param proportional_to_priorities: if true, samples are selected randomly according to stored priority. Probability
                                            of a sample being returned is its priority over sum of all priorities in buffer
                                           Otherwise, samples are selected with uniform probability
        :param decay_old_samples_priority: if true, older samples are returned less frequently. Priority of a sample is
                                            scaled linearly by its position in the buffer  
        :return: batch
        """
        if not proportional_to_priorities:
            return ReplayBuffer.get_batch(self, batch_size)

        if self.parallel:
            return self.parallel_get_batch(batch_size, decay_old_samples_priority)

        batch_ids = []
        min_pri = self.priorities[0].prev_priority_sum
        max_pri = self.priorities[-1].priority_sum

        for i in range(batch_size):
            rnd_nb = random.uniform(0.0, 1.0)
            if decay_old_samples_priority:
                rnd_nb = math.sqrt(rnd_nb)  # shifts distribution towards newer samples
            rnd_pri = min_pri + (max_pri - min_pri) * rnd_nb
            batch_ids.append(self.find_id_by_sampled_value(rnd_pri))

        return ReplayBuffer.get_batch(self, batch_size, batch_ids)

    def parallel_get_batch(self, batch_size, decay_old_samples_priority=False):
        if len(self.priorities_journal) > 1000:  # when journal is too long, it may be faster to re-sync priorities list
            self.pool.terminate()
            self.init_worker_pool()
            self.priorities_journal = []
        else:
            if len(self.priorities_journal) > 0: # if there are any unsynced changes
                for queue in self.worker_queues:
                    queue.put(self.priorities_journal)
                self.priorities_journal = []

        batch_ids = self.pool.map(get_random_buffer_id, [decay_old_samples_priority] * batch_size)

        return ReplayBuffer.get_batch(self, batch_size, batch_ids)

    def get_batch_by_ids(self, buffer_ids):
        return ReplayBuffer.get_batch(self, len(buffer_ids), buffer_ids)

    def change_priority(self, buffer_id, new_priority):
        # warning: recalculate_sums must be called before using the buffer
        self.priorities[buffer_id].priority = new_priority

    def recalculate_sums(self):
        self.priorities[0].prev_priority_sum = 0
        self.priorities[0].priority_sum = self.priorities[0].priority

        for i in xrange(1, self.max_size):
            self.priorities[i].prev_priority_sum = self.priorities[i - 1].priority_sum
            self.priorities[i].priority_sum = self.priorities[i].prev_priority_sum + self.priorities[i].priority

        if self.parallel:
            self.pool.terminate()
            self.init_worker_pool()

    def update_oldest_priorities(self, new_priorities):
        # only priorities of the oldest samples can be updated quickly
        # without recalculating priority sums of entire buffer
        for new_priority in new_priorities:
            self.state_buffer.buffer.append(self.state_buffer.buffer.pop(0))
            self.action_buffer.buffer.append(self.action_buffer.buffer.pop(0))
            self.reward_buffer.buffer.append(self.reward_buffer.buffer.pop(0))
            self.next_state_buffer.buffer.append(self.next_state_buffer.buffer.pop(0))
            self.done_buffer.buffer.append(self.done_buffer.buffer.pop(0))
            
            if self.parallel:
                self.priorities_journal.append(None)

            self.priorities.pop(0)
            self.priorities.add(new_priority)


# global variables for worker processes
g_queue = None
g_priorities = None


def init_worker_process(priorities, max_size, queue_ids, queues):
    global g_queue
    global g_priorities
    global g_max_size

    g_priorities = priorities
    g_max_size = max_size
    g_queue = queues[queue_ids.get()]  # assigns a queue to each process


def get_random_buffer_id(decay_old_samples_priority):
    global g_priorities
    global g_max_size

    # sync changes made to priorities list to this g_priorities
    while not g_queue.empty():
        # g_queue contains lists of changes (either adding or removal of a priority object)
        priorities_journal = g_queue.get()
        for priority in priorities_journal:
            if len(g_priorities) >= g_max_size:
                g_priorities.pop(0)

            if priority is None:
                g_priorities.pop(0)
            else:
                g_priorities.append(priority)

    min_pri = g_priorities[0].prev_priority_sum
    max_pri = g_priorities[-1].priority_sum

    rnd_nb = random.uniform(0.0, 1.0)
    if decay_old_samples_priority:
        rnd_nb = math.sqrt(rnd_nb)  # shifts distribution towards newer samples
    rnd_pri = min_pri + (max_pri - min_pri) * rnd_nb
    return find_id_by_sampled_value(rnd_pri)


def find_id_by_sampled_value(value):
    global g_priorities

    from_id = 0
    to_id = len(g_priorities) - 1

    while from_id < to_id:
        middle_id = (from_id + to_id) // 2
        middle_value = g_priorities[middle_id].priority_sum

        if value > middle_value:
            from_id = middle_id + 1
        elif value < middle_value:
            to_id = middle_id
        else:
            return middle_id

    return to_id
