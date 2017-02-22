import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import NeuralNetwork, TargetNeuralNetwork, SquaredLossOptimizer, MaxOutputOptimizer, NeuralNetworkComposition

class DDPG:
    def __init__(self, actor_network, q_network, discount_factor=0.9, batch_size=128, replay_buffer_size=100000, learning_rate=0.0001, actor_target_approach_rate=0.999, q_target_approach_rate=0.999):
        self.state_dim = q_network.input_dims[0]
        self.action_dim = q_network.input_dims[1]
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.actor_target_approach_rate = actor_target_approach_rate
        self.q_target_approach_rate = q_target_approach_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork(self.actor_network, self.actor_target_approach_rate)

        self.q_network = q_network
        self.q_target_network = TargetNeuralNetwork(self.q_network, self.q_target_approach_rate)
        self.q_optimizer = SquaredLossOptimizer(q_network, tf.train.AdamOptimizer(learning_rate))

        self.actor_critic_network = NeuralNetworkComposition([actor_network, q_network], [1])
        self.actor_optimizer = MaxOutputOptimizer(self.actor_critic_network, tf.train.AdamOptimizer(learning_rate), actor_network.trainable_parameters)

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.state_dim, self.action_dim)

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, ids = self.replay_buffer.get_batch(self.batch_size)

        # q_target_batch = reward_batch + q_target_network(next_state_batch, actor_target(next_state_batch))
        next_action_batch = self.actor_target_network.predict([next_state_batch])
        next_q_value_batch = self.q_target_network.predict([next_state_batch, next_action_batch])
        q_target_batch = []
        for i in range(self.batch_size): 
            if done_batch[i]:
                q_target_batch.append(reward_batch[i])
            else :
                q_target_batch.append(reward_batch[i] + self.discount_factor * next_q_value_batch[i])
        q_target_batch = np.resize(q_target_batch, [self.batch_size, 1])

        self.q_optimizer.train([state_batch, action_batch], q_target_batch)
        self.actor_optimizer.train([state_batch, state_batch])

        self.q_target_network.approach_source_parameters()
        self.actor_target_network.approach_source_parameters()

    def action(self, state):
        action = self.actor_network.predict([state])
        return action

    def noisy_action(self, state, mean=0, stddev=1.0):
        return self.actor_network.predict([state]) + np.random.normal(mean, stddev, self.action_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
