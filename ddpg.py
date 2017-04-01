# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
from actorcritic import create_actor_critic_network

class DDPG:
    def __init__(self, actor_network, q_network, discount_factor=0.9, tf_optimizer=tf.train.AdamOptimizer(0.0001), actor_target_approach_rate=0.999, q_target_approach_rate=0.999):
        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.discount_factor = discount_factor
        self.actor_target_approach_rate = actor_target_approach_rate
        self.q_target_approach_rate = q_target_approach_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork("Actor_target", self.actor_network, self.actor_target_approach_rate)

        self.q_network = q_network
        self.q_target_network = TargetNeuralNetwork("Q_target", self.q_network, self.q_target_approach_rate)
        self.q_optimizer = SquaredLossOptimizer(q_network, tf_optimizer)

        self.actor_critic_network, _actor, _critic = create_actor_critic_network("ActorCritic", actor_network, self.q_target_network)
        self.actor_optimizer = MaxOutputOptimizer(self.actor_critic_network, tf_optimizer, self.actor_network.get_parameters())

        self.actor_network.session.run(tf.global_variables_initializer())

    def train(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.train_q(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.train_actor(state_batch)

    def train_actor(self, state_batch):
        self.actor_optimizer.train([state_batch])
        self.actor_target_network.approach_source_parameters()

    def train_q(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = len(state_batch)
        # q_target_batch = reward_batch + discount_factor * q_target_network(next_state_batch, actor_target(next_state_batch))
        next_action_batch = self.actor_target_network.predict_batch([next_state_batch])
        next_q_value_batch = self.q_target_network.predict_batch([next_state_batch, next_action_batch])
        q_target_batch = []
        for i in xrange(batch_size):
            if done_batch[i]:
                q_target_batch.append(reward_batch[i])
            else:
                q_target_batch.append(reward_batch[i] + self.discount_factor * next_q_value_batch[i])

        q_target_batch = np.resize(q_target_batch, [batch_size, 1])

        self.q_optimizer.train([state_batch, action_batch], q_target_batch)
        self.q_target_network.approach_source_parameters()

    def action(self, state):
        action = self.actor_network.predict([state])
        return action
