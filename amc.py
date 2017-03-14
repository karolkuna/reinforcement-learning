# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
from actorcritic import create_actor_model_critic_network

class AMC:
    def __init__(self, actor_network, model_network, reward_network, value_network, forward_steps=1, discount_factor=0.9, batch_size=128, replay_buffer_size=100000, learning_rate=0.0001, actor_target_approach_rate=0.999, value_target_approach_rate=0.999):
        if forward_steps < 1:
            raise Exception("At least one forward step has to be executed!")

        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.forward_steps = 1
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.actor_target_approach_rate = actor_target_approach_rate
        self.value_target_approach_rate = value_target_approach_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork(actor_network.name + "_target", actor_network, actor_target_approach_rate)

        self.model_network = model_network
        self.model_optimizer = SquaredLossOptimizer(model_network, tf.train.AdamOptimizer(learning_rate))

        self.reward_network = reward_network
        self.reward_optimizer = SquaredLossOptimizer(reward_network, tf.train.AdamOptimizer(learning_rate))

        self.value_network = value_network
        self.value_target_network = TargetNeuralNetwork(value_network.name + "_target", value_network, value_target_approach_rate)
        self.value_optimizer = SquaredLossOptimizer(value_network, tf.train.AdamOptimizer(learning_rate))

        self.actor_ac_network = create_actor_model_critic_network(
            "Actor_AC", actor_network, model_network, reward_network, self.value_target_network, 1 # TODO: should actor learn from multiple forward steps too?
        )
        self.value_ac_network = create_actor_model_critic_network(
            "Value_AC", self.actor_target_network, model_network, reward_network, self.value_target_network, forward_steps
        )

        self.actor_optimizer = MaxOutputOptimizer(
            self.actor_ac_network, tf.train.AdamOptimizer(learning_rate), actor_network.get_parameters()
        )

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.state_dim, self.action_dim)

        self.actor_network.session.run(tf.global_variables_initializer())

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, ids = self.replay_buffer.get_batch(self.batch_size)

        value_target_batch = self.value_ac_network.predict_batch([state_batch])

        for i in range(self.batch_size): 
            if done_batch[i]:
                value_target_batch[i] = reward_batch[i] # TODO: when forward_steps > 1, episode ends are not handled properly

        value_target_batch = np.resize(value_target_batch, [self.batch_size, 1])

        self.value_optimizer.train([state_batch], value_target_batch)
        self.model_optimizer.train([state_batch, action_batch], next_state_batch)
        self.reward_optimizer.train([state_batch, action_batch], reward_batch)
        self.actor_optimizer.train([state_batch])

        self.value_target_network.approach_source_parameters()
        self.actor_target_network.approach_source_parameters()

    def train_model(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, ids = self.replay_buffer.get_batch(self.batch_size)
        self.model_optimizer.train([state_batch, action_batch], next_state_batch)

    def reset_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.state_dim, self.action_dim)

    def action(self, state):
        action = self.actor_network.predict([state])
        return action

    def noisy_action(self, state, mean=0, stddev=1.0):
        return self.actor_network.predict([state]) + np.random.normal(mean, stddev, self.action_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
