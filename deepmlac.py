# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import NeuralNetwork, TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
import actorcritic as ac

class DeepMLAC:
    def __init__(self, actor_network, model_network, reward_network, value_network, forward_steps=1, discount_factor=0.9, trace_decay=0.9, actor_tf_optimizer=tf.train.AdamOptimizer(0.0001), model_tf_optimizer=tf.train.AdamOptimizer(0.001), reward_tf_optimizer=tf.train.AdamOptimizer(0.001), value_tf_optimizer=tf.train.AdamOptimizer(0.001),  actor_target_approach_rate=0.99, value_target_approach_rate=0.99):
        if forward_steps < 1:
            raise Exception("At least one forward step has to be executed!")

        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.forward_steps = 1
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.actor_target_approach_rate = actor_target_approach_rate
        self.value_target_approach_rate = value_target_approach_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork(actor_network.name + "_target", actor_network, actor_target_approach_rate)

        self.model_network = model_network
        self.model_optimizer = SquaredLossOptimizer(model_network, model_tf_optimizer)

        self.reward_network = reward_network
        self.reward_optimizer = SquaredLossOptimizer(reward_network, reward_tf_optimizer)

        self.value_network = value_network
        self.value_target_network = TargetNeuralNetwork(value_network.name + "_target", value_network, value_target_approach_rate)
        self.value_optimizer = SquaredLossOptimizer(value_network, value_tf_optimizer)

        self.actor_ac_network, _actors, _models, _rewards, _values = ac.create_actor_model_critic_network(
            "Actor_AC", actor_network, model_network, reward_network, self.value_target_network, self.discount_factor, 1, False # TODO: should actor learn from multiple forward steps too?
        )
        self.value_ac_network, _actors, self.value_ac_models, self.value_ac_rewards, self.value_ac_values = ac.create_actor_model_critic_network(
            "Value_AC", self.actor_target_network, model_network, reward_network, self.value_target_network, self.discount_factor, forward_steps, True
        )

        self.actor_optimizer = MaxOutputOptimizer(
            self.actor_ac_network, actor_tf_optimizer, actor_network.get_parameters()
        )

        self.td_error_network = ac.create_model_based_td_error_network("TD_error", self.actor_target_network,
            self.model_network, self.reward_network, self.value_network, self.discount_factor
        )
        self.model_error_network = ac.create_squared_error_network("Model_error", model_network)
        self.reward_error_network = ac.create_squared_error_network("Reward_error", reward_network)

        self.actor_network.session.run(tf.global_variables_initializer())

    def train(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.train_model(state_batch, action_batch, next_state_batch)
        self.train_reward(state_batch, action_batch, reward_batch)
        self.train_actor(state_batch)
        self.train_value(state_batch)

    def train_actor(self, state_batch):
        self.actor_optimizer.train([state_batch])
        self.actor_target_network.approach_source_parameters()

    def train_model(self, state_batch, action_batch, next_state_batch):
        self.model_optimizer.train([state_batch, action_batch], next_state_batch)

    def train_reward(self, state_batch, action_batch, reward_batch):
        self.reward_optimizer.train([state_batch, action_batch], reward_batch)

    def train_value(self, state_batch):
        batch_size = len(state_batch)

        fetches = self.value_ac_network.custom_fetch([state_batch], fetch_layers=[
            [n.get_output_layer().get_output() for n in self.value_ac_models],
            [n.get_output_layer().get_output() for n in self.value_ac_rewards],
            [n.get_output_layer().get_output() for n in self.value_ac_values]
        ])

        model_outputs = fetches[0]
        reward_outputs = fetches[1]
        value_outputs = fetches[2]

        value_state_batch = []
        value_target_batch = []

        for b in xrange(batch_size):
            value_deltas = [0] * self.forward_steps

            for step in xrange(self.forward_steps):
                value_delta = reward_outputs[step][b] + self.discount_factor * value_outputs[step + 1][b] - value_outputs[step][b]
                value_deltas[step] += value_delta

                for t in xrange(step):
                    value_deltas[t] += pow(self.discount_factor * self.trace_decay, step - t) * value_delta

            current_state = state_batch[b]
            for step in xrange(self.forward_steps):
                value_state_batch.append(current_state)
                value_target_batch.append(value_outputs[step][b] + value_delta[step])
                current_state = model_outputs[step]

        value_target_batch = np.resize(value_target_batch, [batch_size * self.forward_steps, 1])

        self.value_optimizer.train([value_state_batch], value_target_batch)
        self.value_target_network.approach_source_parameters()

    def action(self, state):
        action = self.actor_network.predict([state])
        return action

    def get_td_error(self, state):
        return self.get_td_error_batch([state])[0]

    def get_td_error_batch(self, state_batch):
        return self.td_error_network.predict_batch([state_batch])

    def get_model_error(self, state, action, next_state):
        return self.get_model_error_batch([state], [action], [next_state])[0]

    def get_model_error_batch(self, state_batch, action_batch, next_state_batch):
        return np.asarray(self.model_error_network.predict_batch([state_batch, action_batch, next_state_batch])).reshape(-1)

    def get_reward_error(self, state, action, reward):
        return self.get_reward_error_batch([state], [action], [reward])[0]

    def get_reward_error_batch(self, state_batch, action_batch, reward_batch):
        return np.asarray(self.reward_error_network.predict_batch([state_batch, action_batch, reward_batch])).reshape(-1)
