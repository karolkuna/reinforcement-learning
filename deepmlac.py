# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
from actorcritic import create_actor_model_critic_network

class DeepMLAC:
    def __init__(self, actor_network, model_network, reward_network, value_network, forward_steps=1, discount_factor=0.9, trace_decay=0.9, batch_size=128, replay_buffer_size=100000, learning_rate=0.0001, actor_target_approach_rate=0.999, value_target_approach_rate=0.999):
        if forward_steps < 1:
            raise Exception("At least one forward step has to be executed!")

        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.forward_steps = 1
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
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

        self.actor_ac_network, _actors, _models, _rewards, _values = create_actor_model_critic_network(
            "Actor_AC", actor_network, model_network, reward_network, self.value_target_network, self.discount_factor, 1, False # TODO: should actor learn from multiple forward steps too?
        )
        self.value_ac_network, _actors, self.value_ac_models, self.value_ac_rewards, self.value_ac_values = create_actor_model_critic_network(
            "Value_AC", self.actor_target_network, model_network, reward_network, self.value_target_network, self.discount_factor, forward_steps, True
        )

        self.actor_optimizer = MaxOutputOptimizer(
            self.actor_ac_network, tf.train.AdamOptimizer(learning_rate), actor_network.get_parameters()
        )

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.state_dim, self.action_dim)

        self.actor_network.session.run(tf.global_variables_initializer())

    def train(self, train_actor=True, train_model=True, train_reward=True, train_value=True):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, ids = self.replay_buffer.get_batch(self.batch_size)

        if train_actor:
            self.actor_optimizer.train([state_batch])
            self.actor_target_network.approach_source_parameters()

        if train_model:
            self.model_optimizer.train([state_batch, action_batch], next_state_batch)

        if train_reward:
            self.reward_optimizer.train([state_batch, action_batch], reward_batch)

        if train_value:
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

            for b in range(self.batch_size):
                value_deltas = [0] * self.forward_steps

                for step in range(self.forward_steps):
                    value_delta = reward_outputs[step][b] + self.discount_factor * value_outputs[step + 1][b] - value_outputs[step][b]
                    value_deltas[step] += value_delta

                    for t in range(step):
                        value_deltas[t] += pow(self.discount_factor * self.trace_decay, step - t) * value_delta

                current_state = state_batch[b]
                for step in range(self.forward_steps):
                    value_state_batch.append(current_state)
                    value_target_batch.append(value_outputs[step][b] + value_delta[step])
                    current_state = model_outputs[step]

            value_target_batch = np.resize(value_target_batch, [self.batch_size * self.forward_steps, 1])

            self.value_optimizer.train([value_state_batch], value_target_batch)
            self.value_target_network.approach_source_parameters()

    def reset_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.state_dim, self.action_dim)

    def action(self, state):
        action = self.actor_network.predict([state])
        return action

    def noisy_action(self, state, mean=0, stddev=1.0):
        return self.actor_network.predict([state]) + np.random.normal(mean, stddev, self.action_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)