# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
from actorcritic import create_actor_model_critic_network

class AMC:
    def __init__(self, actor_network, model_network, reward_network, value_network, forward_steps=1, discount_factor=0.9, tf_optimizer=tf.train.AdamOptimizer(0.0001), actor_target_approach_rate=0.999, value_target_approach_rate=0.999):
        if forward_steps < 1:
            raise Exception("At least one forward step has to be executed!")

        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.forward_steps = 1
        self.discount_factor = discount_factor
        self.actor_target_approach_rate = actor_target_approach_rate
        self.value_target_approach_rate = value_target_approach_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork(actor_network.name + "_target", actor_network, actor_target_approach_rate)

        self.model_network = model_network
        self.model_optimizer = SquaredLossOptimizer(model_network, tf_optimizer)

        self.reward_network = reward_network
        self.reward_optimizer = SquaredLossOptimizer(reward_network, tf_optimizer)

        self.value_network = value_network
        self.value_target_network = TargetNeuralNetwork(value_network.name + "_target", value_network, value_target_approach_rate)
        self.value_optimizer = SquaredLossOptimizer(value_network, tf_optimizer)

        self.actor_ac_network, _actors, _models, _rewards, _values = create_actor_model_critic_network(
            "Actor_AC", actor_network, model_network, reward_network, self.value_target_network, self.discount_factor, 1, False # TODO: should actor learn from multiple forward steps too?
        )
        self.value_ac_network, _actors, _models, _rewards, _values = create_actor_model_critic_network(
            "Value_AC", self.actor_target_network, model_network, reward_network, self.value_target_network, self.discount_factor, forward_steps, False
        )

        self.actor_optimizer = MaxOutputOptimizer(
            self.actor_ac_network, tf_optimizer, actor_network.get_parameters()
        )

        self.actor_network.session.run(tf.global_variables_initializer())

    def train(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.train_model(state_batch, action_batch, next_state_batch)
        self.train_reward(state_batch, action_batch, reward_batch)
        self.train_actor(state_batch)
        self.train_value(state_batch, done_batch)

    def train_actor(self, state_batch):
        self.actor_optimizer.train([state_batch])
        self.actor_target_network.approach_source_parameters()

    def train_model(self, state_batch, action_batch, next_state_batch):
        self.model_optimizer.train([state_batch, action_batch], next_state_batch)

    def train_reward(self, state_batch, action_batch, reward_batch):
        self.reward_optimizer.train([state_batch, action_batch], reward_batch)

    def train_value(self, state_batch, done_batch):
        value_target_batch = self.value_ac_network.predict_batch([state_batch])
        self.value_optimizer.train([state_batch], value_target_batch)
        self.value_target_network.approach_source_parameters()

    def action(self, state):
        action = self.actor_network.predict([state])
        return action
