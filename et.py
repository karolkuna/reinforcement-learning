# Copyright 2017 Karol Kuna

import tensorflow as tf
import numpy as np
from replaybuffer import ReplayBuffer
from neuralnetwork import NeuralNetwork, TargetNeuralNetwork
from optimizers import SquaredLossOptimizer, MaxOutputOptimizer
from layers import ScalarMultiplyLayer, AdditionLayer

class ET:
    def __init__(self, actor_network, model_network, reward_network, eligibility_network, value_network, discount_factor=0.9, batch_size=128, replay_buffer_size=100000, learning_rate=0.0001, actor_target_approach_rate=0.999, value_target_approach_rate=0.999):
        self.state_dim = actor_network.input_dims[0]
        self.action_dim = actor_network.output_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate

        self.actor_network = actor_network
        self.actor_target_network = TargetNeuralNetwork(actor_network.name + "_target", actor_network, actor_target_approach_rate)

        self.model_network = model_network
        self.model_optimizer = SquaredLossOptimizer(model_network, tf.train.AdamOptimizer(learning_rate))

        self.reward_network = reward_network
        self.reward_optimizer = SquaredLossOptimizer(reward_network, tf.train.AdamOptimizer(learning_rate))

        self.value_network = value_network
        self.value_target_network = TargetNeuralNetwork(value_network.name + "_target", value_network, value_target_approach_rate)
        self.value_optimizer = SquaredLossOptimizer(value_network, tf.train.AdamOptimizer(learning_rate))

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.state_dim, self.action_dim)


    def train(self):
