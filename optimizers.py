# Copyright 2017 Karol Kuna

import tensorflow as tf
import neuralnetwork

class SquaredLossOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), var_list=None):
        if not network.is_compiled():
            raise Exception("Cannot run optimizer on uncompiled network!")

        self.network = network
        self.tf_optimizer = tf_optimizer
        self.var_list = var_list

        self.target = tf.placeholder("float", [None, self.network.get_output_layer().get_size()])
        self.cost = tf.reduce_mean(tf.square(self.network.get_output_layer().get_output() - self.target))
        self.optimizer_op = tf_optimizer.minimize(self.cost, var_list=var_list)

    def train(self, inputs, target):
        feed_dict = dict(zip([l.get_output() for l in self.network.get_input_placeholder_layers()], inputs))
        feed_dict[self.network.is_training] = True
        feed_dict[self.target] = target

        self.network.session.run(self.optimizer_op, feed_dict=feed_dict)

class MaxOutputOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), var_list=None):
        if not network.is_compiled():
            raise Exception("Cannot run optimizer on uncompiled network!")

        self.network = network
        self.tf_optimizer = tf_optimizer
        self.var_list = var_list

        self.cost = -tf.reduce_sum(self.network.get_output_layer().get_output())
        self.optimizer_op = tf_optimizer.minimize(self.cost, var_list=var_list)

    def train(self, inputs):
        feed_dict = dict(zip([l.get_output() for l in self.network.get_input_placeholder_layers()], inputs))
        feed_dict[self.network.is_training] = True

        self.network.session.run(self.optimizer_op, feed_dict=feed_dict)
