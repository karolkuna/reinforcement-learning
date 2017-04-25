# Copyright 2017 Karol Kuna

import tensorflow as tf
import neuralnetwork

class SquaredLossOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), parameters=None, l2=None, summary_writer=None):
        if not network.is_compiled():
            raise Exception("Cannot run optimizer on uncompiled network!")

        self.network = network
        self.tf_optimizer = tf_optimizer

        self.target = tf.placeholder("float", [None, self.network.get_output_layer().get_size()])
        self.cost = tf.reduce_mean(tf.square(self.network.get_output_layer().get_output() - self.target))

        if parameters is None and l2 is not None:
            raise Exception("Parameters argument must be set when L2 regularization is enabled!")

        if parameters is not None:
            var_list = [p.tf_variable for p in parameters if p.trainable]

        if l2 is None:
            self.total_cost = self.cost
        else:
            regularizable_vars = [p.tf_variable for p in parameters if p.regularizable]
            if len(regularizable_vars) == 0:
                raise Exception("Regularization is enabled, but there are no parameters to regularize!")

            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in regularizable_vars])
            self.total_cost = self.cost + l2 * l2_loss

        self.optimizer_op = tf_optimizer.minimize(self.total_cost, var_list=var_list)
        self.summary_writer = summary_writer
        self.summary_op = tf.summary.scalar(network.name + "_squared_loss", self.total_cost)
        self.step = 0

    def train(self, inputs, target):
        feed_dict = dict(zip([l.get_output() for l in self.network.get_input_placeholder_layers()], inputs))
        feed_dict[self.network.is_training] = True
        feed_dict[self.target] = target

        cost, summary = self.network.session.run([self.optimizer_op, self.summary_op], feed_dict=feed_dict)
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summary, self.step)
            self.step += 1

class MaxOutputOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), parameters=None, l2=None):
        if not network.is_compiled():
            raise Exception("Cannot run optimizer on uncompiled network!")

        self.network = network
        self.tf_optimizer = tf_optimizer
        self.cost = -tf.reduce_sum(self.network.get_output_layer().get_output())

        if parameters is None and l2 is not None:
            raise Exception("Parameters argument must be set when L2 regularization is enabled!")

        if parameters is not None:
            var_list = [p.tf_variable for p in parameters if p.trainable]

        if l2 is None:
            self.total_cost = self.cost
        else:
            regularizable_vars = [p.tf_variable for p in parameters if p.regularizable]
            if len(regularizable_vars) == 0:
                raise Exception("Regularization is enabled, but there are no parameters to regularize!")

            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in regularizable_vars])
            self.total_cost = self.cost + l2 * l2_loss


        self.optimizer_op = tf_optimizer.minimize(self.total_cost, var_list=var_list)

    def train(self, inputs):
        feed_dict = dict(zip([l.get_output() for l in self.network.get_input_placeholder_layers()], inputs))
        feed_dict[self.network.is_training] = True

        self.network.session.run(self.optimizer_op, feed_dict=feed_dict)
