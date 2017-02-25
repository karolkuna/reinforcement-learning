import tensorflow as tf
import neuralnetwork

class SquaredLossOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), var_list=None):
        self.network = network
        self.tf_optimizer = tf_optimizer
        self.var_list = var_list

        self.target = tf.placeholder("float", [None, self.network.output_dim])
        self.cost = tf.reduce_mean(tf.square(self.network.output - self.target))
        self.optimizer_op = tf_optimizer.minimize(self.cost, var_list=var_list)

    def train(self, inputs, target):
        feed_dict = dict(zip(self.network.inputs, inputs))
        feed_dict[self.target] = target

        self.network.session.run(self.optimizer_op, feed_dict=feed_dict)

class MaxOutputOptimizer:
    def __init__(self, network, tf_optimizer=tf.train.AdamOptimizer(0.0001), var_list=None):
        self.network = network
        self.tf_optimizer = tf_optimizer
        self.var_list = var_list

        self.cost = -tf.reduce_sum(self.network.output)
        self.optimizer_op = tf_optimizer.minimize(self.cost, var_list=var_list)

    def train(self, inputs):
        feed_dict = dict(zip(self.network.inputs, inputs))

        self.network.session.run(self.optimizer_op, feed_dict=feed_dict)