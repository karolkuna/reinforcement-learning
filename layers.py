# Copyright 2017 Karol Kuna

import tensorflow as tf
from math import sqrt

class Layer:
    def get_id(self):
        return self.id

    def set_id(self, layer_id):
        self.id = layer_id

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_parameters(self):
        return self.parameters

    def get_input_layers(self):
        return self.input_layers

    def get_output(self):
        return self.output

    def get_parameter_count(self):
        pass

    def compile(self, network):
        pass

    def copy(self, new_name, input_layers):
        pass


class InputLayer(Layer):
    def __init__(self, name, size):
        self.id = None
        self.name = name
        self.size = size
        self.parameters = None
        self.input_layers = []
        self.output = None

    def set_input_layer(self, input_layer):
        self.input_layers = [input_layer]

    def get_parameter_count(self):
        return 0

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        if len(self.input_layers) == 0:
            self.output = tf.placeholder("float", [None, self.size], name=(network.name + "_" + self.name))
        else:
            self.output = self.input_layers[0].get_output()

        self.parameters = []

    def copy(self, new_name, input_layers):
        return InputLayer(self.name, self.size)


class FullyConnectedLayer(Layer):
    def __init__(self, name, size, input_layer, activation_fn=tf.nn.relu, weights_init_stddev=1.0, bias_init_stddev=1.0):
        self.id = None
        self.name = name
        self.size = size
        self.parameters = None
        self.input_layers = [input_layer]
        self.output = None

        self.activation_fn = activation_fn
        self.weights_init_stddev = weights_init_stddev
        self.bias_init_stddev = bias_init_stddev
        self.W = None
        self.b = None

    def get_parameter_count(self):
        return 2

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        if self.parameters is None:
            self.W = tf.Variable(
                tf.random_normal([self.input_layers[0].get_size(), self.size], mean=0.0,
                                 stddev=(self.weights_init_stddev / sqrt(self.input_layers[0].get_size()))),
                                 name=(network.name + "_" + self.name + "_W"))
            self.b = tf.Variable(tf.random_normal([self.size], mean=0.0, stddev=self.bias_init_stddev),
                                 name=(network.name + "_" + self.name + "_b"))
            self.parameters = [self.W, self.b]
        else:
            self.W = self.parameters[0]
            self.b = self.parameters[1]

        self.output = self.activation_fn(tf.matmul(self.input_layers[0].get_output(), self.W) + self.b, name=(network.name + "_" + self.name))

    def copy(self, new_name, input_layers):
        return FullyConnectedLayer(new_name, self.size, input_layers[0], self.activation_fn, self.weights_init_stddev, self.bias_init_stddev)


class ConcatLayer(Layer):
    def __init__(self, name, input_layers):
        self.id = None
        self.name = name
        self.size = sum([l.get_size() for l in input_layers])
        self.parameters = None
        self.input_layers = input_layers
        self.output = None

    def get_parameter_count(self):
        return 0

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        self.output = tf.concat(axis=1, values=[l.get_output() for l in self.input_layers], name=(network.name + "_" + self.name))
        self.parameters = []

    def copy(self, new_name, input_layers):
        return ConcatLayer(new_name, input_layers)


class AdditionLayer(Layer):
    def __init__(self, name, input_layers):
        if (len(input_layers) != 2):
            raise Exception("Addition layers requires 2 inputs!")

        if (input_layers[0].get_size() != input_layers[1].get_size()):
            raise Exception("Addition layer inputs must have same dimension!")

        self.id = None
        self.name = name
        self.size = input_layers[0].get_size()
        self.parameters = None
        self.input_layers = input_layers
        self.output = None

    def get_parameter_count(self):
        return 0

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        self.output = tf.add(self.input_layers[0].get_output(), self.input_layers[1].get_output(), name=(network.name + "_" + self.name))
        self.parameters = []

    def copy(self, new_name, input_layers):
        return AdditionLayer(new_name, input_layers)


class ScalarMultiplyLayer(Layer):
    def __init__(self, name, input_layer, scalar):
        self.id = None
        self.name = name
        self.size = input_layer.get_size()
        self.parameters = None
        self.input_layers = [input_layer]
        self.output = None
        self.scalar = scalar

    def get_parameter_count(self):
        return 0

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        self.output = tf.scalar_mul(self.scalar, self.input_layers[0].get_output())
        self.parameters = []

    def copy(self, new_name, input_layers):
        return ScalarMultiplyLayer(new_name, self.input_layers[0], self.scalar)


class DropoutLayer(Layer):
    def __init__(self, name, input_layer, keep_probability):
        self.id = None
        self.name = name
        self.size = input_layer.get_size()
        self.parameters = None
        self.input_layers = [input_layer]
        self.output = None
        self.keep_probability = keep_probability

    def get_parameter_count(self):
        return 0

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        dropout = tf.nn.dropout(self.input_layers[0].get_output(), self.keep_probability, name=(network.name + "_" + self.name))
        self.output = tf.cond(network.is_training, lambda: dropout, lambda: tf.identity(self.input_layers[0].get_output()))
        self.parameters = []

    def copy(self, new_name, input_layers):
        return DropoutLayer(new_name, input_layers[0], self.keep_probability)


class BatchNormalizationLayer(Layer):
    def __init__(self, name, input_layer, momentum=0.99, epsilon=0.001, offset=0.0, scale=1.0, moving_mean_init=0.0, moving_variance_init=1.0):
        self.id = None
        self.name = name
        self.size = input_layer.get_size()
        self.parameters = None
        self.input_layers = [input_layer]
        self.momentum = momentum
        self.epsilon = epsilon
        self.offset = offset
        self.scale = scale
        self.moving_mean_init = moving_mean_init
        self.moving_variance_init = moving_variance_init
        self.output = None

    def get_parameter_count(self):
        return 2

    def compile(self, network):
        if self.output is not None:
            raise Exception("Layer " + self.name + " is already compiled!")

        self.offset_const = tf.constant(self.offset, shape=[self.size], name=(network.name + "_" + self.name + "_offset"))
        self.scale_const = tf.constant(self.scale, shape=[self.size], name=(network.name + "_" + self.name + "_scale"))

        if self.parameters is None:
            self.moving_mean = tf.Variable(tf.constant(self.moving_mean_init, shape=[self.size]), name=(network.name + "_" + self.name + "_moving_mean"), trainable=False)
            self.moving_variance = tf.Variable(tf.constant(self.moving_variance_init, shape=[self.size]), name=(network.name + "_" + self.name + "_moving_variance"), trainable=False)
            self.parameters = [self.moving_mean, self.moving_variance]
        else:
            self.moving_mean = self.parameters[0]
            self.moving_variance = self.parameters[1]

        self.batch_mean, self.batch_variance = tf.nn.moments(self.input_layers[0].get_output(), axes=[0], name=(network.name + "_" + self.name + "_moments")) # TODO: use axes=[0,1,2] for image inputs

        def moments_training():
            update_mean = tf.assign_sub(self.moving_mean, (1 - self.momentum) * (self.moving_mean - self.batch_mean))
            update_variance = tf.assign_sub(self.moving_variance, (1 - self.momentum) * (self.moving_variance - self.batch_variance))

            with tf.control_dependencies([update_mean, update_variance]):
                return tf.identity(self.batch_mean), tf.identity(self.batch_variance)

        def moments_testing():
            # TODO: should moving averages be updated during testing?
            return tf.identity(self.moving_mean), tf.identity(self.moving_variance)

        mean, variance = tf.cond(network.is_training, moments_training, moments_testing)

        self.output = tf.nn.batch_normalization(self.input_layers[0].get_output(), mean, variance, self.offset_const, self.scale_const, self.epsilon, name=(network.name + "_" + self.name))

    def copy(self, new_name, input_layers):
        return BatchNormalizationLayer(new_name, input_layers[0], self.momentum, self.epsilon, self.offset, self.scale, self.moving_mean_init, self.moving_variance_init)

