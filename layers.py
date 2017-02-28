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

        self.output = self.activation_fn(tf.matmul(self.input_layers[0].get_output(), self.W) + self.b, name=self.name)

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
            raise Exception("Layer " + self.name +  " is already compiled!")

        self.output = tf.concat(1, [l.get_output() for l in self.input_layers], name=(network.name + "_" + self.name))
        self.parameters = []

    def copy(self, new_name, input_layers):
        return ConcatLayer(new_name, input_layers)
