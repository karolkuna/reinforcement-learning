# Copyright 2017 Karol Kuna

import tensorflow as tf
from layers import InputLayer
from parameter import Parameter


class INeuralNetwork:
    def get_input_layer(self, input_id):
        pass

    def set_input_layer(self, input_id, input_layer):
        pass

    def get_output_layer(self):
        pass

    def get_input_placeholder_layers(self):
        pass

    def compile(self, output):
        pass

    def is_compiled(self):
        pass

    def set_parameters(self, parameters):
        pass

    def get_parameters(self):
        pass

    def copy(self, new_name, reuse_parameters=False):
        pass

    def predict_batch(self, inputs):
        pass

    def predict(self, inputs):
        pass


class NeuralNetwork(INeuralNetwork):
    def __init__(self, name, session, input_dims):
        self.name = name
        self.session = session
        self.input_dims = input_dims

        self.layers = []
        self.connections = []
        self.parameters = []
        self.input_layers = []
        self.input_placeholder_layers = []

        for input_id, input_dim in enumerate(input_dims):
            input_layer = InputLayer(self.name + "_input_" + str(input_id), input_dim)
            self.input_layers.append(input_layer)
            self.input_placeholder_layers.append(input_layer)

        self.is_training = tf.placeholder(tf.bool, name=(name + "_is_training"))

        self.output_layer = None
        self.output_dim = None
        self.compiled = False

    def get_input_layer(self, input_id):
        return self.input_layers[input_id]

    def set_input_layer(self, input_id, input_layer):
        if self.compiled:
            raise Exception("Network is already compiled. Changes are not allowed!")

        if self.input_dims[input_id] != input_layer.get_size():
            raise Exception("Input dimensions do not match!")

        self.input_placeholder_layers.remove(self.input_layers[input_id])
        self.input_layers[input_id].set_input_layer(input_layer)

    def get_output_layer(self):
        return self.output_layer

    def get_input_placeholder_layers(self):
        return self.input_placeholder_layers

    def explore_layer_inputs(self, layer):
        for input_layer in layer.get_input_layers():
            if input_layer.get_id() is None:
                self.explore_layer_inputs(input_layer)

        layer.set_id(len(self.layers))
        self.layers.append(layer)
        self.connections.append([l.get_id() for l in layer.get_input_layers()])

    def compile(self, output_layer, unconnected_layers=None):
        if self.compiled:
            raise Exception("Network is already compiled!")

        self.output_layer = output_layer
        self.output_dim = output_layer.get_size()

        self.layers = []
        # search network backwards from output to find all connected layers
        self.explore_layer_inputs(output_layer)

        if unconnected_layers:
            for l in unconnected_layers:
                self.explore_layer_inputs(l)

        for input_layer in self.input_layers:
            if input_layer.get_id() is None:
                raise Exception("Output is unreachable from input " + input_layer.name + "!\nNetwork: " + str(self))

        for layer in self.layers:
            layer.compile(self)
            self.parameters.extend(layer.get_parameters())

        self.compiled = True

    def is_compiled(self):
        return self.compiled

    def set_parameters(self, parameters):
        if self.compiled:
            raise Exception("Cannot change parameters of compiled network!")

        expected_parameter_count = sum([l.get_parameter_count() for l in self.layers])
        if len(parameters) != expected_parameter_count:
            raise Exception(
                "Expected " + str(expected_parameter_count) + " parameters, but got " + str(len(parameters)) + " instead!")

        for layer in self.layers:
            layer.set_parameters(parameters[0:layer.get_parameter_count()])
            parameters = parameters[layer.get_parameter_count():]

    def get_parameters(self):
        return self.parameters

    def copy(self, new_name, reuse_parameters=False):
        if not self.compiled:
            raise Exception("Cannot make a copy of uncompiled network!")

        new_network = NeuralNetwork(new_name, self.session, self.input_dims)

        for layer_id, layer in enumerate(self.layers):
            is_input = False
            for input_id, input_layer in enumerate(self.input_layers):
                if layer == input_layer:
                    is_input = True
                    new_network.layers.append(new_network.input_layers[input_id])
                    break

            if is_input:
                continue

            input_layers = [new_network.layers[i] for i in self.connections[layer_id]]
            layer_copy = layer.copy(layer.name, input_layers)
            new_network.layers.append(layer_copy)
            if layer == self.output_layer:
                new_network.output_layer = layer_copy

        if reuse_parameters:
            new_network.set_parameters(self.parameters)

        return new_network

    def predict_batch(self, inputs):
        if not self.compiled:
            raise Exception("Network must be compiled first!")

        feed_dict = dict(zip([l.get_output() for l in self.input_placeholder_layers], inputs))
        feed_dict[self.is_training] = False

        return self.session.run(self.output_layer.get_output(), feed_dict=feed_dict)

    def predict(self, inputs):
        return self.predict_batch([[i] for i in inputs])[0]

    def custom_fetch(self, inputs, fetch_layers):
        feed_dict = dict(zip([l.get_output() for l in self.input_placeholder_layers], inputs))
        feed_dict[self.is_training] = False

        return self.session.run(fetch_layers, feed_dict=feed_dict)

    def __str__(self):
        network_str = ""
        for layer in self.layers:
            network_str += str([l.get_name() for l in layer.get_input_layers()]) + " --> " + layer.get_name() + "\n"

        return network_str


class TargetNeuralNetwork(INeuralNetwork):
    def __init__(self, name, source_network, approach_rate):
        self.name = name
        self.source_network = source_network
        self.approach_rate = approach_rate

        if not source_network.is_compiled():
            raise Exception("Cannot create a target network from uncompiled source network!")

        self.exponential_ma = tf.train.ExponentialMovingAverage(decay=approach_rate)
        self.approach_parameters_op = self.exponential_ma.apply([par.tf_variable for par in source_network.get_parameters()])
        self.target_parameters = [Parameter(self.exponential_ma.average(par.tf_variable), trainable=False) for par in source_network.get_parameters()]
        self.target_network = source_network.copy(name)
        self.target_network.set_parameters(self.target_parameters)
        self.target_network.compile(self.target_network.get_output_layer())
        self.session = self.source_network.session
        self.input_dims = source_network.input_dims
        self.is_training = self.target_network.is_training

    def approach_source_parameters(self):
        self.session.run(self.approach_parameters_op)

    def get_input_layer(self, input_id):
        return self.target_network.get_input(input_id)

    def set_input_layer(self, input_id, input_layer):
        return self.target_network.get_input(input_id)

    def get_output_layer(self):
        return self.target_network.get_output_layer()

    def get_input_placeholder_layers(self):
        return self.target_network.get_input_placeholder_layers()

    def compile(self, output):
        return self.target_network.compile(output)

    def is_compiled(self):
        return self.target_network.is_compiled()

    def set_parameters(self, parameters):
        raise Exception("Changing parameters of a target network is not supported!")

    def get_parameters(self):
        return self.target_parameters

    def copy(self, new_name, reuse_parameters=False):
        return self.target_network.copy(new_name, reuse_parameters)

    def predict_batch(self, inputs):
        return self.target_network.predict_batch(inputs)

    def predict(self, inputs):
        return self.target_network.predict(inputs)
