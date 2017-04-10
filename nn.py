# Copyright 2017 Karol Kuna

import tensorflow as tf
from neuralnetwork import NeuralNetwork
from layers import *


def fully_connected(name, session, input_dims, layer_dims, output_dim, activation_fn, output_bounds=None, batch_norm=False):
    network = NeuralNetwork(name, session, input_dims)
    if len(input_dims) == 1:
        prev_layer = network.get_input_layer(0)
    else:
        prev_layer = ConcatLayer("concat_inputs", network.input_layers)

    for i, size in enumerate(layer_dims):
        if batch_norm:
            bn = BatchNormalizationLayer("bn_hidden_" + str(i), prev_layer)
            prev_layer = bn

        layer = FullyConnectedLayer("hidden_" + str(i) + "_" + str(size), size, prev_layer, activation_fn)
        prev_layer = layer

    if batch_norm:
        bn = BatchNormalizationLayer("bn_output", prev_layer)
        prev_layer = bn

    output_layer = FullyConnectedLayer("output", output_dim, prev_layer, tf.identity)
    if output_bounds is not None:
        output_layer = BoundingLayer("bounding", output_layer, output_bounds.low, output_bounds.high)

    network.compile(output_layer)
    return network
