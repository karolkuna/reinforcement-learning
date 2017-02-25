import tensorflow as tf
import math
import layers

class NeuralNetwork:
    def __init__(self, name, session, input_dims, layer_sizes, output_dim, activation_fn=tf.nn.relu, weights_init_stddev=1.0,
                 bias_init_stddev=1.0, shared_parameters=None, input_variables=None):
        self.name = name
        self.session = session
        self.input_dims = input_dims
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.weights_init_stddev = weights_init_stddev
        self.bias_init_stddev = bias_init_stddev

        self.input_layers = []
        self.layers = []
        self.inputs = []
        self.trainable_parameters = []

        for input_id, input_dim in enumerate(input_dims):
            if input_variables is None or input_variables[input_id] is None:
                input_layer = InputLayer(input_dim, name=(self.name + "_inputLayer" + str(input_id)))
            else:
                input_layer = IdentityLayer(input_dim, input_variables[input_id])

            self.input_layers.append(input_layer)
            self.inputs.append(input_layer.get_output())

        prev_layer = None
        if len(self.input_layers) == 1:
            prev_layer = self.input_layers[0]
        else:
            prev_layer = ConcatLayer(self.input_layers, name=(self.name + "_concatLayer"))

        for layer_id, layer_size in enumerate(layer_sizes):
            layer = FullyConnectedLayer(prev_layer, layer_size, activation_fn, name=(self.name + "_hiddenLayer" + str(layer_id)))
            self.layers.append(layer)
            prev_layer = layer

        self.output_layer = FullyConnectedLayer(prev_layer, output_dim, tf.identity, name=(self.name + "_outputLayer"))
        self.layers.append(self.output_layer)

        if shared_parameters is None:
            prev_layer_size = sum(input_dims)
            for layer in self.layers:
                layer.init_parameters(weights_init_stddev / math.sqrt(prev_layer_size), bias_init_stddev)
                self.trainable_parameters.extend(layer.get_trainable_parameters())
                prev_layer_size = layer.size
        else:
            par_id = 0
            for layer in self.layers:
                layer.share_parameters(shared_parameters[par_id:par_id + layer.parameter_count])
                self.trainable_parameters.extend(layer.get_trainable_parameters())
                par_id += layer.parameter_count

        self.output = self.output_layer.get_output()

    def predict(self, inputs):
        return self.session.run(self.output, feed_dict=dict(zip(self.inputs, inputs)))


class TargetNeuralNetwork:
    def __init__(self, source_network, approach_rate):
        """
        :type source_network: NeuralNetwork
        """
        self.source_network = source_network
        self.exponential_ma = tf.train.ExponentialMovingAverage(decay=approach_rate)
        self.approach_parameters_op = self.exponential_ma.apply(source_network.trainable_parameters)
        self.shared_parameters = [self.exponential_ma.average(par) for par in source_network.trainable_parameters]
        self.network = NeuralNetwork(source_network.name + "_target",
                                     source_network.session,
                                     source_network.input_dims,
                                     source_network.layer_sizes,
                                     source_network.output_dim,
                                     source_network.activation_fn,
                                     source_network.weights_init_stddev,
                                     source_network.bias_init_stddev,
                                     self.shared_parameters)

    def approach_source_parameters(self):
        self.network.session.run(self.approach_parameters_op)

    def predict(self, inputs):
        return self.network.predict(inputs)


class NeuralNetworkComposition:
    def __init__(self, networks, composition_inputs):
        self.name = "_".join([nn.name for nn in networks])
        self.original_networks = networks
        self.networks = []
        self.inputs = []

        for network_id, original_network in enumerate(self.original_networks):
            input_variables = [None] * len(original_network.input_dims)
            for input_id in range(len(input_variables)):
                input_network_id = composition_inputs.pop(0)
                if input_network_id is not None:
                    input_variables[input_id] = self.networks[input_network_id].output

            network = NeuralNetwork(self.name + "_" + str(network_id),
                                    original_network.session,
                                    original_network.input_dims,
                                    original_network.layer_sizes,
                                    original_network.output_dim,
                                    original_network.activation_fn,
                                    original_network.weights_init_stddev,
                                    original_network.bias_init_stddev,
                                    original_network.trainable_parameters,
                                    input_variables)
            self.networks.append(network)

            for input_id in range(len(input_variables)):
                if input_variables[input_id] is None:
                    self.inputs.append(network.inputs[input_id])

        self.session = self.networks[0].session
        self.output = self.networks[-1].output

    def predict(self, inputs):
        return self.session.run(self.output, feed_dict=dict(zip(self.inputs, inputs)))

