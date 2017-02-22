import tensorflow as tf
import math

class Layer:
    def get_size(self):
        pass

    def get_output(self):
        pass

    def get_trainable_parameters(self):
        pass


class IdentityLayer(Layer):
    def __init__(self, size, input):
        self.size = size
        self.output = input

    def get_size(self):
        return self.size

    def get_output(self):
        return self.output

    def get_trainable_parameters(self):
        return []

class InputLayer(Layer):
    def __init__(self, size, name=None):
        self.size = size
        self.name = name
        self.output = tf.placeholder("float", [None, size], name=self.name)

    def get_size(self):
        return self.size

    def get_output(self):
        return self.output

    def get_trainable_parameters(self):
        return []


class FullyConnectedLayer(Layer):
    parameter_count = 2

    def __init__(self, input_layer, size, activation_fn=tf.nn.relu, name=None):
        self.input_layer = input_layer
        self.input_dim = input_layer.get_size()
        self.size = size
        self.activation_fn = activation_fn
        self.name = name
        self.W = None
        self.b = None
        self.parameters = None
        self.output = None

    def get_size(self):
        return self.size

    def get_output(self):
        return self.output

    def get_trainable_parameters(self):
        return [self.W, self.b]

    def init_parameters(self, weights_init_stddev=1.0, bias_init_stddev=1.0):
        self.W = tf.Variable(tf.random_normal([self.input_dim, self.size], mean=0.0, stddev=weights_init_stddev), name=(self.name + "_W"))
        self.b = tf.Variable(tf.random_normal([self.size], mean=0.0, stddev=bias_init_stddev), name=(self.name + "_b"))
        self.parameters = [self.W, self.b]

        self.output = self.activation_fn(tf.matmul(self.input_layer.get_output(), self.W) + self.b, name=self.name)

    def share_parameters(self, source_parameters):
        self.W = source_parameters[0]
        self.b = source_parameters[1]
        self.parameters = [self.W, self.b]

        self.output = self.activation_fn(tf.matmul(self.input_layer.get_output(), self.W) + self.b, name=self.name)


class ConcatLayer(Layer):
    def __init__(self, layers, name):
        self.size = 0
        self.name = name
        inputs = []

        for layer in layers:
            self.size += layer.get_size()
            inputs.append(layer.get_output())

        self.output = tf.concat(1, inputs, name=self.name)

    def get_size(self):
        return self.size

    def get_output(self):
        return self.output

    def get_trainable_parameters(self):
        return []


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

        for input_id in range(len(input_dims)):
            input_dim = input_dims[input_id]
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

        for layer_id in range(len(layer_sizes)):
            layer_size = layer_sizes[layer_id]
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
    def __init__(self, networks, composition_input_ids):
        self.name = "_".join([nn.name for nn in networks])
        self.original_networks = networks
        self.networks = []
        self.inputs = []

        for id in range(len(self.original_networks)):
            original_network = self.original_networks[id]
            input_variables = [None] * len(original_network.input_dims)
            if id > 0:
                input_variables[composition_input_ids[id - 1]] = self.networks[id - 1].output

            network = NeuralNetwork(self.name + "_" + str(id),
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

            for input_id in range(len(network.inputs)):
                if input_id == 0 or input_id != composition_input_ids[id - 1]:
                    self.inputs.append(network.inputs[input_id])

        self.session = self.networks[0].session
        self.output = self.networks[-1].output

    def predict(self, inputs):
        return self.session.run(self.output, feed_dict=dict(zip(self.inputs, inputs)))

