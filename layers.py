import tensorflow as tf

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
