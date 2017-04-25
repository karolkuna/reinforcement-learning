class Parameter:
    def __init__(self, tf_variable, trainable=True, regularizable=False):
        self.tf_variable = tf_variable
        self.trainable = trainable
        self.regularizable = regularizable
