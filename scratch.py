import numpy as np
import nnfs

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.001 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.weights += np.dot(self.weights.T, dvalues)
        self.dbiases += np.sum(dvalues, axis=1, keepdims=True)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.max(0, inputs)

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)


a = np.array([[ 1, 2, 3 ], [ 4, 5, 6 ]])
print(np.sum(a, axis=0, keepdims=True))
print(np.sum(a, axis=1, keepdims=True))