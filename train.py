import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, 
                                             keepdims=True))
        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()



# Cross-entropy loss
class Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = y_pred.shape[0]
        # Probabilities for target values
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]
        # Losses
        negative_log_likelihoods = -np.log(y_pred)
        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true
        # Average loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss
    def backward(self, dvalues, y_true):
        samples = dvalues.shape[0]
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

# Create dataset
X, y  = spiral_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Create 2nd Dense layer with 3 input features
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Make a forward pass of training data 
dense1.forward(X)
activation1.forward(dense1.output)

# Make a forward pass through 2nd Dense layer
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# Calculate loss from output of activation2
loss = loss_function.forward(activation2.output, y)
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions==y)
print('acc:', accuracy)

# Backward pass
loss_function.backward(activation2.output, y)
activation2.backward(loss_function.dvalues)
dense2.backward(activation2.dvalues)
activation1.backward(dense2.dvalues)
dense1.backward(activation1.dvalues)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)