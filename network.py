import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()

X, y = spiral_data(100, 3)   


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # dvalues -> batch_count * neuron_count
        # inputs -> batch_count * input_count
        # weights -> input_count * neuron_count
        # biases -> 1 * neuron_count
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        # dL/db = dL/da*da/dz*dz/db = a'(z) * 1 = 1 if z>0, 0 if z <=0
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_inputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.output = norm_inputs
    def backward(self, dvalues):
        # dSij / dzik (i = sample_index, jth Softmax output, kth summed input)
        # = Sij * (kd - Sik) = Sij * kd - Sij * Sik (cd is Kronecker Delta Function / Matrix)
        # (jth Softmax output in ith sample) * (Kronecker Delta) - (jth Softmax output in ith sample) * (kth Softmax output in ith sample)
        self.dinputs = np.empty_like(dvalues)
        '''
        Example:
        self.output = np.array([
            [0.7, 0.2, 0.1],   # Sample 1
            [0.1, 0.3, 0.6]    # Sample 2
        ])
        dvalues = np.array([
            [1, 0, 0],   # Sample 1: true class is class 0
            [0, 0, 1]    # Sample 2: true class is class 2
        ])
        '''
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # single_output: single softmax output
            # single_dvalues: single gradient from the next layer
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # pyright: ignore[reportAttributeAccessIssue]
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_Cross_Entropy_Loss(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # save for backprop
        self.y_pred = y_pred_clipped
        self.y_true = y_true
        n_samples = len(y_pred)
        # for classification
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        # for one hot
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)
    # we do not need dvalues in this backpropagation method (no next layer for dvalues)
    def backward(self, dvalues):
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])
        # if labels are sparse, turn them to a one hot vector
        if len(self.y_true.shape) == 1:
            y_true = np.eye(n_labels)[self.y_true]
        self.dinputs = self.y_true / self.y_pred
        # Normalize gradients over the batch so that their scale
        # is independent of the batch size.
        self.dinputs /= n_samples


class Activation_Softmax_Loss_Categorical_Cross_Entropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Categorical_Cross_Entropy_Loss()
    def forward(self, inputs, y_true):
        self.y_true = y_true
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.y_pred = self.output
        return self.loss.calculate(self.output, y_true)
    def backward(self):
        n_samples = len(self.y_true)
        if len(self.y_true.shape) == 2:
            y_idx = np.argmax(self.y_true, axis=1)
        else:
            y_idx = self.y_true
        self.dinputs = np.copy(self.y_pred)
        self.dinputs[range(n_samples), y_idx] -= 1
        # Normalize
        self.dinputs /= n_samples
