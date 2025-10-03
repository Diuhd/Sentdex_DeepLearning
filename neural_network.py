import numpy as np

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
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_inputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.output = norm_inputs
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
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
        return -np.log(correct_confidences) # type: ignore
    
    # we do not need dvalues in this backpropagation method (no next layer for dvalues)
    def backward(self, dvalues=None, y_true=None):
        """
        Computes dL/dy_pred for CCE.
        If dvalues/y_true are provided, use them (like in NNFS book).
        Otherwise use cached self.y_pred/self.y_true.
        """
        if dvalues is None:
            dvalues = self.y_pred
        if y_true is None:
            y_true = self.y_true

        samples = dvalues.shape[0]
        n_labels = dvalues.shape[1]

        # numerical safety
        dvalues_safe = np.clip(dvalues, 1e-7, 1-1e-7)

        # sparse -> one-hot (handles (n,) or (n,1))
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            indices = y_true.ravel().astype(int)
            y_true_onehot = np.eye(n_labels)[indices]
        else:
            y_true_onehot = y_true

        # dL/dy_pred
        self.dinputs = -y_true_onehot / dvalues_safe
        self.dinputs /= samples


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
    def backward(self, dvalues=None, y_true=None):
        """
        Simplified gradient for Softmax+CCE:
          dinputs = (y_pred - y_true_onehot) / samples
        Accepts optional dvalues (softmax output) and y_true like in the book.
        """
        if dvalues is None:
            dvalues = self.y_pred
        if y_true is None:
            y_true = self.y_true

        samples = dvalues.shape[0]

        # indices for correct classes (supports sparse or one-hot)
        if y_true.ndim == 2:
            y_idx = np.argmax(y_true, axis=1)
        else:
            y_idx = y_true.ravel().astype(int)

        self.dinputs = dvalues.copy()
        self.dinputs[np.arange(samples), y_idx] -= 1
        self.dinputs /= samples
