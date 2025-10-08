import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, 
                 weight_regularizer_l1=0., bias_regularizer_l1=0.,
                 weight_regularizer_l2=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # dvalues -> batch_count * neuron_count
        # inputs -> batch_count * input_count
        # weights -> input_count * neuron_count
        # biases -> 1 * neuron_count
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

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

class Layer_Dropout:
    def __init__(self, rate) -> None:
        # In here, rate means the rate of dropout (ex. 0.9 -> 1x1, 0x9)
        # ie. TensorFlow & Keras -> rate of dropout, PyTorch -> rate of non-dropout
        self.rate = 1 - rate # We change rate of dropout to rate of non-dropout for easy use
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate # Bernoulli distribution
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        # let a value in the Bernoulli distribution of 'rate' at index i be {r_i}.
        # * rate here means self.rate *
        # the pd of Dropout is 1/rate when {r_i} == 1, and 0 when {r_i} == 0.
        # so, the pd of Dropout can be generallized to {r_i}/rate, which is equal to self.binary_mask.
        self.dinputs = dvalues * self.binary_mask

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # pyright: ignore[reportAttributeAccessIssue]
        data_loss = np.mean(sample_losses)
        return data_loss
    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

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
        if dvalues == None: dvalues = self.y_pred
        if y_true == None: y_true = self.y_true
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= n_samples
        

class Activation_Softmax_Loss_Categorical_Cross_Entropy(Loss):
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

class Optimizer_SGD:
    # Note: momentum SGD is based off of classical momentum, not nesterov momentum
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.current_learning_rate = learning_rate
        self.momentum = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.iterations * self.decay))
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_AdaGrad:
    def __init__(self, learning_rate=1., decay=0., eps=1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.eps = eps
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.eps)
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., eps=1e-7, rho=0.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.rho = rho
        self.iterations = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * 1 / (1 + self.decay * self.iterations)
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.eps)
    def post_update_params(self):
        self.iterations += 1

# Generally, Adam optimizers perform best in 1e-3 decaying to 1e-4.
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., eps=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
    def pre_update_params(self):
        self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # EWMA formula: m_t​=(1−α)m_(t−1)​+αg_t​
        # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums   + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected   = layer.bias_momentums   / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache   + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected   = layer.bias_cache   / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.eps)
        layer.biases  += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.eps)

    def post_update_params(self):
        self.iterations += 1


