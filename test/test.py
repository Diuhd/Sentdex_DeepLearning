import numpy as np
from timeit import timeit
from neural_network import Activation_Softmax, Categorical_Cross_Entropy_Loss, Activation_Softmax_Loss_Categorical_Cross_Entropy

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
def f1():
    softmax_loss = Activation_Softmax_Loss_Categorical_Cross_Entropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs
def f2():
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss = Categorical_Cross_Entropy_Loss()
    loss.backward(activation.output, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)

print(t2/t1)