import numpy as np
import neural_network as nn
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)
dense1 = nn.Layer_Dense(2, 3)
activation1 = nn.Activation_ReLU()

dense2 = nn.Layer_Dense(3, 3)
loss_activation = nn.Activation_Softmax_Loss_Categorical_Cross_Entropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(activation1.output, y)
print(loss_activation.output[:5])
print('loss:', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print(accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
