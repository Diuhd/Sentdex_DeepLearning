import numpy as np
import neural_network as nn
from nnfs.datasets import spiral_data
from nnfs import init

init()

X, y = spiral_data(samples=100, classes=3)
dense1 = nn.Layer_Dense(2, 64)
activation1 = nn.Activation_ReLU()

dense2 = nn.Layer_Dense(64, 3)
loss_activation = nn.Activation_Softmax_Loss_Categorical_Cross_Entropy()

optimizer = nn.Optimizer_RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
'''
1.0 -> Too high, loss gets stuck in 0.6~0.8 (local minimum)
0.85 -> Still high, loss gets stuck in 0.45~0.8
1.0, decay=0.01 -> Learning rate decreased too fast -- loss barely changes
1.0, decay=0.001 -> 
'''

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

