import numpy as np

logits = np.array([
    [2.5, 0.3, -1.7, 3.0],
    [1.0, 2.2, 0.1, -0.5]
])
dvalues = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.5, -0.5, 0.0, 0.0]
])

softmax_output = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
output = np.empty_like(dvalues)
for index, (single_dvalue, single_softmax) in enumerate(zip(dvalues, softmax_output)):
    single_softmax = single_softmax.reshape(-1, 1)
    jacobian = np.diagflat(single_softmax) - np.dot(single_softmax, single_softmax.T)
    output[index] = np.dot(jacobian, single_dvalue)
print(output)
