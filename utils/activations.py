import numpy as np

### Functions to compute activation in forward propagation
def linear(Z: np.ndarray) -> np.ndarray:
	''' Compute the linear activation function.
	Mainly used for output layer in case of classification. '''
	return Z

def sigmoid(Z: np.ndarray) -> np.ndarray:
	''' Compute the sigmoid activation function.
	Mainly used for output layer in case of classification. '''
	return 1 / (1 + np.exp(-Z))

def relu(Z: np.ndarray) -> np.ndarray:
	''' Compute the relu activation function.
	Commonly used for hidden layers. '''
	return np.maximum(0, Z)

def softmax(Z: np.ndarray) -> np.ndarray:
	''' Compute the softmax activation function.
	Commonly used for output layer.
	Numerically stable version. '''
	return np.exp(Z - np.max(Z)) / np.exp(Z - np.max(Z)).sum(axis=0, keepdims=True)

def tanh(Z: np.ndarray) -> np.ndarray:
	''' Compute the tanh activation function.
	Commonly used for hidden layers. '''
	return (np.exp(Z) - np.exp(-Z)) \
		/ (np.exp(Z) + np.exp(-Z))

def leaky_relu(Z: np.ndarray) -> np.ndarray:
	''' Compute the leaky relu activation function.
	Commonly used for hidden layers. '''
	return np.maximum(0.01 * Z, Z)

### Derivative functions of activations function to perform backward propagation
## In hidden layer only (non final)
def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
	''' Compute the derivative of sigmoid activation function
	when used as a hidden layer. '''
	return Z * (1 - Z)

def tanh_derivative(Z: np.ndarray) -> np.ndarray:
	''' Compute the derivative of tanh activation function
	when used as a hidden layer. '''
	return 1 - (Z**2)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
	''' Compute the derivative of relu activation function
	when used as a hidden layer. '''
	Z = Z.copy()
	Z[Z <= 0] = 0
	Z[Z > 0] = 1
	return Z

def leaky_relu_derivative(Z: np.ndarray) -> np.ndarray:
	''' Compute the derivative of relu activation function.
	when used as a hidden layer. '''
	Z = Z.copy()
	Z[Z < 0] = 0.01
	Z[Z >= 0] = 1
	return Z
