import numpy as np
from typing import Tuple
import os

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
	Commonly used for output layer. '''
	return np.exp(Z) / np.sum(np.exp(Z))

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
	Z[Z <= 0] = 0
	Z[Z > 0] = 1
	return Z

def leaky_relu_derivative(Z: np.ndarray) -> np.ndarray:
	''' Compute the derivative of relu activation function.
	when used as a hidden layer. '''
	Z[Z < 0] = 0.01
	Z[Z >= 0] = 1
	return Z

# TODO modify because was previously logistic regression class
# TODO we need to create a model class which will take dense layer model and train them
# TODO check initialization values
class DenseLayer():
	''' A dense model'''

	# Supported initialization, regularization and optimization algorithm
	supported_initialization = ['random', 'zeros', 'he', 'other'] #TODO delete other
	supported_regularization = ['l2', None]
	supported_optimization = ['momentum', 'rmsprop', 'adam', None]
	supported_activation = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
	supported_final_activation = ['softmax', 'linear', 'sigmoid']

	def __init__(self, input_shape: int, output_shape: int, \
			final: bool = False, \
			initialization: str = 'random', activation: str = 'relu', \
			alpha: float = 0.001):#, beta_1: float = 0.9, beta_2: float = 0.99, \
			#epsilon: float = 1e-8, lambda_: float = 1.0, max_iter: int = 10000, \
			#regularization: str = 'l2', optimization: str = None, \
			#early_stopping: bool = False, decay: bool = False, \
			#decay_rate: float = 0.1, decay_interval: int or None = 1000) -> None:
		assert isinstance(final, bool)
		assert isinstance(input_shape, int)
		assert isinstance(output_shape, int)
		assert isinstance(alpha, float)
		# assert isinstance(beta_1, float) and (beta_1 > 0 and beta_1 <= 1)
		# assert isinstance(beta_2, float) and (beta_2 > 0 and beta_2 <= 1)
		# assert isinstance(epsilon, float)
		# assert isinstance(lambda_, float)
		# assert isinstance(max_iter, int)
		# assert isinstance(early_stopping, bool)
		# assert isinstance(decay, bool)
		# assert isinstance(decay_rate, float) and decay_rate >= 0 \
		# 	and decay_rate <= 1
		# assert (isinstance(decay_interval, int) and decay_interval > 0) or \
		# 	isinstance(decay_interval, None)
		# assert regularization in self.supported_regularization
		# assert optimization in self.supported_optimization
		assert initialization in self.supported_initialization
		self.weights, self.bias = self.parameters_initialization(input_shape, output_shape, initialization)
		self.alpha = alpha
		self.original_alpha = alpha
		# self.beta_1 = beta_1
		# self.beta_2 = beta_2
		# self.max_iter= max_iter
		# self.regularization = regularization
		# self.optimization = optimization
		if final is False:
			assert activation in self.supported_activation
			act = self.supported_activation
			derivative_functions = [relu_derivative, sigmoid_derivative, \
				tanh_derivative, leaky_relu_derivative]
			self.derivative = derivative_functions[self.supported_activation.index(activation)]
			activation_functions = [relu, sigmoid, tanh, leaky_relu]
		else:
			assert activation in self.supported_final_activation
			act = self.supported_final_activation
			self.derivative = None
			activation_functions = [softmax, linear, sigmoid]
		self.activation = activation_functions[act.index(activation)]
		self.activation_name = activation
		# self.lambda_ = lambda_ if regularization != None else 0
		# if self.lambda_ < 0:
		# 	raise ValueError("Lambda must be positive")
		# self.early_stopping = early_stopping
		# self.decay = decay
		# self.decay_rate = decay_rate
		# self.decay_interval = decay_interval if decay_interval != None \
		# 	else max_iter
		# self.initialize_step_size()
		# self.initialize_velocity()

	def parameters_initialization(self, input_shape: int, output_shape: int, \
		init: str) -> Tuple[np.ndarray, np.ndarray]:
		'''Initialize parameters (weights and bias) of model'''
		# random initialization with number between 0 and 1
		if init == 'random':
			weights = np.random.randn(output_shape, input_shape) * 0.01
		# init with zeros
		elif init == 'zeros':
			weights = np.zeros((output_shape, input_shape))
		# init with He initialization, would make more sense in a NN as 1 would be shape of l - 1
		elif init == 'he':
			weights = np.random.randn(output_shape, input_shape) * np.sqrt(2 / input_shape)
		elif init == 'other': # TODO delete
			print(output_shape, input_shape)
			weights = np.random.randn(output_shape, input_shape) / np.sqrt(input_shape)
		bias = np.zeros((output_shape, 1))
		print(weights, bias)
		return weights, bias

	def forward(self, A_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		''' Predict activation result according to A_prev (input) and parameters. '''
		Z = np.dot(self.weights, A_prev) + self.bias
		return self.activation(Z), Z

	def final_backward(self, A_last: np.ndarray, Y: np.ndarray, \
		A_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		m = A_prev.shape[1]
		dZ = A_last - Y.T
		dW = 1/m * np.dot(dZ, A_prev.T)
		db = 1/m * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(self.weights.T, dZ)
		return dA_prev, dW, db

	def hidden_backward(self, dA: np.ndarray, A_prev: np.ndarray, \
			Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		assert self.derivative is not None
		m = A_prev.shape[1]
		dZ = dA * self.derivative(Z)
		dW = 1/m * np.dot(dZ, A_prev.T)
		db = 1/m * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(self.weights.T, dZ)
		return dA_prev, dW, db

	def update_parameters(self, dW: np.ndarray, db:np.ndarray) -> None:
		self.weights = self.weights - self.alpha * dW
		self.bias = self.bias - self.alpha * db

	# def cost_derivative(self, x: np.ndarray, y: np.ndarray) \
	# 		-> np.ndarray or None:
	# 	''' Derivative calculus of cost, necessary to perform gradient descent'''
	# 	if not isinstance(x, np.ndarray) \
	# 			or not np.issubdtype(x.dtype, np.number) \
	# 			or x.ndim != 2 or x.size == 0 \
	# 			or x.shape[1] != self.theta.shape[0] - 1:
	# 		return None
	# 	if not isinstance(y, np.ndarray) \
	# 			or not np.issubdtype(y.dtype, np.number) \
	# 			or y.ndim != 2 or y.shape != (x.shape[0], 1):
	# 		return None
	# 	m = y.shape[0]
	# 	X = np.insert(x, 0, 1.0, axis=1)
	# 	y_hat = 1 / (1 + np.exp(-X @ self.theta))
	# 	theta_cp = self.theta.copy()
	# 	theta_cp[0][0] = 0
	# 	return (1 / m) * (X.T.dot(y_hat - y) + self.lambda_ * theta_cp)

	# def create_mini_batches(self, x: np.ndarray, y: np.ndarray, \
	# 		batch_size: int) -> list:
	# 	'''
	# 	Function to create mini batches of input and output for mini batches
	# 	gradient descent.
	# 	If batch_size == 1, it's stochastic gradient descent
	# 	'''
	# 	mini_batches = []
	# 	p = np.random.permutation(len(x))
	# 	shuffled_x, shuffled_y = x[p], y[p]
	# 	for i in range((shuffled_x.shape[0] // batch_size)):
	# 		if x.shape[0] % batch_size == 0 :
	# 			x_batch = shuffled_x[(i * batch_size) : ((i + 1) * batch_size)]
	# 			y_batch = shuffled_y[(i * batch_size) : ((i + 1) * batch_size)]
	# 		else :
	# 			x_batch = shuffled_x[(i * batch_size):]
	# 			y_batch = shuffled_y[(i * batch_size):]
	# 		mini_batches.append((x_batch, y_batch))
	# 	return mini_batches

	# def initialize_velocity(self) -> None:
	# 	''' Initialize velocity for performing momentum or adam optimization'''
	# 	self.velocity = np.zeros(self.theta.shape)

	# def initialize_step_size(self) -> None:
	# 	''' Initialize velocity for performing RMSprop or adam optimization'''
	# 	self.step_size = np.zeros(self.theta.shape)

	# def update_without_optimization(self, gradients: np.ndarray) -> np.ndarray:
	# 	''' Perform simple gradient descent update of parameters (theta) '''
	# 	return self.alpha * gradients

	# def update_with_momentum(self, gradients: np.ndarray) -> np.ndarray:
	# 	''' Perform parameters (theta) update with momentum '''
	# 	self.velocity = self.beta_1 * self.velocity \
	# 		+ (1 - self.beta_1) * gradients
	# 	return self.alpha * self.velocity

	# def update_with_rmsprop(self, gradients: np.ndarray)-> np.ndarray:
	# 	''' Perform parameters (theta) update with RMSprop'''
	# 	self.step_size = self.beta_2 * self.step_size \
	# 		+ (1 - self.beta_2) * gradients
	# 	return self.alpha * gradients / (np.sqrt(gradients) + self.epsilon)

	# def update_with_adam(self, gradients: np.ndarray, time: int) -> None:
	# 	''' Perform parameters (theta) update with Adam '''
	# 	self.velocity = self.beta_1 * self.velocity \
	# 		+ (1 - self.beta_1) * gradients
	# 	corrected_velocity = self.velocity / (1 -  self.beta_1**time)
	# 	self.step_size = self.beta_2 * self.step_size \
	# 		+ (1 - self.beta_2) * gradients
	# 	corrected_step_size = self.step_size / (1 - self.beta_2**time)
	# 	return self.alpha \
	# 		* (corrected_velocity/(np.sqrt(corrected_step_size) + self.epsilon))

	# def perform_update(self, x: np.ndarray, y: np.ndarray, time: int) -> bool:
	# 	''' Call corresponding update function.
	# 	Perform early stopping if option is activated. '''
	# 	gradients = self.cost_derivative(x, y)
	# 	if self.regularization == 'momentum':
	# 		diff = self.update_with_momentum(gradients)
	# 	elif self.regularization == 'rmsprop':
	# 		diff = self.update_with_rmsprop(gradients)
	# 	elif self.regularization == 'adam':
	# 		diff = self.update_with_adam(gradients, time)
	# 	else:
	# 		diff = self.update_without_optimization(gradients)
	# 	self.theta = self.theta - diff
	# 	if self.early_stopping and diff.all() < 1e-6:
	# 		return True
	# 	return False

	# def perform_learning_rate_decay(self, nb_epoch: int):
	# 	''' Perform learning rate (alpha decay) each decay interval '''
	# 	self.alpha = self.original_alpha \
	# 		/ (1 + self.decay_rate * np.floor(nb_epoch / self.decay_interval))

	# def gradient_descent(self, x: np.ndarray, y: np.ndarray, \
	# 		x_val: np.ndarray or None = None, y_val: np.ndarray or None = None, \
	# 		batch_size: int or None = None) -> Tuple[list, list] or None:
	# 	'''
	# 	Batch_size = 1 to run stochastic gradient descent
	# 	x_val and y_val : Validation set to stop training if stable, if not
	# 	present, training will stop once a threshold of 1e-6 is reached between
	# 	two thetas iteration. MSE on validation set is only checked each 100
	# 	epochs to avoid slowing down the training.
	# 	'''

	# 	t = 0 # initialize for adam
	# 	if batch_size == None or batch_size <= 0 : # if batch_size None or
	# 		# non valid, batch gradient descent is performed
	# 		batch_size = x.shape[1]
	# 	# Iterate over epochs
	# 	for i in range(0, self.max_iter):
	# 		# Mini_batches are created
	# 		mini_batches = self.create_mini_batches(x, y, batch_size)
	# 		for x_batch, y_batch in mini_batches:
	# 			t += 1
	# 			early_stop = self.perform_update(x_batch, y_batch, t)
	# 		# Case if early stopping by low difference in theta
	# 		if early_stop is True:
	# 			break
	# 		# Early-stopping option is performed according to MSE
	# 		if y_val is not None and x_val is not None and i % 100 == 0:
	# 			mse.append(self.mean_squared_error(y_val, self.predict(x_val)))
	# 			if len(mse) >= 2 and mse[-2] - mse[-1] < 1e-6:
	# 				self.theta = previous_theta
	# 				break
	# 			previous_theta = self.theta
	# 		# Learning rate (alpha) decay if option activated
	# 		if self.decay:
	# 			self.perform_learning_rate_decay(i)
	# 	# Return MSE if stored
	# 	if y_val is not None and x_val is not None:
	# 		return mse

	# def mean_squared_error(self, y: np.ndarray, y_hat: np.ndarray) -> float:
	# 	'''
	# 	Calculate mean squared error between an expected and predicted output
	# 	'''
	# 	if not isinstance(y, np.ndarray) \
	# 			or not np.issubdtype(y.dtype, np.number) \
	# 			or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
	# 		return None
	# 	if not isinstance(y_hat, np.ndarray) \
	# 			or not np.issubdtype(y_hat.dtype, np.number) \
	# 			or y_hat.ndim != 2 or y_hat.shape[1] != 1 \
	# 			or y_hat.shape[0] != y.shape[0]:
	# 		return None
	# 	mse = ((y_hat - y) ** 2).mean(axis=None)
	# 	return float(mse)

	# def save_values_npz(self, filepath='theta.npz') -> None:
	# 	''' Save theta into an npz file'''
	# 	if (os.path.isfile(filepath)):
	# 			os.remove(filepath)
	# 	try:
	# 		np.savez(filepath, theta=self.theta)
	# 	except:
	# 		print("\033[91mOops, can't save values in {} file.\033[0m".format(filepath))

	# def get_values_npz(self, filepath='theta.npz') -> None:
	# 	''' Retrieve theta from an npz file '''
	# 	try:
	# 		values = np.load(filepath)
	# 		assert np.issubdtype(values['theta'].dtype, np.number) and values['theta'].shape == self.theta.shape
	# 		self.theta = values['theta']
	# 	except:
	# 		print("\033[91mOops, can't get values from {} file.\033[0m".format(filepath))

	# def set_values(self, theta: np.ndarray) -> None:
	# 	''' Set theta according to values if shapes are compatibles '''
	# 	assert np.issubdtype(theta.dtype, np.number) and theta.shape == self.theta.shape
	# 	self.theta = theta


	# def l2(self) -> np.ndarray:
	# 	''' Perform L2 regularization '''
	# 	theta_cp = self.theta.copy()
	# 	theta_cp[0][0] = 0
	# 	return np.sum(theta_cp ** 2)
