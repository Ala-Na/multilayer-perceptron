from utils.activations import *
import numpy as np
from typing import Tuple

class DenseLayer():
	''' A dense model '''

	# Supported initialization, regularization and optimization algorithm
	supported_initialization = ['random', 'zeros', 'he', 'xavier', 'yosh']
	supported_regularization = ['l2', None]
	supported_optimization = ['momentum', 'rmsprop', 'adam', None]

	# Supported activations functions for hidden and output layer
	supported_activation = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
	supported_final_activation = ['softmax', 'linear', 'sigmoid']

	def __init__(self, input_shape: int, output_shape: int, \
			final: bool = False, \
			initialization: str = 'random', activation: str = 'relu', \
			alpha: float = 0.001, regularization: str = None, lambda_: float = 1.0, \
			beta_1: float = 0.9, beta_2: float = 0.99, epsilon: float = 1e-8, \
			optimization: str = None) -> None :

		# Basics checks
		assert isinstance(final, bool)
		assert isinstance(input_shape, int)
		assert isinstance(output_shape, int)
		assert isinstance(alpha, float)
		assert isinstance(beta_1, float) and (beta_1 > 0 and beta_1 <= 1)
		assert isinstance(beta_2, float) and (beta_2 > 0 and beta_2 <= 1)
		assert isinstance(epsilon, float)
		assert isinstance(lambda_, float) and lambda_ >= 0.0 and lambda_ <= 1.0
		assert regularization in self.supported_regularization
		assert optimization in self.supported_optimization
		assert initialization in self.supported_initialization

		# Assignation and others checks
		self.weights, self.bias = self.initialize_parameters(input_shape, output_shape, initialization)
		self.alpha = alpha
		self.original_alpha = alpha
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.optimization = optimization
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
		self.lambda_ = lambda_ if regularization != None else 0
		self.initialize_step_size()
		self.initialize_velocity()

	def initialize_parameters(self, input_shape: int, output_shape: int, \
		init: str) -> Tuple[np.ndarray, np.ndarray]:
		'''Initialize parameters (weights and bias) of model'''
		np.random.seed(42) # TODO delete
		if init == 'random':
			weights = np.random.randn(output_shape, input_shape) * 0.01
		elif init == 'zeros':
			weights = np.zeros((output_shape, input_shape))
		elif init == 'he':
			weights = np.random.randn(output_shape, input_shape) * np.sqrt(2./input_shape)
		elif init == 'xavier':
			weights = np.random.randn(output_shape, input_shape) / np.sqrt(1./input_shape)
		elif init == 'yosh':
			weights = np.random.randn(output_shape, input_shape) / np.sqrt(2./(input_shape + output_shape))
		bias = np.zeros((output_shape, 1))
		return weights, bias

	def initialize_velocity(self) -> None:
		''' Initialize velocity for performing momentum or adam optimization'''
		self.weights_velocity = np.zeros(self.weights.shape)
		self.bias_velocity = np.zeros(self.bias.shape)

	def initialize_step_size(self) -> None:
		''' Initialize velocity for performing RMSprop or adam optimization'''
		self.weights_step_size = np.zeros(self.weights.shape)
		self.bias_step_size = np.zeros(self.bias.shape)

	def forward(self, A_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		''' Predict activation result according to A_prev (input) and parameters. '''
		Z = np.dot(self.weights, A_prev) + self.bias
		return self.activation(Z), Z

	def final_backward(self, A_last: np.ndarray, Y: np.ndarray, \
		A_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		m = A_prev.shape[1]
		dZ = A_last - Y.T
		dW = 1/m * np.dot(dZ, A_prev.T)
		if self.lambda_ != 0.0:
			dW += (self.lambda_/m) * self.weights
		db = 1/m * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(self.weights.T, dZ)
		return dA_prev, dW, db

	def hidden_backward(self, dA: np.ndarray, A_prev: np.ndarray, \
			Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		assert self.derivative is not None
		m = A_prev.shape[1]
		dZ = dA * self.derivative(Z)
		dW = 1/m * np.dot(dZ, A_prev.T)
		if self.lambda_ != 0.0:
			dW += (self.lambda_/m) * self.weights
		db = 1/m * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(self.weights.T, dZ)
		return dA_prev, dW, db

	def update_parameters(self, dW: np.ndarray, db:np.ndarray, time: int) -> None:
		if self.optimization == 'momentum':
			self.update_with_momentum(dW, db)
		elif self.optimization == 'rmsprop':
			self.update_with_rmsprop(dW, db)
		elif self.optimization == 'adam':
			self.update_with_adam(dW, db, time)
		else:
			self.weights = self.weights - self.alpha * dW
			self.bias = self.bias - self.alpha * db

	def update_with_momentum(self, dW: np.ndarray, db:np.ndarray) -> np.ndarray:
		''' Perform parameters update with momentum '''
		self.weights_velocity = self.beta_1 * self.weights_velocity \
			+ (1 - self.beta_1) * dW
		self.bias_velocity = self.beta_1 * self.bias_velocity \
			+ (1 - self.beta_1) * db
		self.weights = self.weights - self.alpha * self.weights_velocity
		self.bias = self.bias - self.alpha * self.bias_velocity

	def update_with_rmsprop(self, dW: np.ndarray, db:np.ndarray)-> np.ndarray:
		''' Perform parameters update with RMSprop'''
		self.weights_step_size = self.beta_2 * self.weights_step_size \
			+ (1 - self.beta_2) * (dW**2)
		self.bias_step_size = self.beta_2 * self.bias_step_size \
			+ (1 - self.beta_2) * (db**2)
		self.weights = self.weights - self.alpha \
			* dW / (np.sqrt(self.weights_step_size) + self.epsilon)
		self.bias = self.bias - self.alpha \
			* db / (np.sqrt(self.bias_step_size) + self.epsilon)

	def update_with_adam(self, dW: np.ndarray, db:np.ndarray, time: int) -> None:
		''' Perform parameters update with Adam '''
		self.weights_velocity = self.beta_1 * self.weights_velocity \
			+ (1 - self.beta_1) * dW
		self.bias_velocity = self.beta_1 * self.bias_velocity \
			+ (1 - self.beta_1) * db
		corrected_weights_velocity = self.weights_velocity / (1 -  self.beta_1**time)
		corrected_bias_velocity = self.bias_velocity / (1 -  self.beta_1**time)
		self.weights_step_size = self.beta_2 * self.weights_step_size \
			+ (1 - self.beta_2) * (dW**2)
		self.bias_step_size = self.beta_2 * self.bias_step_size \
			+ (1 - self.beta_2) * (db**2)
		corrected_weights_step_size = self.weights_step_size / (1 - self.beta_2**time)
		corrected_bias_step_size = self.bias_step_size / (1 - self.beta_2**time)
		self.weights = self.weights - self.alpha * \
			(corrected_weights_velocity / (np.sqrt(corrected_weights_step_size) + self.epsilon))
		self.bias = self.bias - self.alpha * \
			(corrected_bias_velocity / (np.sqrt(corrected_bias_step_size) + self.epsilon))
