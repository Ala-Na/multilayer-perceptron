from typing import Tuple
from utils.dense_layer import DenseLayer
import numpy as np

class SimpleNeuralNetwork():
	''' A simple Neural Network model. '''

	supported_loss = ['cross_entropy', 'binary_cross_entropy']
	supported_regularization = ['l2', None]
	supported_optimization = ['rmsprop', 'adam', 'momentum', None]

	def __init__(self, layers: list, X: np.ndarray, Y: np.ndarray, \
		X_val: np.ndarray | None = None, Y_val: np.ndarray | None = None, \
		alpha: float = 0.001, loss: str = 'cross_entropy', \
		regularization: str = None, lambda_: float = 1.0, \
		optimization: str | None = None, beta_1: float = 0.9, \
		beta_2: float = 0.99, epsilon: float = 1e-8, \
		name: str = 'Model') -> None:
		''' Initiation function of model '''

		# Basics checks
		assert loss in self.supported_loss
		assert regularization in self.supported_regularization
		assert isinstance(lambda_, float) and lambda_ >= 0.0 and lambda_ <= 1.0
		assert isinstance(alpha, float) and alpha > 0.0 and alpha < 1.0
		assert optimization in self.supported_optimization
		assert isinstance(beta_1, float) and (beta_1 > 0 and beta_1 <= 1)
		assert isinstance(beta_2, float) and (beta_2 > 0 and beta_2 <= 1)
		assert isinstance(epsilon, float)

		# Assignation and others checks
		loss_functions = [self.cross_entropy_loss, self.binary_cross_entropy_loss]
		self.loss = loss_functions[self.supported_loss.index(loss)]
		self.lambda_ = lambda_ if regularization != None else 0
		self.alpha = alpha
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		assert self._check_valid_arrays(loss) == True
		self.layers = layers
		assert self._check_valid_layers() == True
		self.losses = []
		self.val_losses = []
		self.infos = {}
		self.optimization = optimization
		self.name = name
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self._uniformate_layers(regularization)

	def _check_valid_arrays(self, loss: str) -> bool:
		''' Check validity of training and validation set. '''
		if not isinstance(self.X, np.ndarray) and self.X.ndim == 2 and self.X.size != 0:
			print("\033[93mError in received input\033[0m")
			return False
		elif not isinstance(self.Y, np.ndarray) and self.Y.ndim == 2 and self.Y.size != 0:
			print("\033[93mError in received output\033[0m")
			return False
		if loss == 'binary_cross_entropy' and self.Y.shape[1] != 2:
			print("\033[93mCan't use binary cross entropy loss on more than 2 classes\033[0m")
			return False
		if self.X_val is not None or self.Y_val is not None:
			if self.X_val is None or self.Y_val is None:
				print("\033[93mMissing either validation input or output\033[0m")
				return False
			elif not isinstance(self.X_val, np.ndarray) and self.X_val.size != 0 \
				and self.X_val.shape[1] == self.X.shape[1]:
				print("\033[93mError in received validation input\033[0m")
				return False
			elif not isinstance(self.Y_val, np.ndarray) and self.Y_val.size != 0 \
				and self.Y_val.shape[1] == self.Y.shape[1]:
				print("\033[93mError in received validation output\033[0m")
				return False
		return True

	def _check_valid_layers(self) -> bool:
		''' Check validity of layers inside model. '''
		for layer in self.layers:
			if not isinstance(layer, DenseLayer):
				print("\033[93mSimpleNeuralNetwork must receive and list of DenseLayer\033[0m")
				return False
		if self.X.shape[1] != self.layers[0].weights.shape[1]:
			print("\033[93mError between input shape and first layer input shape\033[0m")
			return False
		elif self.Y.shape[1] != self.layers[-1].weights.shape[0]:
			print("\033[93mError between output shape and last layer output shape\033[0m")
			return False
		L = len(self.layers)
		for i, layer in enumerate(self.layers):
			if i != 0:
				if layer.weights.shape[1] != l_prev.weights.shape[0] :
					print("\033[93mError of compatibility between layers\033[0m")
					return False
			if i != L - 1 and layer.derivative is None:
				print("\033[93mFinal layer in non-final position\033[0m")
				return False
			l_prev = layer
		if l_prev.derivative is not None:
			print("\033[93mNon final layer in final position\033[0m")
			return False
		return True

	def _uniformate_layers(self, regularization) -> None:
		''' Standardize layers settings with model settings. '''
		info = False
		for layer in self.layers:
			if info == False and (layer.alpha != self.alpha or layer.lambda_ != self.lambda_ \
				or layer.optimization != self.optimization or layer.beta_1 != self.beta_1 \
				or layer.beta_2 != self.beta_2 or layer.epsilon != self.epsilon):
				info = True
				print("\033[93mAll layers will be standardized with model settings\033[0m")
			layer.alpha = self.alpha
			layer.lambda_ = self.lambda_
			layer.beta_1 = self.beta_1
			layer.beta_2 = self.beta_2
			layer.epsilon = self.epsilon
			layer.optimization = self.optimization

	def forward_propagation(self, val: bool = False) -> np.ndarray:
		''' Perform the forward propgation in model neural network. '''
		if val == False:
			layers_activations = [self.X.T]
			self.infos["A0"] = self.X.T
		else:
			layers_activations = [self.X_val.T]
		for i, layer in enumerate(self.layers):
			A, Z = layer.forward(layers_activations[i])
			layers_activations.append(A)
			if val == False:
				self.infos["Z" + str(i + 1)] = Z
				self.infos["A" + str(i + 1)] = A
		return A

	def regularization_cost(self) -> float:
		''' Calculate the regularization factor for L2 regularized cost. '''
		m = self.Y.shape[1]
		cst_reg = (1/m) * (self.lambda_/2)
		weights_reg = 0
		for layer in self.layers:
			weights_reg += np.sum(np.square(layer.weights))
		return cst_reg * weights_reg

	def binary_cross_entropy_loss(self) -> float:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y)
		for binary classification '''
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		assert self.layers[L - 1].activation_name == 'sigmoid' \
			or self.layers[L - 1].activation_name == 'softmax'
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		m = self.Y.shape[0]
		cost = -1/m * np.sum(self.Y * np.log(A_last.T + eps) \
			+ (1 - self.Y) * np.log(1 - A_last.T + eps))
		if self.lambda_ != 0:
			cost += self.regularization_cost()
		cost_val = None
		if isinstance(self.X_val, np.ndarray) and isinstance(self.Y_val, np.ndarray):
			A_last_val = self.forward_propagation(val=True)
			m_val = self.Y_val.shape[0]
			cost_val = -1/m_val * np.sum(self.Y_val * np.log(A_last_val.T + eps) \
				+ (1 - self.Y_val) * np.log(1 - A_last_val.T + eps))
			if self.lambda_ != 0:
				cost_val += self.regularization_cost()
		return np.squeeze(cost), np.squeeze(cost_val)

	def cross_entropy_loss(self) -> float:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y)
		for multi-classes classification '''
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		assert self.layers[L - 1].activation_name == 'softmax'
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		cost = -np.mean(self.Y * np.log(A_last.T + eps))
		if self.lambda_ != 0:
			cost += self.regularization_cost()
		cost_val = None
		if isinstance(self.X_val, np.ndarray) and isinstance(self.Y_val, np.ndarray):
			A_last_val = self.forward_propagation(val=True)
			cost_val = -np.mean(self.Y_val * np.log(A_last_val.T + eps))
			if self.lambda_ != 0:
				cost_val += self.regularization_cost()
		return np.squeeze(cost), np.squeeze(cost_val)

	def backward_propagation(self) -> None:
		''' Perform the backward propgation in model neural network. '''
		L = len(self.layers)
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		A_prev = self.infos["A" + str(L - 1)]
		dA_prev, dW, db = self.layers[-1].final_backward(A_last, self.Y, A_prev)
		self.infos["dA" + str(L - 1)] = dA_prev
		self.infos["dW" + str(L)] = dW
		self.infos["db" + str(L)] = db
		for l in range(L - 1, 0, -1):
			A_prev = self.infos["A" + str(l - 1)]
			Z = self.infos["Z" + str(l)]
			dA_prev, dW, db = self.layers[l - 1].hidden_backward(dA_prev, \
				A_prev, Z)
			self.infos["dA" + str(l - 1)] = dA_prev
			self.infos["dW" + str(l)] = dW
			self.infos["db" + str(l)] = db

	def update(self, time: int) -> None:
		''' Update parameters for each layer. '''
		for i, layer in enumerate(self.layers):
			dW = self.infos["dW" + str(i + 1)]
			db = self.infos["db" + str(i + 1)]
			layer.update_parameters(dW, db, time)

	def fit(self, nb_iterations: int = 10000) -> None:
		''' Perform forward and backward propagation on a given number of epochs. '''
		assert nb_iterations > 0
		print("{} training:".format(self.name))
		for i in range(nb_iterations):
			self.forward_propagation()
			train_loss, val_loss = self.loss()
			self.losses.append(train_loss)
			print("epoch {}/{} - loss: {}".format(i + 1, nb_iterations, self.losses[i]), end='')
			if val_loss != None:
				self.val_losses.append(val_loss)
				print(" - val_loss: {}".format(self.val_losses[i]), end='')
			print(end='\n')
			self.backward_propagation()
			self.update(i + 1)
		return self.losses, self.val_losses # historique des loss
