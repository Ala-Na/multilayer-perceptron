from typing import Tuple
from dense import DenseLayer
import numpy as np

class SimpleNeuralNetwork():

	supported_loss = ['cross_entropy', 'binary_cross_entropy']

	def __init__(self, X: np.ndarray, Y: np.ndarray, \
		layers: list, \
		X_val: np.ndarray = None, Y_val: np.ndarray = None, \
		loss: str = 'cross_entropy'):
		''' Initiation function of model '''
		assert loss in self.supported_loss
		self.loss = loss
		# TODO check X and Y (assert np.numpy, etc...)
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		#TODO check shapes of layers
		self.layers = layers
		self.infos= {}
		return

	def forward_propagation(self) -> None:
		''' Perform the forward propgation in model neural network. '''
		layers_activations = [self.X.T]
		self.infos["A0"] = self.X.T
		for i, layer in enumerate(self.layers):
			A, Z = layer.forward(layers_activations[i])
			layers_activations.append(A)
			self.infos["Z" + str(i + 1)] = Z
			self.infos["A" + str(i + 1)] = A

	def binary_cross_entropy_loss(self) -> float:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y) '''
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		assert self.layers[L - 1].activation_name == 'sigmoid'
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		m = self.Y.shape[1]
		cost = -1/m * np.sum(self.Y * np.log(A_last + eps) + (1 - self.Y) * np.log(1 - A_last + eps))
		return np.squeeze(cost)

	def cross_entropy_loss(self) -> float:
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		assert self.layers[L - 1].activation_name == 'softmax'
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		cost = -np.mean(self.Y * np.log(A_last.T + eps))
		return np.squeeze(cost)

	def backward_propagation(self) -> None:
		L = len(self.layers)
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		A_prev = self.infos["A" + str(L - 1)]
		dA_prev, dW, db = self.layers[-1].final_backward(A_last, self.Y, A_prev)
		self.infos["dA" + str(L - 1)] = dA_prev
		self.infos["dW" + str(L)] = dW
		self.infos["db" + str(L)] = db
		print("dW", str(L), dW.shape)
		print("db", str(L), db.shape)
		print("dA", str(L - 1), dA_prev.shape)
		for l in range(L - 1, 0, -1):
			A = self.infos["A" + str(l)]
			Z = self.infos["Z" + str(l)]
			dA_prev, dW, db = self.layers[l - 1].hidden_backward(dA_prev, \
				A, Z)
			print("dW", str(l), dW.shape)
			print("db", str(l), db.shape)
			print("dA", str(l - 1), dA_prev.shape)
			self.infos["dA" + str(l - 1)] = dA_prev
			self.infos["dW" + str(l)] = dW
			self.infos["db" + str(l)] = db

	def update(self) -> None:
		for i, layer in enumerate(self.layers):
			dW = self.infos["dW" + str(i + 1)]
			db = self.infos["db" + str(i + 1)]
			layer.update_parameters(dW, db)
