from typing import Tuple
from dense import DenseLayer
import numpy as np

class Model():
	def __init__(self, X: np.ndarray, Y: np.ndarray, \
		X_val: np.ndarray, Y_val: np.ndarray, \
		layers: list):
		''' Initiation function of model '''
		# TODO check X and Y (assert np.numpy, etc...)
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		#TODO check shapes of layers
		self.layers = layers
		self.activations = []
		self.derivatives = []
		return

	def forward_propagation(self) -> None:
		layers_activations = [self.input]
		for i, layer in enumerate(self.layers):
			A = layer.predict(self.activations[i])
			layers_activations.append(A)
		self.activations = layers_activations
		# TODO save previous activations ?

	def cost(self) -> np.ndarray or None:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y) '''
		eps=1e-15
		A_last = self.activations[-1]
		one_vec = np.ones((1, self.Y.shape[1]))
		m = self.Y.shape[0]
		return ((-1 / m) * (self.Y.T.dot(np.log(A_last + eps)) \
			+ (one_vec - self.Y).T.dot(np.log(one_vec - A_last + eps))) \
			+ ((self.lambda_ / (2 * m)) * self.l2(self.theta))).item()

	def backward_propagation(self) -> np.ndarray:

