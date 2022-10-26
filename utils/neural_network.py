from typing import Tuple
from utils.dense_layer import DenseLayer
import numpy as np
from utils.metrics import *
import pickle

class SimpleNeuralNetwork():
	''' A simple Neural Network model. '''

	supported_loss = ['cross_entropy', 'binary_cross_entropy']
	supported_regularization = ['l2', None]
	supported_optimization = ['rmsprop', 'adam', 'momentum', None]
	supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

	def __init__(self, layers: list, X: np.ndarray, Y: np.ndarray, \
		X_val: np.ndarray or None = None, Y_val: np.ndarray or None = None, \
		alpha: float = 0.001, loss: str = 'binary_cross_entropy', \
		regularization: str = None, lambda_: float = 1.0, \
		optimization: str or None = None, beta_1: float = 0.9, \
		beta_2: float = 0.99, epsilon: float = 1e-8, \
		name: str = 'Model', metrics: list = [], stop: int or None = None) -> None:
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
		assert isinstance(stop, int) or stop is None

		# Assignation and others checks
		loss_functions = [self.cross_entropy_loss, self.binary_cross_entropy_loss]
		self.loss = loss_functions[self.supported_loss.index(loss)]
		self.lambda_ = lambda_ if regularization != None else 0
		self.alpha = alpha
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		if X_val is not None and Y_val is not None:
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
		for metric in metrics:
			assert metric in self.supported_metrics, "\033[91mMetric {} not recognized\033[0m".format(metric)
		self.metrics = metrics
		self.stop = stop

	def _check_valid_arrays(self, loss: str) -> bool:
		''' Check validity of training and validation set. '''
		if not isinstance(self.X, np.ndarray) and self.X.ndim == 2 and self.X.size != 0:
			print("\033[91mError in received input\033[0m")
			return False
		elif not isinstance(self.Y, np.ndarray) and self.Y.ndim == 2 and self.Y.size != 0:
			print("\033[91mError in received output\033[0m")
			return False
		if loss == 'binary_cross_entropy' and self.Y.shape[0] != 2:
			print("\033[91mCan't use binary cross entropy loss on more than 2 classes\033[0m")
			return False
		if self.X_val is not None or self.Y_val is not None:
			if self.X_val is None or self.Y_val is None:
				print("\033[91mMissing either validation input or output\033[0m")
				return False
			elif not isinstance(self.X_val, np.ndarray) and self.X_val.size != 0 \
				and self.X_val.shape[1] == self.X.shape[1]:
				print("\033[91mError in received validation input\033[0m")
				return False
			elif not isinstance(self.Y_val, np.ndarray) and self.Y_val.size != 0 \
				and self.Y_val.shape[1] == self.Y.shape[1]:
				print("\033[91mError in received validation output\033[0m")
				return False
		return True

	def _check_valid_layers(self) -> bool:
		''' Check validity of layers inside model. '''
		for layer in self.layers:
			if not isinstance(layer, DenseLayer):
				print("\033[91mSimpleNeuralNetwork must receive and list of DenseLayer\033[0m")
				return False
		if self.X.shape[0] != self.layers[0].weights.shape[1]:
			print("\033[91mError between input shape and first layer input shape\033[0m")
			return False
		elif self.Y.shape[0] != self.layers[-1].weights.shape[0]:
			print("\033[91mError between output shape and last layer output shape\033[0m")
			return False
		L = len(self.layers)
		for i, layer in enumerate(self.layers):
			if i != 0:
				if layer.weights.shape[1] != l_prev.weights.shape[0] :
					print("\033[91mError of compatibility between layers\033[0m")
					return False
			if i != L - 1 and layer.derivative is None:
				print("\033[91mFinal layer in non-final position\033[0m")
				return False
			l_prev = layer
		if l_prev.derivative is not None:
			print("\033[91mNon final layer in final position\033[0m")
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
			if regularization is not None:
				layer.lambda_ = self.lambda_
				layer.regularization = regularization
			layer.beta_1 = self.beta_1
			layer.beta_2 = self.beta_2
			layer.epsilon = self.epsilon
			layer.optimization = self.optimization

	def forward_propagation(self, val: bool = False) -> np.ndarray:
		''' Perform the forward propgation in model neural network. '''
		if val == False:
			self.infos.clear()
			layers_activations = [self.X]
			self.infos["A0"] = self.X
		else:
			layers_activations = [self.X_val]
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
		print("cst reg and wieghts reg", cst_reg, weights_reg)
		return cst_reg * weights_reg

	def binary_cross_entropy_loss(self, Y: np.ndarray = None, \
		Y_pred: np.ndarray = None) -> float:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y)
		for binary classification '''
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		val = False
		assert self.layers[L - 1].activation_name == 'sigmoid' \
			or self.layers[L - 1].activation_name == 'softmax'
		if Y is None or Y_pred is None:
			A_last = self.infos["A" + str(L)] # last activation value (= prediction)
			Y_pred = np.clip(np.argmax(A_last, axis=0), eps, 1. - eps)
			Y = np.argmax(self.Y, axis=0)
			val = True
		else:
			Y_pred = np.clip(np.argmax(Y_pred, axis=0), eps, 1. - eps)
		cost = -np.mean((1 - Y) * np.log(1 - Y_pred) + Y * np.log(Y_pred), axis=0)
		print("cost pre reg", cost)
		if self.lambda_ != 0:
			cost += self.regularization_cost()
		print(cost)
		cost_val = None
		if val == True and isinstance(self.X_val, np.ndarray) \
			and isinstance(self.Y_val, np.ndarray):
			A_last_val = self.forward_propagation(val=True)
			Y_pred_val = np.clip(np.argmax(A_last_val, axis=0), eps, 1. - eps)
			Y_val = np.argmax(self.Y_val, axis=0)
			cost_val = -np.mean((1 - Y_val) * np.log(1 - Y_pred_val) + Y_val * np.log(Y_pred_val), axis=0)
			if self.lambda_ != 0:
				cost_val += self.regularization_cost()
		return np.squeeze(cost), np.squeeze(cost_val)

	def cross_entropy_loss(self, Y: np.ndarray = None, \
		Y_pred: np.ndarray = None) -> float:
		''' Calculus of cost = loss (difference) on all	predicted
		(A_last for last layer activation) and expected outputs (Y)
		for multi-classes classification '''
		eps = 1e-15 # to avoid log(0)
		L = len(self.layers)
		val = False
		assert self.layers[L - 1].activation_name == 'softmax'
		if Y is None or Y_pred is None:
			A_last = self.infos["A" + str(L)] # last activation value (= prediction)
			Y_pred = np.clip(np.argmax(A_last, axis=0), eps, 1. - eps)
			Y = np.argmax(self.Y, axis=0)
			val = True
		m = Y.shape[1]
		cost = -np.sum(self.Y * np.log(Y_pred)) / m
		if self.lambda_ != 0:
			cost += self.regularization_cost()
		cost_val = None
		if val == True and isinstance(self.X_val, np.ndarray) and \
			isinstance(self.Y_val, np.ndarray):
			A_last_val = self.forward_propagation(val=True)
			cost_val = -np.sum(self.Y_val * np.log(np.clip(A_last_val, eps, 1. - eps))) / m
			if self.lambda_ != 0:
				cost_val += self.regularization_cost()
		return np.squeeze(cost), np.squeeze(cost_val)

	def backward_propagation(self) -> None:
		''' Perform the backward propgation in model neural network. '''
		L = len(self.layers)
		A_last = self.infos["A" + str(L)] # last activation value (= prediction)
		A_prev = self.infos["A" + str(L - 1)]
		dA_prev, dW, db = self.layers[-1].final_backward(A_last, self.Y, A_prev)
		self.infos["dW" + str(L)] = dW
		self.infos["db" + str(L)] = db
		for l in range(L - 1, 0, -1):
			A_prev = self.infos["A" + str(l - 1)]
			Z = self.infos["Z" + str(l)]
			dA_prev, dW, db = self.layers[l - 1].hidden_backward(dA_prev, \
				A_prev, Z)
			self.infos["dW" + str(l)] = dW
			self.infos["db" + str(l)] = db

	def update(self, time: int) -> None:
		''' Update parameters for each layer. '''
		for i, layer in enumerate(self.layers):
			dW = self.infos["dW" + str(i + 1)]
			db = self.infos["db" + str(i + 1)]
			layer.update_parameters(dW, db, time)

	def save_current_parameters(self) -> None:
		''' Save each layer current parameters. Useful for early stopping. '''
		for layer in self.layers:
			layer.save_parameters()

	def rewind_parameters(self) -> None:
		''' Rewind each layer to previous saved parameters. Useful for early stopping. '''
		for layer in self.layers:
			layer.rewind_saved_parameters()

	def save_model(self) -> None:
		''' Save model in a pickle file. '''
		with open(self.name + ".pkl", 'wb') as outp:
			pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

	def fit(self, nb_iterations: int = 10000) -> dict:
		''' Perform forward and backward propagation on a given number of epochs. '''
		assert nb_iterations > 0
		metrics = Metrics()
		L = len(self.layers)
		best_val_loss, best_val_epoch = None, None
		acc = []
		val_acc = []
		prec = []
		val_prec = []
		rec = []
		val_rec = []
		f1 = []
		val_f1 = []
		print("\n{} training:\n-------------".format(self.name))
		for i in range(nb_iterations):
			self.forward_propagation()
			train_loss, val_loss = self.loss()
			self.losses.append(train_loss)
			if val_loss != None:
				self.val_losses.append(val_loss)
			display = False
			if i == 0 or (i + 1) % 100 == 0 or i == nb_iterations - 1:
				display = True
			if display:
				print("epoch {}/{} - loss: {:.3f}".format(i + 1, nb_iterations, self.losses[i]), end='')
			metrics.set_values(np.argmax(self.Y, axis=0), np.argmax(self.infos["A" + str(L)], axis=0))
			if 'accuracy' in self.metrics:
				acc.append(metrics.accuracy())
				if display:
					print(" - acc: {:3.2f}".format(acc[i] * 100), end='')
			if 'precision' in self.metrics:
				prec.append(metrics.precision())
				if display:
					print(" - prec: {:3.2f}".format(prec[i] * 100), end='')
			if 'recall' in self.metrics:
				rec.append(metrics.recall())
				if display:
					print(" - rec: {:3.2f}".format(rec[i] * 100), end='')
			if 'f1' in self.metrics:
				f1.append(metrics.f1_score())
				if display:
					print(" - f1: {:3.2f}".format(f1[i] * 100), end='')
			if val_loss != None:
				if display:
					print(" - val_loss: {:.3f}".format(self.val_losses[i]), end='')
				A_last_val = self.forward_propagation(val=True)
				metrics.set_values(np.argmax(self.Y_val, axis=0), np.argmax(A_last_val, axis=0))
				if 'accuracy' in self.metrics:
					val_acc.append(metrics.accuracy())
					if display:
						print(" - val_acc: {:3.2f}".format(val_acc[i] * 100), end='')
				if 'precision' in self.metrics:
					val_prec.append(metrics.precision())
					if display:
						print(" - val_prec: {:3.2f}".format(val_prec[i] * 100), end='')
				if 'recall' in self.metrics:
					val_rec.append(metrics.recall())
					if display:
						print(" - val_rec: {:3.2f}".format(val_rec[i] * 100), end='')
				if 'f1' in self.metrics:
					val_f1.append(metrics.f1_score())
					if display:
						print(" - val_f1: {:3.2f}".format(val_f1[i] * 100), end='')
				if display:
					print(end='\n')
			if self.stop is not None:
				if val_loss is not None and (best_val_loss is None or round(best_val_loss, 6) > round(val_loss, 6)):
					best_val_loss, best_val_epoch = val_loss, i
					self.save_current_parameters()
				if best_val_epoch is not None and best_val_epoch < i - self.stop:
					print("\033[92mEarly stopping at epoch {} to get back to epoch {} (val_loss: {:6.6f}, best_val_loss {:6.6f})\033[0m".format(i + 1, best_val_epoch + 1, val_loss, best_val_loss))
					self.rewind_parameters()
					break
			self.backward_propagation()
			self.update(i + 1)
		print()
		if best_val_epoch is None:
			best_val_epoch = nb_iterations
		self.save_model()
		return {"loss": self.losses, "val_loss": self.val_losses, \
			"best_val_epoch": best_val_epoch, "acc": acc, "val_acc": val_acc, \
			"prec": prec, "val_prec": val_prec, "rec": rec, "val_rec": val_rec, \
			"f1": f1, "val_f1": val_f1} # historique

	def prediction(self, to_predict: np.ndarray, targets: np.ndarray) -> np.ndarray:
		''' Perform prediction of an array. '''
		layers_activations = [to_predict]
		for i, layer in enumerate(self.layers):
			A, _ = layer.forward(layers_activations[i])
			layers_activations.append(A)
		pred = np.argmax(A, axis=0)
		loss, _ = self.loss(pred, targets)
		return pred, loss
