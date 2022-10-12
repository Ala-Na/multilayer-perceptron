import numpy as np

class ANN():
	def __init__(self, layers_size):
		self.layers_size = layers_size
		self.parameters = {}
		self.L = len(self.layers_size)
		self.n = 0
		self.costs = []

	def sigmoid(self, Z):
		return 1 / (1 + np.exp(-Z))

	def softmax(self, Z):
		expZ = np.exp(Z - np.max(Z))
		return expZ / expZ.sum(axis=0, keepdims=True)

	def initialize_parameters(self):
		np.random.seed(1)

		for l in range(1, len(self.layers_size)):
			self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
				self.layers_size[l - 1])
			self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

	def forward(self, X):
		store = {}

		A = X.T
		for l in range(self.L - 1):
			Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
			A = self.sigmoid(Z)
			store["A" + str(l + 1)] = A
			store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
			store["Z" + str(l + 1)] = Z

		Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
		A = self.softmax(Z)
		store["A" + str(self.L)] = A
		store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
		store["Z" + str(self.L)] = Z

		return A, store

	def sigmoid_derivative(self, Z):
		s = 1 / (1 + np.exp(-Z))
		return s * (1 - s)

	def backward(self, X, Y, store):

		derivatives = {}

		store["A0"] = X.T

		A = store["A" + str(self.L)]
		dZ = A - Y.T

		dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
		db = np.sum(dZ, axis=1, keepdims=True) / self.n
		dAPrev = store["W" + str(self.L)].T.dot(dZ)

		derivatives["dW" + str(self.L)] = dW
		derivatives["db" + str(self.L)] = db

		for l in range(self.L - 1, 0, -1):
			dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
			dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
			db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
			if l > 1:
				dAPrev = store["W" + str(l)].T.dot(dZ)

			derivatives["dW" + str(l)] = dW
			derivatives["db" + str(l)] = db

		return derivatives

	def fit(self, X, Y, learning_rate=0.01, n_iterations=2500):
		np.random.seed(1)

		self.n = X.shape[0]

		self.layers_size.insert(0, X.shape[1])

		self.initialize_parameters()
		for loop in range(n_iterations):
			A, store = self.forward(X)
			cost = -np.mean(Y * np.log(A.T+ 1e-8))
			derivatives = self.backward(X, Y, store)

			for l in range(1, self.L + 1):
				self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
					"dW" + str(l)]
				self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
					"db" + str(l)]


			# if loop % 100 == 0:
			print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

			if loop % 10 == 0:
				self.costs.append(cost)

	def predict(self, X, Y):
		A, cache = self.forward(X)
		y_hat = np.argmax(A, axis=0)
		Y = np.argmax(Y, axis=1)
		accuracy = (y_hat == Y).mean()
		return accuracy * 100


train_x = np.asarray([[0.06666667,0.76862745,0.32156863,0.00392157,0.03529412], [0.06666667,0.76862745,0.32156863,0.00392157,0.03529412],[0.12156863,0.75294118,0.27843137,0.08627451,0.03529412],[0.21960784,0.74509804,0.26666667,0.00784314,0.01960784]])
train_y = np.asarray([0, 0, 1, 2])

def one_hot_class_encoder(to_encode: np.ndarray, nb_classes: int) -> np.ndarray:
	''' Encode an array of classes from 0 to nb_classes into a one hot
	matrix. '''
	assert np.min(to_encode) >= 0 and np.max(to_encode) < nb_classes
	encoded = np.zeros((to_encode.shape[0], nb_classes))
	encoded[np.arange(to_encode.shape[0]), to_encode] = 1.0
	return encoded

if __name__ == '__main__':
	train_y = one_hot_class_encoder(train_y, 3)
	layers_dims = [6, 3]

	ann = ANN(layers_dims)
	ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=3)
	print("Train Accuracy:", ann.predict(train_x, train_y))
