from utils.dense_layer import DenseLayer
from utils.neural_network import SimpleNeuralNetwork
import numpy as np

x_train = np.asarray([[0.06666667,0.76862745,0.32156863,0.00392157,0.03529412], [0.06666667,0.76862745,0.32156863,0.00392157,0.03529412],[0.12156863,0.75294118,0.27843137,0.08627451,0.03529412],[0.21960784,0.74509804,0.26666667,0.00784314,0.01960784]])
y_train = np.asarray([0, 0, 1, 1])
x_val = np.asarray([[0.06666667,0.76862745,0.32156863,0.00392157,0.03529413]])
y_val = np.asarray([0])

def one_hot_class_encoder(to_encode: np.ndarray, nb_classes: int) -> np.ndarray:
	''' Encode an array of classes from 0 to nb_classes into a one hot
	matrix. '''
	assert np.min(to_encode) >= 0 and np.max(to_encode) < nb_classes
	encoded = np.zeros((to_encode.shape[0], nb_classes))
	encoded[np.arange(to_encode.shape[0]), to_encode] = 1.0
	return encoded

if __name__ == "__main__":
	y_train = one_hot_class_encoder(y_train, 2)
	y_val = one_hot_class_encoder(y_val, 2)
	dense1 = DenseLayer(5, 6, activation='relu', initialization='he')
	dense2 = DenseLayer(6, 2, final=True, activation='softmax', initialization='he')
	model = SimpleNeuralNetwork([dense1, dense2], x_train, y_train, x_val, y_val, optimization=None, name="No opti")
	losses, val_losses = model.fit(3)

	dense1 = DenseLayer(5, 6, activation='relu', initialization='he')
	dense2 = DenseLayer(6, 2, final=True, activation='softmax', initialization='he')
	model = SimpleNeuralNetwork([dense1, dense2], x_train, y_train, x_val, y_val, optimization='momentum', name="Momentum")
	losses, val_losses = model.fit(3)

	dense1 = DenseLayer(5, 6, activation='relu', initialization='he')
	dense2 = DenseLayer(6, 2, final=True, activation='softmax', initialization='he')
	model = SimpleNeuralNetwork([dense1, dense2], x_train, y_train, x_val, y_val, optimization='rmsprop', name="RMSprop")
	losses, val_losses = model.fit(3)

	dense1 = DenseLayer(5, 6, activation='relu', initialization='he')
	dense2 = DenseLayer(6, 2, final=True, activation='softmax', initialization='he')
	model = SimpleNeuralNetwork([dense1, dense2], x_train, y_train, x_val, y_val, optimization='adam', name="Adam")
	losses, val_losses = model.fit(3)
