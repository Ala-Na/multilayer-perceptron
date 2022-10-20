import numpy as np
import pandas as pd
from utils.data_manipulator import *
from utils.dense_layer import *
from utils.neural_network import *

if __name__ == "__main__":
	# First, retrieve labels nd data from .csv file
	df = pd.read_csv("data.csv", header=None, index_col=0)
	labels = df.iloc[:, 0]
	datas = df.iloc[:, 1:]
	if df.isnull().values.any():
		print("\033[91mNaN values in dataset : File corrupted\033[0m")
		exit(1)

	# Transform it in numpy array and replace M and B by numbers
	labels = labels.to_numpy(dtype=np.dtype(str))
	datas = datas.to_numpy(dtype=np.dtype(float))
	labels_nbr = np.select([labels =='B', labels == 'M'], [0 , 1], labels).astype(int)

	# Encode labels into one-hot vectors
	one_hot_labels = one_hot_class_encoder(labels_nbr, 2)

	# Cut into training/validation sets and scale x
	x_train, x_val, y_train, y_val = subsets_creator(datas, one_hot_labels, 0.8)
	x_train, fqrt, tqrt = scaler(x_train, 'robust')
	x_val, _, _ = scaler(x_val, 'robust', fqrt, tqrt)

	# Create NN
	dense1 = DenseLayer(30, 256, activation='relu', initialization='he')
	dense2 = DenseLayer(256, 128, activation='relu', initialization='he')
	dense3 = DenseLayer(128, 64, activation='relu', initialization='he')
	dense4 = DenseLayer(64, 32, activation='relu', initialization='he')
	dense5 = DenseLayer(32, 2, final=True, activation='softmax', initialization='he')
	model = SimpleNeuralNetwork([dense1, dense2, dense3, dense4, dense5], x_train, y_train, x_val, y_val, optimization=None, name="First try", alpha=0.0001)
	losses, val_losses = model.fit(5000)


