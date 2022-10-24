import numpy as np
import pandas as pd
from utils.data_manipulator import *
from utils.dense_layer import *
from utils.neural_network import *
import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Multilayer Perceptron - Program")
	parser.add_argument("--metrics", nargs="+", choices=['accuracy', 'precision', 'recall', 'f1'], type=str, help="Optional. Metrics to display other than loss. Available options: accuracy, precision, recall and f1.")
	parser.add_argument("--trainset_only", action='store_true', default=False, help="Optional. No creation of a validation set. Set to false by default.")
	parser.add_argument("--epochs", default=10000, type=int, help="Optional. Number of epochs to train our model on.")
	parser.add_argument("--dim", default=[64, 32, 16, 16, 8], type=int, nargs="+", help="Optional. Set denses dimensions. Note that denses have always relu activation, except for final dense having softmax activation.")
	parser.add_argument("--opti", choices=[None, 'rmsprop', 'adam', 'momentum'], default=None, type=str, help="Optional. Choice of optimization algorithm.")
	parser.add_argument("--alpha", type=float, default=0.001, help="Optional. Choice of learning rate alpha.")
	parser.add_argument("--name", type=str, default="Model", help="Optional. Model name to be saved.")
	parser.add_argument("--stop", type=int, default=None, help="Optional. Implement early stopping comparing validation set and training set losses, on stop value threshold (stop training when val_loss increase or is stable on stop value epochs). Set to None by default (no eraly stopping). Won't work if --trainset_only is activated.")
	parser.add_argument("--lambda_", type=float, default=1., help="Optional. Implement regularization on all layers by lambda value (must be between 0 and 1). Prevent overfitting.")
	args = parser.parse_args()

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
	if args.trainset_only == False:
		x_train, x_val, y_train, y_val = subsets_creator(datas, one_hot_labels, 0.8)
		x_train, fqrt, tqrt = scaler(x_train, 'robust')
		x_val, _, _ = scaler(x_val, 'robust', fqrt, tqrt)
		x_train, x_val, y_train, y_val = x_train.T, x_val.T, y_train.T, y_val.T
	else:
		x_train, y_train = united_shuffle(datas, one_hot_labels)
		x_train, fqrt, tqrt = scaler(x_train, 'robust')
		x_train, x_val, y_train, y_val = x_train.T, None, y_train.T, None

	# Create NN
	denses = []
	print("Quick model description\n-------------------")
	print("Learning rate = {} | Optimization algorithm = {} | Lamdba = {}\n".format(args.alpha, args.opti, args.lambda_))
	for i, dim in enumerate(args.dim):
		if i == 0:
			print("Layer 1:\ninput dim = 30 | output dim = {} | activation = relu".format(dim))
			denses.append(DenseLayer(30, dim, activation='relu', initialization='he'))
		else:
			print("Layer {}:\ninput dim = {} | output dim = {} | activation = relu".format(i + 1, args.dim[i - 1], dim))
			denses.append(DenseLayer(args.dim[i - 1], dim, activation='relu', initialization='he'))
	print("Layer {}:\ninput dim = {} | output dim = 2 | activation = softmax\n-------------------\n".format(len(args.dim) + 1, args.dim[-1]))
	denses.append(DenseLayer(args.dim[-1], 2, final=True, activation='softmax', initialization='xavier'))
	model = SimpleNeuralNetwork(denses, x_train, y_train, x_val, y_val, optimization=args.opti, alpha=args.alpha, metrics=args.metrics, stop=args.stop, lambda_=args.lambda_)
	history = model.fit(args.epochs)

	# Save model in .csv
	if os.path.isfile('experiments.csv'):
		df = pd.read_csv('experiments.csv')
	else :
		df = pd.DataFrame(columns=['Alpha', 'Opti', 'Lambda', 'Max epoch', \
		'Last epoch', 'Dims', 'Stop', 'Loss', 'Val_loss', 'Acc', 'Val_acc', \
		'Prec', 'Val_prec', 'F1', 'Val_f1'])
	new_df = pd.DataFrame.from_dict({'Alpha': args.alpha, 'Opti': args.opti, \
		'Lambda': args.lambda_, 'Max epoch': args.epochs, \
		'Last epoch': history["best_val_epoch"], 'Dims': args.dim, \
		'Stop': args.stop, 'Loss': history["loss"], \
		'Val_loss': history["val_loss"], 'Acc': history["acc"], \
		'Val_acc': history["val_acc"], 'Prec': history["prec"], \
		'Val_prec': history["val_prec"], 'F1': history["f1"], \
		'Val_f1': history["val_f1"]}, orient='index').transpose()
	full_df = pd.concat([df, new_df])
	full_df.to_csv('experiments.csv', index=False)
