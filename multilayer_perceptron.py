from re import M
import numpy as np
import pandas as pd
from utils.data_manipulator import *
from utils.dense_layer import *
from utils.neural_network import *
from utils.metrics import *
import argparse
import os
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Multilayer Perceptron - Program")
	parser.add_argument("--seed", type=int, help="Optional. Set seed for random draw.")
	parser.add_argument("--metrics", nargs="+", default=[], choices=['accuracy', 'precision', 'recall', 'f1'], type=str, help="Optional. Metrics to display other than loss. Available options: accuracy, precision, recall and f1.")
	parser.add_argument("--trainset_only", action='store_true', default=False, help="Optional. No creation of a validation set. Set to false by default.")
	parser.add_argument("--epochs", default=10000, type=int, help="Optional. Number of epochs to train our model on.")
	parser.add_argument("--dim", default=[64, 32, 16, 16, 8], type=int, nargs="+", help="Optional. Set denses dimensions. Note that denses have always relu activation, except for final dense having softmax activation.")
	parser.add_argument("--opti", choices=[None, 'rmsprop', 'adam', 'momentum'], default=None, type=str, help="Optional. Choice of optimization algorithm.")
	parser.add_argument("--alpha", type=float, default=0.001, help="Optional. Choice of learning rate alpha.")
	parser.add_argument("--name", type=str, default="Model", help="Optional. File model name when saved or file name to open.")
	parser.add_argument("--stop", type=int, default=None, help="Optional. Implement early stopping comparing validation set and training set losses, on stop value threshold (stop training when val_loss increase or is stable on stop value epochs). Set to None by default (no eraly stopping). Won't work if --trainset_only is activated.")
	parser.add_argument("--lambda_", type=float, default=1., help="Optional. Implement regularization on all layers by lambda value (must be between 0 and 1). Prevent overfitting.")
	parser.add_argument("--show_all", action='store_true', default=False, help="Optional. Show all models saved in experiments.csv on a graph. Set to false by default.")
	parser.add_argument("--show_only", action='store_true', default=False, help="Optional. Only show models saved in experiments.csv on a graph. No training performed. Set to false by default.")
	parser.add_argument("--predict", type=str, default=None, help="Optional. Set program in prediction mode. A dataset's path to perform a prediction on must be given. Use --name option to set model to use.")
	args = parser.parse_args()

	if args.trainset_only is True and (args.stop != None or args.show_only is False):
		print("\033[93mOptions --trainset_only and --stop | --show_only are not compatible.\033[0m")
		exit(1)
	elif args.show_all is True and args.show_only is True:
		print("\033[93mOptions --show_all and --show_only are not compatible.\033[0m")
		exit(1)
	elif args.predict != None and (len(args.metrics) > 0 \
		or args.trainset_only is True or args.epochs is True or args.opti != None \
		or args.stop != None or args.lambda_ != 1. or args.show_all is True \
		or args.show_only is True):
		print("\033[93mOptions --predict is only compatible with --name option.\033[0m")
		exit(1)

	if args.predict is None and args.show_only == False:
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
			x_train, x_val, y_train, y_val = subsets_creator(datas, one_hot_labels, 0.8, args.seed)
			x_train, fqrt, tqrt = scaler(x_train, 'robust')
			x_val, _, _ = scaler(x_val, 'robust', fqrt, tqrt)
			x_train, x_val, y_train, y_val = x_train.T, x_val.T, y_train.T, y_val.T
		else:
			x_train, y_train = united_shuffle(datas, one_hot_labels, args.seed)
			x_train, fqrt, tqrt = scaler(x_train, 'robust')
			x_train, x_val, y_train, y_val = x_train.T, None, y_train.T, None

		# Create NN
		denses = []
		print("\nQuick model description\n-------------------")
		print("Learning rate = {} | Optimization algorithm = {} | Lambda = {}\n".format(args.alpha, args.opti, args.lambda_))
		for i, dim in enumerate(args.dim):
			if i == 0:
				print("Layer 1:\ninput dim = 30 | output dim = {} | activation = relu".format(dim))
				denses.append(DenseLayer(30, dim, activation='relu', initialization='he', seed= args.seed))
			else:
				print("Layer {}:\ninput dim = {} | output dim = {} | activation = relu".format(i + 1, args.dim[i - 1], dim))
				denses.append(DenseLayer(args.dim[i - 1], dim, activation='relu', initialization='he', seed= args.seed))
		print("Layer {}:\ninput dim = {} | output dim = 2 | activation = softmax\n-------------------\n".format(len(args.dim) + 1, args.dim[-1]))
		denses.append(DenseLayer(args.dim[-1], 2, final=True, activation='softmax', initialization='xavier', seed= args.seed))
		model = SimpleNeuralNetwork(denses, x_train, y_train, x_val, y_val, name=args.name, optimization=args.opti, alpha=args.alpha, metrics=args.metrics, stop=args.stop, lambda_=args.lambda_)
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
			'Val_prec': history["val_prec"], 'Rec': history["rec"], \
			'Val_rec': history["val_rec"], 'F1': history["f1"], \
			'Val_f1': history["val_f1"]}, orient='index').transpose()
		full_df = pd.concat([df, new_df])
		full_df.to_csv('experiments.csv', index=False)

		# Show current model on a graph
		if len(args.metrics) > 0:
			fig, axs = plt.subplots(2, figsize=(20, 20))
			axs[1].set_xlabel("Epochs")
			axs[0].set_xlabel("Epochs")
			axs[0].plot(history["loss"], label="Loss", color="royalblue")
			if len(history["val_loss"]) > 0:
				axs[0].plot(history["val_loss"], label="Val loss", color="lightcoral")
		else:
			plt.plot(history["loss"], label="Loss", color="royalblue")
			plt.title("Loss evolution on newly trained model")
			if len(history["val_loss"]) > 0:
				plt.plot(history["val_loss"], label="Val loss", color="lightcoral")
			plt.legend()
		if len(history["acc"]) > 0:
			axs[1].plot(history["acc"], label="Accuracy", color="limegreen", linestyle='dashdot')
		if len(history["val_acc"]) > 0:
			axs[1].plot(history["val_acc"], label="Val accuracy", color="orange", linestyle='dashdot')
		if len(history["prec"]) > 0:
			axs[1].plot(history["prec"], label="Precision", color="limegreen", linestyle='dashed')
		if len(history["val_prec"]) > 0:
			axs[1].plot(history["val_prec"], label="Val precision", color="orange", linestyle='dashed')
		if len(history["rec"]) > 0:
			axs[1].plot(history["rec"], label="Recall", color="limegreen", linestyle='dotted')
		if len(history["val_rec"]) > 0:
			axs[1].plot(history["val_rec"], label="Val recall", color="orange", linestyle='dotted')
		if len(history["f1"]) > 0:
			axs[1].plot(history["f1"], label="F1 score", color="limegreen", linestyle='solid')
		if len(history["val_f1"]) > 0:
			axs[1].plot(history["val_f1"], label="Val f1 score", color="orange", linestyle='solid')
		if len(args.metrics) > 0:
			fig.legend()
			fig.suptitle("Metrics evolution on newly trained model")
		plt.show()

	elif args.show_only is True:
		try:
			full_df = pd.read_csv('experiments.csv')
		except:
			print("\033[93mCan't open experiments.csv\033[0m")
			exit(1)

	# Show models on graph
	if args.show_all == True or args.show_only == True:
		color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
			for i in range(len(full_df))]
		legend_lines = [Line2D([0], [0], color=color[i], linestyle='solid') for i in range(len(full_df))]
		legend_title = ["Model " + str(i + 1) for i in range(len(full_df))]

		fig, axs = plt.subplots(2, figsize=(20, 20))
		axs[1].set_xlabel("Epochs")
		axs[0].set_xlabel("Epochs")
		legend0 = [Line2D([0], [0], color='black', lw=6, alpha=0.5, linestyle='solid'), \
			Line2D([0], [0], color='black', lw=1, linestyle='solid')]
		legend1 = [Line2D([0], [0], color='grey', lw=6, alpha=0.5, linestyle='solid'), \
			Line2D([0], [0], color='grey', lw=1, linestyle='solid'), \
			Line2D([0], [0], color='black', lw=1, linestyle='dashdot'), \
			Line2D([0], [0], color='black', lw=1, linestyle='dashed'), \
			Line2D([0], [0], color='black', lw=1, linestyle='dotted'), \
			Line2D([0], [0], color='black', lw=1, linestyle='solid')]

		for i in range(len(full_df)):
			if type(full_df.iloc[i]['Loss']) == str:
				array = full_df.iloc[i]['Loss'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Loss']
			axs[0].plot(array, label="Loss", color=color[i], linewidth=6, alpha=0.5)
			if type(full_df.iloc[i]['Val_loss']) == str:
				array = full_df.iloc[i]['Val_loss'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Val_loss']
			axs[0].plot(array, label="Val loss", color=color[i])
			if type(full_df.iloc[i]['Acc']) == str:
				array = full_df.iloc[i]['Acc'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Acc']
			axs[1].plot(array, label="Accuracy", color=color[i], linestyle='dashdot', linewidth=6, alpha=0.5)
			if type(full_df.iloc[i]['Val_acc']) == str:
				array = full_df.iloc[i]['Val_acc'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Val_acc']
			axs[1].plot(array, label="Val acc", color=color[i], linestyle='dashdot')
			if type(full_df.iloc[i]['Prec']) == str:
				array = full_df.iloc[i]['Prec'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Prec']
			axs[1].plot(array, label="Prec", color=color[i], linestyle='dashed', linewidth=6, alpha=0.5)
			if type(full_df.iloc[i]['Val_prec']) == str:
				array = full_df.iloc[i]['Val_prec'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Val_prec']
			axs[1].plot(array, label="Val prec", color=color[i], linestyle='dashed')
			if type(full_df.iloc[i]['Val_prec']) == str:
				array = full_df.iloc[i]['Val_prec'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Rec']
			axs[1].plot(array, label="Rec", color=color[i], linestyle='dotted', linewidth=6, alpha=0.5)
			if type(full_df.iloc[i]['Val_rec']) == str:
				array = full_df.iloc[i]['Val_rec'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Val_rec']
			axs[1].plot(array, label="Val rec", color=color[i], linestyle='dotted')
			if type(full_df.iloc[i]['F1']) == str:
				array = full_df.iloc[i]['F1'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['F1']
			axs[1].plot(array, label="F1", color=color[i], linestyle='solid', linewidth=6, alpha=0.5)
			if type(full_df.iloc[i]['Val_f1']) == str:
				array = full_df.iloc[i]['Val_f1'][1:-2].split(", ")
				while '' in array:
					array.remove('')
				if len(array) > 0:
					array = [float(nb) for nb in array]
			else:
				array = full_df.iloc[i]['Val_f1']
			axs[1].plot(array, label="Val F1", color=color[i], linestyle='solid')

		axs[0].legend(legend0, ['Train loss', 'Val loss'])
		axs[1].legend(legend1, ['Train', 'Val', 'Accuracy', 'Precision', 'Recall', 'F1 score'])
		fig.suptitle("Metrics evolution on all trained models")
		fig.legend(legend_lines, legend_title)
		plt.show()

	if args.predict != None:
		# Open file with dataset
		# NOTE : No indication of dataset format, so I suppose it's same as initial dataset
		try:
			df = pd.read_csv(args.predict, header=None, index_col=0)
		except:
			print("\033[93mCan't open {} file\033[0m".format(args.predict))
			exit(1)
		labels = df.iloc[:, 0]
		datas = df.iloc[:, 1:]
		if df.isnull().values.any():
			print("\033[91mNaN values in dataset : File corrupted\033[0m")
			exit(1)

		# Transform it in numpy array and replace M and B by numbers
		labels = labels.to_numpy(dtype=np.dtype(str))
		datas = datas.to_numpy(dtype=np.dtype(float))
		datas, _, _ = scaler(datas, 'robust')
		labels_nbr = np.select([labels =='B', labels == 'M'], [0 , 1], labels).astype(int)

		# Retrieve model from pickle file
		try:
			model = None
			with open(args.name, 'rb') as inp:
				model = pickle.load(inp)
		except:
			print("\033[93mCan't retrieve model object in {} file\033[0m".format(args.name))
			exit(1)
		prediction = model.prediction(datas.T)
		prediction_labels = np.select([prediction == 0, prediction == 1], ['B' , 'M'], prediction).astype(str)

		print('Prediction for {} elements:\n'.format(len(prediction_labels)), prediction_labels, '\n')
		metrics = Metrics(labels_nbr, prediction)
		print('=> Accuracy: {:5.2f} | Precision: {:5.2f} | Recall: {:5.2f} | F1 score: {:5.2f}'.format(metrics.accuracy() * 100, metrics.precision() * 100, \
			metrics.recall() * 100, metrics.f1_score() * 100))
