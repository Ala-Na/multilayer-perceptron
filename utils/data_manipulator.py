from unittest.mock import NonCallableMagicMock
import numpy as np
from typing import Tuple

def united_shuffle(x: np.ndarray, y: np.ndarray, seed: int or None = None):
	''' Shuffles randomly x and y sets (datas and labels) while keeping their
	correspondence. '''
	if seed != None:
		np.random.seed(seed)
	p = np.random.permutation(len(x))
	return x[p], y[p]

def subsets_creator(x:np.ndarray, y:np.ndarray, proportion: float, seed: int or None = None):
	''' Shuffles and splits the dataset (given by x and y) into a training and
	a test set, while respecting the given proportion of examples to be kept in
	the	training set.
	Returns x_train, x_test, y_train, y_test. '''
	assert isinstance(proportion, float) and proportion > 0.0 and proportion < 1.0, "\033[93mError in proportion for subsets creation.\033[0m"
	ind_split = (int)(x.shape[0] * proportion)
	x, y = united_shuffle(x, y, seed)
	return (x[:ind_split, :], x[ind_split:, :], y[:ind_split, :], y[ind_split:, :])

def one_hot_class_encoder(to_encode: np.ndarray, nb_classes: int) -> np.ndarray:
	''' Encode an array of classes from 0 to nb_classes into a one hot
	matrix. '''
	assert np.min(to_encode) >= 0 and np.max(to_encode) < nb_classes
	encoded = np.zeros((to_encode.shape[0], nb_classes))
	encoded[np.arange(to_encode.shape[0]), to_encode] = 1.0
	return encoded


def scaler(x: np.ndarray, option: str ='mean_normalization', \
		lst1: np.ndarray or None = None, lst2: np.ndarray or None = None) \
		-> Tuple[np.ndarray, list, list] or None:
	available_options = ['mean_normalization', 'min_max', 'robust']
	assert option in available_options, "\033[93mScaler option not recognized.\033[0m"
	if lst1 is not None and lst2 is not None:
		assert (lst1.shape[0] == x.shape[1]), "\033[93mError in shapes.\033[0m"
		assert (lst2.shape[0] == x.shape[1]), "\033[93mError in shapes.\033[0m"
	if option == 'mean_normalization':
		if lst1 is None or lst2 is None:
			lst1 = np.mean(x, axis=0) # means
			lst2 = np.std(x, axis=0) # stds
		x = (x - lst1) / lst2
		return x, lst1, lst2 # scaled array, means, stds
	elif option == 'min_max':
		if lst1 is None or lst2 is None:
			lst1 = np.amin(x, axis=0) # mins
			lst2 = np.amax(x, axis=0) # maxs
		print(x)
		print(lst2, lst1)
		x = (x - lst1) / (lst2 - lst1)
		return x, lst1, lst2 # scaled array, mins, maxs
	elif option == 'robust':
		if lst1 is None or lst2 is None:
			lst1 = np.percentile(x, 25, axis=0) # first quartil
			lst2 = np.percentile(x, 75, axis=0) # third quartil
		x = (x - lst1) / (lst2 - lst1)
		return x, lst1, lst2 # scaled array, first qrt, third
