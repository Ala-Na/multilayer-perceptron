import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
	''' Compute the softmax function '''
	return np.exp(x) / np.sum(np.exp(x))

