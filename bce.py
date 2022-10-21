import numpy as np

def binary_cross_entropy_loss(Y, Y_pred) -> float:
	''' Calculus of cost = loss (difference) on all	predicted
	(A_last for last layer activation) and expected outputs (Y)
	for binary classification '''
	eps = 1e-7 # to avoid log(0)
	m = Y.shape[0]
	print(Y.shape)
	Y_pred = np.clip(Y_pred, eps, 1. - eps)
	term_0 = (1-Y) * np.log(1-Y_pred + 1e-7)
	term_1 = Y * np.log(Y_pred + 1e-7)
	return -np.mean(term_0+term_1, axis=0)

print(binary_cross_entropy_loss(np.array([1, 1, 1]).reshape(-1, 1),
                         np.array([1, 1, 0]).reshape(-1, 1)))
