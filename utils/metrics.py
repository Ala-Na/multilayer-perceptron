import numpy as np

class Metrics():

	def __init__(self, expected: np.darray, predicted: np.ndarray) -> None:
		assert isinstance(expected, np.ndarray) \
			and isinstance(predicted, np.ndarray) \
			and expected.size != 0 \
			and expected.shape == predicted.shape
		self.y = expected
		self.y_hat = predicted

	def set_values(self, expected: np.darray, predicted: np.ndarray) -> None:
		''' Set expected and predicted values. '''
		assert isinstance(expected, np.ndarray) \
			and isinstance(predicted, np.ndarray) \
			and expected.size != 0 \
			and expected.shape == predicted.shape
		self.y = expected
		self.y_hat = predicted

	def set_expected(self, expected: np.darray) -> None:
		''' Set only expected values. '''
		assert isinstance(expected, np.ndarray) \
			and expected.size != 0 \
			and expected.shape == self.y.shape
		self.y = expected

	def set_predicted(self, predicted: np.darray) -> None:
		''' Set only predicted values. '''
		assert isinstance(predicted, np.ndarray) \
			and predicted.size != 0 \
			and predicted.shape == self.y_hat.shape
		self.y_hat = predicted

	def accuracy(self) -> float:
		''' Compute the accuracy score. '''
		return np.sum(self.y == self.y_hat) / self.y.size

	def precision(self, pos_label=1) -> float:
		'''Compute the precision score. '''
		assert isinstance(pos_label, int) or isinstance(pos_label, str)
		truePos = np.sum((self.y == pos_label) & (self.y_hat == pos_label))
		falsePos = np.sum((self.y != pos_label) & (self.y_hat == pos_label))
		return truePos / (truePos + falsePos)

	def recall(self, pos_label=1) -> float:
		'''Compute the recall score. '''
		assert isinstance(pos_label, int) or isinstance(pos_label, str)
		truePos = np.sum((self.y == pos_label) & (self.y_hat == pos_label))
		falseNeg = np.sum((self.y == pos_label) & (self.y_hat != pos_label))
		return truePos / (truePos + falseNeg)

	def f1_score_(self, pos_label=1):
		'''Compute the F1 score. '''
		assert isinstance(pos_label, int) or isinstance(pos_label, str)
		precision = self.precision(pos_label)
		recall = self.recall(pos_label)
		return 2 * precision * recall / (precision + recall)
