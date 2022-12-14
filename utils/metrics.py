import numpy as np

class Metrics():

	def __init__(self, expected: np.ndarray or None = None, \
		predicted: np.ndarray or None = None) -> None:
		assert (isinstance(expected, np.ndarray) \
			and isinstance(predicted, np.ndarray) \
			and expected.size != 0 and expected.shape == predicted.shape) \
			or (expected is None and predicted is None)
		self.y = expected
		self.y_hat = predicted

	def set_values(self, expected: np.ndarray, predicted: np.ndarray) -> None:
		''' Set expected and predicted values. '''
		assert isinstance(expected, np.ndarray) \
			and isinstance(predicted, np.ndarray) \
			and expected.size != 0 \
			and expected.shape == predicted.shape
		self.y = expected
		self.y_hat = predicted


	def set_expected(self, expected: np.ndarray) -> None:
		''' Set only expected values. '''
		assert isinstance(expected, np.ndarray) \
			and expected.size != 0 \
			and expected.shape == self.y.shape
		self.y = expected

	def set_predicted(self, predicted: np.ndarray) -> None:
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
		div = np.clip(truePos + falsePos, 1e-15, None)
		return truePos / (div)

	def recall(self, pos_label=1) -> float:
		'''Compute the recall score. '''
		assert isinstance(pos_label, int) or isinstance(pos_label, str)
		truePos = np.sum((self.y == pos_label) & (self.y_hat == pos_label))
		falseNeg = np.sum((self.y == pos_label) & (self.y_hat != pos_label))
		div = np.clip(truePos + falseNeg, 1e-15, None)
		return truePos / (div)

	def f1_score(self, pos_label=1):
		'''Compute the F1 score. '''
		assert isinstance(pos_label, int) or isinstance(pos_label, str)
		precision = self.precision(pos_label)
		recall = self.recall(pos_label)
		div = np.clip(precision + recall, 1e-15, None)
		return (2 * precision * recall) / (div)

