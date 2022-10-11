from dense import DenseLayer
from model import Model
import numpy as np

if __name__ == "__main__":
	dense1 = DenseLayer(#shape x, shape a1) -> give a1
	dense2 = DenseLayer(#shape a1, shape a2) -> give a2
	dense3 = DenseLayer(#shape a2, shape output) -> give output
	model = Model(#shape x, shape output, [dense1, dense2, dense3])
	return
