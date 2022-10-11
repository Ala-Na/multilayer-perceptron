import numpy as np

Z = np.asarray([[2., -1., 3.], [1. , -5., -9.]])
Z[Z <= 0] = 0.01
print(Z)
