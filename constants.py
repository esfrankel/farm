import numpy as np

DATA_PATH = './pytorch-made/binarized_mnist.npz'

MNIST_ORDERINGS = [] # list of np arrays
MNIST_ORDERINGS.append(np.arange(784)) # top left going right
MNIST_ORDERINGS.append(np.arange(784).reshape(28, 28).T.flatten()) # top left going down
MNIST_ORDERINGS.append(np.arange(783, -1, -1)) # bottom right going left
MNIST_ORDERINGS.append(np.arange(783, -1, -1).reshape(28, 28).T.flatten()) # bottom right going up
MNIST_ORDERINGS.append(np.flip(np.arange(784).reshape(28, 28), 1).flatten()) # top right going left
MNIST_ORDERINGS.append(np.flip(np.arange(784).reshape(28, 28).T, 1).flatten()) # top right going down
MNIST_ORDERINGS.append(np.flip(np.arange(783, -1, -1).reshape(28,28), 1).flatten()) # bottom left going right
MNIST_ORDERINGS.append(np.flip(np.arange(783, -1, -1).reshape(28, 28).T, 1).flatten()) # bottom left going up

# TODO: add diagonal orderings? add random flood fill orderings?

# 0	1	3	6	10	15
# 2	4	7	11	16
# 5	8	12	17
# 9	13	18
# 14	19
# 20


# 0	1	2	3	...	27
# 28	29	30	31	...	55




# 0	1	3	6	10	15	2	4	7	11	16	21	

# o = [0]*784
# c = 0
# for i in range(55):
# 	for j in range(max(0, i-28), min(i, 28)):
# 		o[]