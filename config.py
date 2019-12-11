import numpy as np

DATA_PATH = './pytorch-made/binarized_mnist.npz'

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


MNIST_ORDERINGS = [] # list of np arrays
MNIST_ORDERINGS.append(np.arange(784)) # top left going right
MNIST_ORDERINGS.append(np.arange(784).reshape(28, 28).T.flatten()) # top left going down
MNIST_ORDERINGS.append(np.arange(783, -1, -1)) # bottom right going left
MNIST_ORDERINGS.append(np.arange(783, -1, -1).reshape(28, 28).T.flatten()) # bottom right going up
MNIST_ORDERINGS.append(np.flip(np.arange(784).reshape(28, 28), 1).flatten()) # top right going left
MNIST_ORDERINGS.append(np.flip(np.arange(784).reshape(28, 28).T, 1).flatten()) # top right going down
MNIST_ORDERINGS.append(np.flip(np.arange(783, -1, -1).reshape(28,28), 1).flatten()) # bottom left going right
MNIST_ORDERINGS.append(np.flip(np.arange(783, -1, -1).reshape(28, 28).T, 1).flatten()) # bottom left going up

CIFAR10_ORDERINGS = []
CIFAR10_ORDERINGS.append(np.arange(3072))


def set_config(spec):

	c = {}

	if spec.startswith('mnist_standard'):
		c = {
			'HIDDEN_LIST': [500, 500],				# MADE hidden layer sizes
			'NUM_MASKS': 1, 						# Number of random mask sets for connection-agnostic training
			'RESAMPLE_EVERY': 20, 					# For efficiency we can choose to resample orders/masks only once every this many steps
			'RUN_EVERY_ORDERING': False,			# Whether to run every ordering on each batch or only one ordering
			'MODEL_NAME': spec,						# Name of model for save location
			'DATASET': 'mnist', 					# Which dataset to use ('mnist' or 'cifar10')
			'MNIST_ORDERINGS': MNIST_ORDERINGS[:1]	# MNIST orderings
		}
	elif spec.startswith('mnist_23m_8o_reo'):
		c = {
			'HIDDEN_LIST': [500, 500],				# MADE hidden layer sizes
			'NUM_MASKS': 23, 						# Number of random mask sets for connection-agnostic training
			'RESAMPLE_EVERY': 20, 					# For efficiency we can choose to resample orders/masks only once every this many steps
			'RUN_EVERY_ORDERING': True,				# Whether to run every ordering on each batch or only one ordering
			'MODEL_NAME': spec,						# Name of model for save location
			'DATASET': 'mnist', 					# Which dataset to use ('mnist' or 'cifar10')
			'MNIST_ORDERINGS': MNIST_ORDERINGS[:8]	# MNIST orderings
		}
	elif spec.startswith('mnist_1k1k_5m_8o_reo'):
		c = {
			'HIDDEN_LIST': [1000, 1000],			# MADE hidden layer sizes
			'NUM_MASKS': 5, 						# Number of random mask sets for connection-agnostic training
			'RESAMPLE_EVERY': 20, 					# For efficiency we can choose to resample orders/masks only once every this many steps
			'RUN_EVERY_ORDERING': True,				# Whether to run every ordering on each batch or only one ordering
			'MODEL_NAME': spec,						# Name of model for save location
			'DATASET': 'mnist', 					# Which dataset to use ('mnist' or 'cifar10')
			'MNIST_ORDERINGS': MNIST_ORDERINGS[:8]	# MNIST orderings
		}
	elif spec.startswith('mnist_1k1k_5m_8o'):
		c = {
			'HIDDEN_LIST': [1000, 1000],			# MADE hidden layer sizes
			'NUM_MASKS': 5, 						# Number of random mask sets for connection-agnostic training
			'RESAMPLE_EVERY': 20, 					# For efficiency we can choose to resample orders/masks only once every this many steps
			'RUN_EVERY_ORDERING': False,				# Whether to run every ordering on each batch or only one ordering
			'MODEL_NAME': spec,						# Name of model for save location
			'DATASET': 'mnist', 					# Which dataset to use ('mnist' or 'cifar10')
			'MNIST_ORDERINGS': MNIST_ORDERINGS[:8]	# MNIST orderings
		}
	elif spec.startswith('mnist_1k1k_1m_8o'):
		c = {
			'HIDDEN_LIST': [1000, 1000],			# MADE hidden layer sizes
			'NUM_MASKS': 1, 						# Number of random mask sets for connection-agnostic training
			'RESAMPLE_EVERY': 20, 					# For efficiency we can choose to resample orders/masks only once every this many steps
			'RUN_EVERY_ORDERING': False,				# Whether to run every ordering on each batch or only one ordering
			'MODEL_NAME': spec,						# Name of model for save location
			'DATASET': 'mnist', 					# Which dataset to use ('mnist' or 'cifar10')
			'MNIST_ORDERINGS': MNIST_ORDERINGS[:8]	# MNIST orderings
		}

	return c
