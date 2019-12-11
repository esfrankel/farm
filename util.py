import numpy as np
import random
import torch
from six.moves import cPickle as pickle

def flip_ordering(o):
	return np.array([i for _, i in sorted(zip(o, list(range(o.size))))])

def flip_orderings(O):
	return [flip_ordering(o) for o in O]

def make_occlusion_mask(case='avg'):
	mask = np.zeros((28,28))
	a = random.randint(0,27)
	b = random.randint(0,27)
	if case == 'botright':
		b = 27
	elif case == 'topleft':
		a = 0
	i1 = min(a,b)
	i2 = max(a,b)+1
	a = random.randint(0,27)
	b = random.randint(0,27)
	if case == 'botright':
		b = 27
	elif case == 'topleft':
		a = 0
	j1 = min(a,b)
	j2 = max(a,b)+1
	mask[i1:i2, j1:j2] = 1
	# print('Occlusion: ', (i1, i2, j1, j2))
	return torch.Tensor(mask.flatten())

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di