import torch
import torch.nn.functional as F
import numpy as numpy

from made import MADE
from inference import Inference
from constants import *
from util import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import pickle

from time import time

# load the dataset
mnist = np.load(DATA_PATH)
xtr, xte = mnist['train_data'], mnist['valid_data']
xtr = torch.from_numpy(xtr).cuda() # [50000, 784]
xte = torch.from_numpy(xte).cuda() # [10000, 784]


def occlusion_experiment():

	I_1 = Inference('made_one_ordering_1', '059_params.pt')
	I_m = Inference('made_many_ordering_1', '059_params.pt')

	k = 5
	B = k**2 # each occlusion mask is tested over B images

	# nsteps = xte.shape[0] // B
	nsteps = 50

	data = {}
	l_1 = 0
	l_m = 0

	for step in tqdm(range(nsteps)):
		x = xte[step*B : step*B+B]
		occ = make_occlusion_mask().cuda()
		x_1 = I_1.fill_occlusion(x, occ)
		x_m = I_m.fill_occlusion(x, occ)
		t_1 = F.mse_loss(x_1, x)
		t_m = F.mse_loss(x_m, x)
		t_1 = t_1.data.item()
		t_m = t_m.data.item()
		print("Batch ", step, t_1, t_m)
		data[step] = (occ.cpu().numpy(), t_1, t_m)
		l_1 += t_1
		l_m += t_m

		x_ = ((1 - (occ * 0.5)) * (x - 0.5)) + 0.5 # highlight the occluded part
		x_1_ = ((1 - (occ * 0.5)) * (x_1 - 0.5)) + 0.5
		x_m_ = ((1 - (occ * 0.5)) * (x_m - 0.5)) + 0.5
		vis = [
			x_.view(k, 28*k, 28).transpose(0,1).reshape(k*28, k*28).cpu(),
			torch.ones(k*28, 28),
			x_1_.view(k, 28*k, 28).transpose(0,1).reshape(k*28, k*28).cpu(),
			torch.ones(k*28, 28),
			x_m_.view(k, 28*k, 28).transpose(0,1).reshape(k*28, k*28).cpu()
		]
		vis = torch.cat(vis, 1).numpy()
		plt.imshow(vis, cmap='gray')
		plt.savefig(f'./vis/flush_botright/{step:03d}.png')


	l_1 /= nsteps
	l_m /= nsteps

	print('Average occlusion reconstruction loss for one ordering: ', l_1)
	print('Average occlusion reconstruction loss for multiple orderings: ', l_m)

	f = open('exp_data_flush_botright.pkl', 'wb')
	pickle.dump(data, f)
	f.close()

def generate_experiment():
	I_1 = Inference('made_eric_run', '059_params.pt')
	I_m = Inference('made_eric_run_many', '059_params.pt')

	curr_time = time.now()
	I_1.create_image_multiple_orderings()
	print("Time taken is {}".format(time.now() - curr_time))

	curr_time = time.now()
	I_m.create_image_multiple_orderings()
	print("Time taken is {}".format(time.now() - curr_time))


if __name__ == '__main__':
	occlusion_experiment()
