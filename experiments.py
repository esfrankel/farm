import torch
import torch.nn.functional as F
import numpy as numpy

from made import MADE
from inference import Inference
from config import *
from util import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import pickle


# load the dataset
mnist = np.load(DATA_PATH)
xtr, xte = mnist['train_data'], mnist['valid_data']
xtr = torch.from_numpy(xtr).cuda() # [50000, 784]
xte = torch.from_numpy(xte).cuda() # [10000, 784]


def occlusion_experiment_many(ckpts, case):
	I = {}
	keys = ckpts.keys()
	for k in keys:
		I[k] = Inference('made_'+ckpts[k][0], f'0{ckpts[k][1]}_params.pt')

	k = 10
	B = k**2

	nsteps = 100

	data = {}
	l = {}
	ll = {}

	for step in tqdm(range(nsteps)):
		xb = xte[step*B : step*B+B]
		occ = make_occlusion_mask(case).cuda()
		x = {}
		xll = {}
		for k in keys:
			x[k], xll[k] = I[k].fill_occlusion(xb, occ)
		t = {}
		for k in keys:
			t[k] = ((x[k]*occ - xb*occ)**2).sum().data.item() / (B * occ.sum().data.item())
		data[step] = (occ.cpu().numpy(), t, xll)
		for k in keys:
			if k not in l.keys():
				l[k] = t[k]
				ll[k] = xll[k]
			else:
				l[k] += t[k]
				ll[k] += xll[k]

	for k in keys:
		l[k] /= nsteps
		ll[k] /= nsteps

	s = '__'.join([ckpts[k][0] for k in keys])

	data[-1] = (l, ll)

	print(f"Results in {case} case:")
	for k in keys:
		print(f'Average occlusion reconstruction loss and NLL for {ckpts[k][0]}: ', l[k], -ll[k])

	f = open(f'./vis/exp_data__{s}.pkl', 'wb')
	pickle.dump(data, f)
	f.close()


def occlusion_experiment_pair(id1, id2, ckpt1=99, ckpt2=99, case='avg', plot=True):

	Path(f'./vis/occ__{id1}__{id2}__{case}').mkdir(exist_ok=False)

	I_1 = Inference('made_'+id1, f'0{ckpt1}_params.pt')
	I_m = Inference('made_'+id2, f'0{ckpt2}_params.pt')

	k = 5
	B = k**2 # each occlusion mask is tested over B images

	# nsteps = xte.shape[0] // B
	nsteps = 5
0
	data = {}
	l_1 = 0
	l_m = 0
	ll_1 = 0
	ll_m = 0

	# for step in tqdm(range(nsteps)):
	for step in range(nsteps):
		x = xte[step*B : step*B+B]
		occ = make_occlusion_mask(case).cuda()
		x_1, xll_1 = I_1.fill_occlusion(x, occ)
		x_m, xll_m = I_m.fill_occlusion(x, occ)
		t_1 = ((x_1*occ - x*occ)**2).sum().data.item() / (B * occ.sum().data.item())
		t_m = ((x_m*occ - x*occ)**2).sum().data.item() / (B * occ.sum().data.item())
		if t_1 > 1 or t_m > 1:
			import pdb; pdb.set_trace()
		# print("Batch ", step, occ.sum().data.item(), t_1, t_m)
		data[step] = (occ.cpu().numpy(), t_1, t_m, xll_1, xll_m)
		l_1 += t_1
		l_m += t_m
		ll_1 += xll_1
		ll_m += xll_m

		if plot:
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
			plt.savefig(f'./vis/occ__{id1}__{id2}__{case}/{step:03d}.png')


	l_1 /= nsteps
	l_m /= nsteps
	ll_1 /= nsteps
	ll_m /= nsteps

	data[-1] = (None, l_1, l_m, ll_1, ll_m)

	print(f'Results for {id1} vs {id2}:')
	print('Average occlusion reconstruction loss and LL for {id1}}: ', l_1, ll_1)
	print('Average occlusion reconstruction loss and LL for multiple orderings: ', l_m, ll_m)

	f = open(f'./vis/occ__{id1}__{id2}__{case}/exp_data.pkl', 'wb')
	pickle.dump(data, f)
	f.close()


if __name__ == '__main__':
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_2_ordering_1', 59, 59, 'avg', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_2_ordering_1', 59, 59, 'botright', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_2_ordering_1', 59, 59, 'topleft', False)

	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_4_ordering_1', 59, 59, 'avg', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_4_ordering_1', 59, 59, 'botright', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_4_ordering_1', 59, 59, 'topleft', False)

	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_8_ordering_1', 59, 59, 'avg', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_8_ordering_1', 59, 59, 'botright', False)
	# occlusion_experiment_pair('mnist_one_ordering_1', 'mnist_8_ordering_1', 59, 59, 'topleft', False)
	
	# ckpts = {
	# 	'1_normal': ('dec9_1_normal', 99),
	# 	'2_normal': ('dec9_2_normal', 99),
	# 	'4_normal': ('dec9_4_normal', 99),
	# 	'8_normal': ('dec9_8_normal', 99),
	# 	'1_multi': ('dec9_1_multi', 99),
	# 	'2_multi': ('dec9_2_multi', 99),
	# 	'4_multi': ('dec9_4_multi', 99),
	# 	'8_multi': ('dec9_8_multi', 99),
	# 	'1_kl': ('dec9_1_kl', 99),
	# 	'2_kl': ('dec9_2_kl', 99),
	# 	'4_kl': ('dec9_4_kl', 99),
	# 	'8_kl': ('dec9_8_kl', 99),
	# }
	# occlusion_experiment_many(ckpts, 'avg')
	# occlusion_experiment_many(ckpts, 'botright')
	# occlusion_experiment_many(ckpts, 'topleft')
	occlusion_experiment_pair('dec9_1_normal', 'dec9_8_normal', 99, 99, 'avg')



# python old_train.py -q=1000,1000 -o=1 -m=dec9_1_normal
# python old_train.py -q=1000,1000 -o=2 -m=dec9_2_normal
# python old_train.py -q=1000,1000 -o=4 -m=dec9_4_normal
# python old_train.py -q=1000,1000 -o=8 -m=dec9_8_normal
# python old_train.py -q=1000,1000 -s=1 -o=1 -m=dec9_1_multi
# python old_train.py -q=1000,1000 -s=2 -o=2 -m=dec9_2_multi
# python old_train.py -q=1000,1000 -s=4 -o=4 -m=dec9_4_multi
# python old_train.py -q=1000,1000 -s=8 -o=8 -m=dec9_8_multi
# python old_train.py -q=1000,1000 -s=1 -o=1 -k -m=dec9_1_kl
# python old_train.py -q=1000,1000 -s=2 -o=2 -k -m=dec9_2_kl
# python old_train.py -q=1000,1000 -s=4 -o=4 -k -m=dec9_4_kl
# python old_train.py -q=1000,1000 -s=8 -o=8 -k -m=dec9_8_kl
# python experiments.py

