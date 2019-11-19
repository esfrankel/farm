import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from constants import *
from util import *


class Inference:

	def __init__(self, exp, ckpt):

		self.model = torch.load(Path('./checkpoints') / exp / ckpt)
		self.model = self.model.cuda()
		self.model.eval()


	def sample_single_ordering(self, B):
		'''Autoregressively samples a batch of size B with a single ordering.'''

		flip_order = [i for _, i in sorted(zip(self.model.m[-1], list(range(784))))]

		x = torch.zeros((B, 784), dtype=torch.float32).cuda()
		for i in range(784):
			y = 0
			for j in range(self.model.num_masks): # average over all masks
				self.model.update_masks(resample_hidden_masks=True, resample_ordering=False)
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, flip_order[i]] = sample

		return x


	def fill_occlusion(self, inp, occ):
		'''
		Fill occlusion over all images in inp using the best possible ordering.

		inp (torch.Tensor): (B, 784) collection of images to run on.
		occ (torch.Tensor): (784,) occlusion mask.  1 means occluded, 0 means not.
		'''

		x = inp * (1 - occ)
		best_ordering = None
		best_score = -1
		for i in range(len(self.model.orderings)):
			ordering = self.model.orderings[i]
			flip_order = flip_ordering(ordering)
			count = 0
			while occ[flip_order[count]] == 0:
				count += 1
			# print(i, count)
			if count > best_score:
				# print("Hi!")
				best_score = count
				best_ordering = ordering
				best_i = i

		# print('Using ordering ', best_i)

		self.model.update_masks(use_ordering=best_ordering)

		flip_order = flip_ordering(best_ordering)
		for i in range(784):
			if occ[flip_order[i]] == 0:
				continue # we only want to sample if the pixel is occluded
			y = 0
			for j in range(self.model.num_masks): # average over all masks
				self.model.update_masks(resample_hidden_masks=True, resample_ordering=False)
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, flip_order[i]] = sample

		return x


if __name__ == '__main__':
	I = Inference('made_more_hiddens_8_orderings', '059_params.pt')
	# import pdb; pdb.set_trace()
	x = I.sample_single_ordering(16)
	vis = x.view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28)
	plt.imshow(vis.cpu().numpy(), cmap='gray')
	plt.show()