import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from config import DATA_PATH
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
				self.model.update_masks()
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, flip_order[i]] = sample

		return x


	def sample_single_mask(self, B):
		'''Autoregressively samples a batch of size B with a single ordering.'''

		flip_order = [i for _, i in sorted(zip(self.model.m[-1], list(range(784))))]

		x = torch.zeros((B, 784), dtype=torch.float32).cuda()
		for i in range(784):
			y = self.model(x)
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

		# self.model.update_ordering(use_ordering=best_ordering)
		self.model.update_masks(use_ordering=best_ordering) # TODO: REVERT CHANGE

		flip_order = flip_ordering(best_ordering)
		ll = 0
		sampled_pixels = 0
		for i in range(784):
			if occ[flip_order[i]] == 0:
				continue # we only want to sample if the pixel is occluded
			y = 0
			for j in range(self.model.num_masks): # average over all masks
				# self.model.update_masks()
				self.model.update_masks(resample_hidden_masks=True, resample_ordering=False) # TODO: REVERT CHANGE
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			p = torch.sigmoid(logits)
			B = Bernoulli(p)
			sample = B.sample()
			x[:, flip_order[i]] = sample
			q = 1-p
			p += (p <= 0) * 1e-8
			q += (q <= 0) * 1e-8
			pixel_ll = (q.log()*(1-inp[:, flip_order[i]]) + p.log()*inp[:, flip_order[i]]).mean().data.item()
			ll += pixel_ll
			if ll != ll: # ll is NaN
				import pdb; pdb.set_trace()
			sampled_pixels += 1

		ll /= sampled_pixels
		return x, ll

	def create_image_multiple_orderings(self, B):
		'''
		B (number): batch size B.
		'''
		flipped_orderings = []
		for i in range(len(self.model.orderings)):
			ordering = self.model.orderings[i]
			flipped_order = flip_ordering(ordering)
			flipped_orderings.append(flipped_order)

		order_index_tracker = zeros(len(self.model.orderings)) # tracks what index of the ordering each ordering is at
		fill_track = np.zeros(784) # tracks how much of the image is filled, 1 is filled

		x = torch.zeros((B, 784), dtype=torch.float32).cuda()
		while sum(fill_track) < 784: # not entirely filled
			for i in range(len(self.model.orderings)):
				if sum(fill_track) >= 784: # if the image just got filled on the last pass
					return x

				flipped_order = flipped_orderings[i]
				ordering_index = order_index_tracker[i]

				if ordering_index == 784: # will get out of bounds error for fill_track; this ordering is done
					continue

				curr_fill_val = flipped_order[ordering_index]

				while fill_track[curr_fill_val] == 1: # if it's already filled
					order_index_tracker[i] += 1
					ordering_index = order_index_tracker[i]
					if ordering_index == 784:
						break #we are going to run into an error if we try and call index of fill_track

					curr_fill_val = flipped_order[ordering_index]

				if ordering_index == 784: # will get out of bounds error for fill_track; this ordering is done
					continue

				y = self.model(x)
				logits = y[:, curr_fill_val]
				B = Bernoulli(torch.sigmoid(logits))
				sample = B.sample()
				x[:, curr_fill_val] = sample

				fill_track[curr_fill_val] = 1

		return x

if __name__ == '__main__':
	I = Inference('made_mnist_4_ordering_4', '059_params.pt')
	# import pdb; pdb.set_trace()
	x = I.sample_single_ordering(16)
	vis = x.view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28)
	plt.imshow(vis.cpu().numpy(), cmap='gray')
	plt.show()
